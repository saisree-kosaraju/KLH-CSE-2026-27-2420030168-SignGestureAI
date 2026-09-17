from pathlib import Path
import argparse
import re
import time

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


ROOT = Path.cwd()
META = ROOT / "metadata"
PROCESSED = ROOT / "processed_data" / "INCLUDE262" / "landmarks64_v1"
INPUT_CSV = META / "include262_official_split_final.csv"
MANIFEST_CSV = META / "include262_landmarks64_manifest_v1.csv"

NUM_FRAMES = 64
POSE_VISIBILITY_MIN = 0.35

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


def safe_name(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "video"


def selected_face_indices():
    groups = [
        mp_face_mesh.FACEMESH_LIPS,
        mp_face_mesh.FACEMESH_LEFT_EYE,
        mp_face_mesh.FACEMESH_RIGHT_EYE,
        mp_face_mesh.FACEMESH_LEFT_EYEBROW,
        mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
    ]
    ids = set()
    for connections in groups:
        for a, b in connections:
            ids.add(int(a))
            ids.add(int(b))
    return np.array(sorted(ids), dtype=np.int32)


FACE_IDS = selected_face_indices()


def make_output_path(row, row_index):
    stem = Path(str(row["video_name"])).stem
    filename = (
        f"{row_index:05d}_"
        f"c{int(row['class_id']):03d}_"
        f"{safe_name(row['label'])}_"
        f"{safe_name(stem)}.npz"
    )
    return PROCESSED / str(row["split"]) / filename


def sample_64_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "cannot_open"

    reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if reported_count <= 0:
        cap.release()
        return None, "invalid_frame_count"

    targets = np.linspace(
        0, max(0, reported_count - 1), NUM_FRAMES
    ).round().astype(np.int64)

    frames = []
    target_pos = 0
    frame_idx = 0
    last_good = None

    while target_pos < NUM_FRAMES:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            break

        last_good = frame

        while target_pos < NUM_FRAMES and frame_idx >= targets[target_pos]:
            frames.append(frame.copy())
            target_pos += 1

        frame_idx += 1

    cap.release()

    if last_good is None:
        return None, "no_decodable_frames"

    while len(frames) < NUM_FRAMES:
        frames.append(last_good.copy())

    return frames[:NUM_FRAMES], "ok"


def landmarks_xyz(landmark_list, expected):
    arr = np.zeros((expected, 3), dtype=np.float32)
    if landmark_list is None:
        return arr, False

    lms = landmark_list.landmark
    n = min(expected, len(lms))
    for i in range(n):
        arr[i, 0] = lms[i].x
        arr[i, 1] = lms[i].y
        arr[i, 2] = lms[i].z

    return arr, True


def pose_xyzw(landmark_list):
    arr = np.zeros((33, 4), dtype=np.float32)
    if landmark_list is None:
        return arr, False

    lms = landmark_list.landmark
    n = min(33, len(lms))
    for i in range(n):
        arr[i, 0] = lms[i].x
        arr[i, 1] = lms[i].y
        arr[i, 2] = lms[i].z
        arr[i, 3] = getattr(lms[i], "visibility", 0.0)

    return arr, True


def selected_face_xyz(landmark_list):
    arr = np.zeros((len(FACE_IDS), 3), dtype=np.float32)
    if landmark_list is None:
        return arr, False

    lms = landmark_list.landmark
    for j, idx in enumerate(FACE_IDS):
        if idx < len(lms):
            arr[j, 0] = lms[idx].x
            arr[j, 1] = lms[idx].y
            arr[j, 2] = lms[idx].z

    return arr, True


def shoulder_scale_pixels(pose, width, height):
    ls = pose[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
    rs = pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]

    if ls[3] >= POSE_VISIBILITY_MIN and rs[3] >= POSE_VISIBILITY_MIN:
        dx = (ls[0] - rs[0]) * width
        dy = (ls[1] - rs[1]) * height
        dist = float(np.sqrt(dx * dx + dy * dy))
        if dist > 1:
            return dist

    return 0.30 * min(width, height)


def rescue_hand_from_wrist(frame_bgr, pose, side, hands_model):
    height, width = frame_bgr.shape[:2]

    wrist_idx = (
        mp_holistic.PoseLandmark.LEFT_WRIST.value
        if side == "left"
        else mp_holistic.PoseLandmark.RIGHT_WRIST.value
    )

    wrist = pose[wrist_idx]
    if wrist[3] < POSE_VISIBILITY_MIN:
        return np.zeros((21, 3), dtype=np.float32), False

    cx = int(round(wrist[0] * width))
    cy = int(round(wrist[1] * height))

    body_scale = shoulder_scale_pixels(pose, width, height)
    radius = int(max(56, 0.55 * body_scale))

    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(width, cx + radius)
    y2 = min(height, cy + radius)

    if x2 - x1 < 24 or y2 - y1 < 24:
        return np.zeros((21, 3), dtype=np.float32), False

    crop = frame_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = hands_model.process(crop_rgb)

    if not result.multi_hand_landmarks:
        return np.zeros((21, 3), dtype=np.float32), False

    hand_lms = result.multi_hand_landmarks[0].landmark
    out = np.zeros((21, 3), dtype=np.float32)

    crop_w = float(x2 - x1)
    crop_h = float(y2 - y1)

    for i, lm in enumerate(hand_lms[:21]):
        px = x1 + lm.x * crop_w
        py = y1 + lm.y * crop_h

        out[i, 0] = px / width
        out[i, 1] = py / height
        out[i, 2] = lm.z

    return out, True


def flatten_frame(left, right, pose, face):
    return np.concatenate(
        [
            left.reshape(-1),
            right.reshape(-1),
            pose.reshape(-1),
            face.reshape(-1),
        ]
    ).astype(np.float32)


def process_video(video_path, holistic_model, hands_model):
    frames, status = sample_64_frames(video_path)
    if frames is None:
        return None, {"status": status}

    left_seq = np.zeros((NUM_FRAMES, 21, 3), dtype=np.float32)
    right_seq = np.zeros((NUM_FRAMES, 21, 3), dtype=np.float32)
    pose_seq = np.zeros((NUM_FRAMES, 33, 4), dtype=np.float32)
    face_seq = np.zeros((NUM_FRAMES, len(FACE_IDS), 3), dtype=np.float32)

    left_detected = np.zeros(NUM_FRAMES, dtype=np.uint8)
    right_detected = np.zeros(NUM_FRAMES, dtype=np.uint8)
    left_rescued = np.zeros(NUM_FRAMES, dtype=np.uint8)
    right_rescued = np.zeros(NUM_FRAMES, dtype=np.uint8)
    pose_detected = np.zeros(NUM_FRAMES, dtype=np.uint8)
    face_detected = np.zeros(NUM_FRAMES, dtype=np.uint8)

    for t, frame_bgr in enumerate(frames):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = holistic_model.process(rgb)

        left, left_ok = landmarks_xyz(result.left_hand_landmarks, 21)
        right, right_ok = landmarks_xyz(result.right_hand_landmarks, 21)
        pose, pose_ok = pose_xyzw(result.pose_landmarks)
        face, face_ok = selected_face_xyz(result.face_landmarks)

        if pose_ok:
            pose_detected[t] = 1
        if face_ok:
            face_detected[t] = 1
        if left_ok:
            left_detected[t] = 1
        if right_ok:
            right_detected[t] = 1

        if (not left_ok) and pose_ok:
            rescued, ok = rescue_hand_from_wrist(
                frame_bgr, pose, "left", hands_model
            )
            if ok:
                left = rescued
                left_rescued[t] = 1

        if (not right_ok) and pose_ok:
            rescued, ok = rescue_hand_from_wrist(
                frame_bgr, pose, "right", hands_model
            )
            if ok:
                right = rescued
                right_rescued[t] = 1

        left_seq[t] = left
        right_seq[t] = right
        pose_seq[t] = pose
        face_seq[t] = face

    combined = np.stack(
        [
            flatten_frame(left_seq[t], right_seq[t], pose_seq[t], face_seq[t])
            for t in range(NUM_FRAMES)
        ],
        axis=0,
    )

    left_final = np.maximum(left_detected, left_rescued)
    right_final = np.maximum(right_detected, right_rescued)

    stats = {
        "status": "ok",
        "left_detect_rate_raw": float(left_detected.mean()),
        "right_detect_rate_raw": float(right_detected.mean()),
        "left_rescue_rate": float(left_rescued.mean()),
        "right_rescue_rate": float(right_rescued.mean()),
        "left_final_rate": float(left_final.mean()),
        "right_final_rate": float(right_final.mean()),
        "both_hands_final_rate": float(
            np.logical_and(left_final, right_final).mean()
        ),
        "pose_rate": float(pose_detected.mean()),
        "face_rate": float(face_detected.mean()),
        "frames": NUM_FRAMES,
        "feature_dim": int(combined.shape[1]),
    }

    arrays = {
        "left_hand": left_seq,
        "right_hand": right_seq,
        "pose": pose_seq,
        "face_selected": face_seq,
        "face_indices": FACE_IDS,
        "combined_raw": combined,
        "left_detected": left_detected,
        "right_detected": right_detected,
        "left_rescued": left_rescued,
        "right_rescued": right_rescued,
        "pose_detected": pose_detected,
        "face_detected": face_detected,
    }

    return arrays, stats


def load_existing_manifest():
    if not MANIFEST_CSV.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(MANIFEST_CSV)
    except Exception:
        return pd.DataFrame()


def save_manifest(rows):
    pd.DataFrame(rows).to_csv(MANIFEST_CSV, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only N not-yet-processed videos for a test run.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    print("=" * 88)
    print("INCLUDE STEP 14 - 64-FRAME MEDIAPIPE FEATURE EXTRACTION V1")
    print("Hands + Pose + Selected Face + Pose-Guided Hand Rescue")
    print("=" * 88)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Final INCLUDE metadata not found:\n{INPUT_CSV}")

    PROCESSED.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    if args.split != "all":
        df = df[df["split"] == args.split].copy()

    df = df.reset_index().rename(columns={"index": "source_index"})

    print(f"Videos selected  : {len(df)}")
    print(f"Frames/video     : {NUM_FRAMES}")
    print(f"Selected face pts: {len(FACE_IDS)}")
    print(
        "Raw feature dim  : "
        f"{21*3 + 21*3 + 33*4 + len(FACE_IDS)*3}"
    )
    print()

    existing_manifest = load_existing_manifest()
    manifest_by_source = {}

    if not existing_manifest.empty and "source_index" in existing_manifest.columns:
        for _, r in existing_manifest.iterrows():
            manifest_by_source[int(r["source_index"])] = r.to_dict()

    started = time.time()
    processed_this_run = 0
    failures_this_run = 0

    holistic_kwargs = dict(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    hands_kwargs = dict(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.40,
        min_tracking_confidence=0.40,
    )

    with mp_holistic.Holistic(**holistic_kwargs) as holistic_model, \
         mp_hands.Hands(**hands_kwargs) as hands_model:

        for _, row in df.iterrows():
            source_index = int(row["source_index"])
            output_path = make_output_path(row, source_index)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            already_ok = (
                output_path.exists()
                and source_index in manifest_by_source
                and str(manifest_by_source[source_index].get("status", "")) == "ok"
            )

            if already_ok and not args.overwrite:
                continue

            if args.limit is not None and processed_this_run >= args.limit:
                break

            t0 = time.time()

            arrays, stats = process_video(
                row["video_path"],
                holistic_model,
                hands_model,
            )

            elapsed = time.time() - t0

            if arrays is None:
                failures_this_run += 1
                record = {
                    "source_index": source_index,
                    "video_name": row["video_name"],
                    "video_path": row["video_path"],
                    "label": row["label"],
                    "class_id": int(row["class_id"]),
                    "split": row["split"],
                    "feature_path": str(output_path),
                    "elapsed_sec": elapsed,
                    **stats,
                }
            else:
                np.savez_compressed(
                    output_path,
                    **arrays,
                    label=np.array(str(row["label"])),
                    class_id=np.array(int(row["class_id"]), dtype=np.int32),
                    split=np.array(str(row["split"])),
                    source_index=np.array(source_index, dtype=np.int32),
                )

                record = {
                    "source_index": source_index,
                    "video_name": row["video_name"],
                    "video_path": row["video_path"],
                    "label": row["label"],
                    "class_id": int(row["class_id"]),
                    "split": row["split"],
                    "feature_path": str(output_path.resolve()),
                    "elapsed_sec": elapsed,
                    **stats,
                }

            manifest_by_source[source_index] = record
            processed_this_run += 1

            rows_to_save = [
                manifest_by_source[k]
                for k in sorted(manifest_by_source)
            ]
            save_manifest(rows_to_save)

            if processed_this_run % 10 == 0 or processed_this_run == 1:
                elapsed_total = (time.time() - started) / 60.0
                good_this_run = processed_this_run - failures_this_run

                print(
                    f"Processed this run {processed_this_run} | "
                    f"OK={good_this_run} | "
                    f"Failed={failures_this_run} | "
                    f"Last={elapsed:.1f}s | "
                    f"Time={elapsed_total:.1f} min"
                )

    manifest = load_existing_manifest()

    if not manifest.empty:
        ok = manifest[manifest["status"] == "ok"].copy()
        failed = manifest[manifest["status"] != "ok"].copy()

        print("\n" + "=" * 88)
        print("CURRENT EXTRACTION STATUS")
        print("=" * 88)
        print(f"Manifest rows       : {len(manifest)}")
        print(f"Successful features : {len(ok)}")
        print(f"Failed              : {len(failed)}")

        if not ok.empty:
            print(f"Mean left final rate : {ok['left_final_rate'].mean():.3f}")
            print(f"Mean right final rate: {ok['right_final_rate'].mean():.3f}")
            print(f"Mean both-hands rate : {ok['both_hands_final_rate'].mean():.3f}")
            print(f"Mean pose rate       : {ok['pose_rate'].mean():.3f}")
            print(f"Mean face rate       : {ok['face_rate'].mean():.3f}")

    print("\nFeature folder:")
    print(PROCESSED)
    print("\nManifest:")
    print(MANIFEST_CSV)

    if args.limit is not None:
        print(
            "\nTEST RUN COMPLETE. If the detection rates look reasonable, "
            "run the same script again without --limit to continue all videos."
        )
    else:
        print("\nExtraction run complete/resumed.")


if __name__ == "__main__":
    main()