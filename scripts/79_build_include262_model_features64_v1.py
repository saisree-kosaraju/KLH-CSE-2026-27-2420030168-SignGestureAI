from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd


ROOT = Path.cwd()
META = ROOT / "metadata"
RESULTS = ROOT / "results"

LANDMARK_MANIFEST = META / "include262_landmarks64_manifest_v1.csv"
LANDMARK_NORM = META / "include262_landmarks64_train_normalization_v1.npz"

OUT_DIR = ROOT / "processed_data" / "INCLUDE262" / "model_features64_v1"
OUT_MANIFEST = META / "include262_model_features64_manifest_v1.csv"
GEOM_NORM = META / "include262_geometry64_train_normalization_v1.npz"
SUMMARY = RESULTS / "include262_model_features64_v1_summary.txt"
SCHEMA_JSON = META / "include262_model_features64_schema_v1.json"

NUM_FRAMES = 64
EPS = 1e-6

# MediaPipe pose indices.
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

# MediaPipe hand skeleton edges.
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# 15 joint-angle triplets per hand.
ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),       # thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),       # index
    (0, 9, 10), (9, 10, 11), (10, 11, 12), # middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),# ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20) # pinky
]

FINGERTIPS = [4, 8, 12, 16, 20]

# Geometry layout.
G_LEFT_COORD = slice(0, 63)
G_RIGHT_COORD = slice(63, 126)
G_LEFT_BONES = slice(126, 146)
G_RIGHT_BONES = slice(146, 166)
G_LEFT_ANGLES = slice(166, 181)
G_RIGHT_ANGLES = slice(181, 196)
G_RELATIONS = slice(196, 217)      # 21 dims
G_LEFT_VEL = slice(217, 280)       # 63
G_RIGHT_VEL = slice(280, 343)      # 63
G_LEFT_ACC = slice(343, 406)       # 63
G_RIGHT_ACC = slice(406, 469)      # 63

GEOMETRY_DIM = 469
LANDMARK_DIM = 534
AVAILABILITY_DIM = 4


def body_reference(pose_frame):
    ls = pose_frame[LEFT_SHOULDER]
    rs = pose_frame[RIGHT_SHOULDER]

    shoulder_origin = (ls[:3] + rs[:3]) / 2.0
    dx = float(ls[0] - rs[0])
    dy = float(ls[1] - rs[1])
    scale = float(np.sqrt(dx * dx + dy * dy))

    if np.isfinite(scale) and scale > 0.03:
        return shoulder_origin.astype(np.float32), scale

    lh = pose_frame[LEFT_HIP]
    rh = pose_frame[RIGHT_HIP]
    hip_origin = (lh[:3] + rh[:3]) / 2.0

    dx = float(shoulder_origin[0] - hip_origin[0])
    dy = float(shoulder_origin[1] - hip_origin[1])
    torso = float(np.sqrt(dx * dx + dy * dy))

    if np.isfinite(torso) and torso > 0.03:
        return shoulder_origin.astype(np.float32), torso

    return shoulder_origin.astype(np.float32), 0.25


def body_normalize(points, origin, scale, subtract_z_origin=False):
    out = np.asarray(points, dtype=np.float32).copy()
    out[..., 0] = (out[..., 0] - origin[0]) / scale
    out[..., 1] = (out[..., 1] - origin[1]) / scale

    if subtract_z_origin:
        out[..., 2] = (out[..., 2] - origin[2]) / scale
    else:
        out[..., 2] = out[..., 2] / scale

    return out


def zscore_valid(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def palm_scale_xy(hand):
    wrist = hand[0, :2]
    middle_mcp = hand[9, :2]
    scale = float(np.linalg.norm(middle_mcp - wrist))

    if scale < 1e-4:
        scale = float(np.linalg.norm(hand[5, :2] - wrist))

    if scale < 1e-4:
        scale = 0.05

    return scale


def hand_centric(hand):
    scale = palm_scale_xy(hand)
    centered = hand - hand[0:1]
    centered = centered / scale
    return centered.astype(np.float32), scale


def bone_lengths(hand_centric_xyz):
    vals = []
    for a, b in HAND_EDGES:
        vals.append(
            np.linalg.norm(
                hand_centric_xyz[b] - hand_centric_xyz[a]
            )
        )
    return np.asarray(vals, dtype=np.float32)


def angle_at(a, b, c):
    v1 = a - b
    v2 = c - b

    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < EPS:
        return 0.0

    cosv = float(np.dot(v1, v2) / denom)
    cosv = float(np.clip(cosv, -1.0, 1.0))
    return float(np.arccos(cosv) / np.pi)  # normalize to [0,1]


def joint_angles(hand_centric_xyz):
    vals = []
    for a, b, c in ANGLE_TRIPLETS:
        vals.append(
            angle_at(
                hand_centric_xyz[a],
                hand_centric_xyz[b],
                hand_centric_xyz[c],
            )
        )
    return np.asarray(vals, dtype=np.float32)


def xy_distance(a, b):
    return float(np.linalg.norm(a[:2] - b[:2]))


def relation_features(
    left_body,
    right_body,
    pose_body,
    left_present,
    right_present,
    pose_present,
):
    """
    21 relation features:
      0:3   left wrist body-normalized xyz
      3:6   right wrist body-normalized xyz
      6:9   left wrist distances to nose, L shoulder, R shoulder
      9:12  right wrist distances to nose, R shoulder, L shoulder
      12:15 inter-hand wrist delta xyz (R-L)
      15    inter-hand wrist distance
      16:21 corresponding fingertip distances
    """
    out = np.zeros(21, dtype=np.float32)
    valid = np.zeros(21, dtype=bool)

    if left_present:
        out[0:3] = left_body[0]
        valid[0:3] = True

    if right_present:
        out[3:6] = right_body[0]
        valid[3:6] = True

    if left_present and pose_present:
        refs = [
            pose_body[NOSE],
            pose_body[LEFT_SHOULDER],
            pose_body[RIGHT_SHOULDER],
        ]
        out[6:9] = np.asarray(
            [xy_distance(left_body[0], r) for r in refs],
            dtype=np.float32,
        )
        valid[6:9] = True

    if right_present and pose_present:
        refs = [
            pose_body[NOSE],
            pose_body[RIGHT_SHOULDER],
            pose_body[LEFT_SHOULDER],
        ]
        out[9:12] = np.asarray(
            [xy_distance(right_body[0], r) for r in refs],
            dtype=np.float32,
        )
        valid[9:12] = True

    if left_present and right_present:
        delta = right_body[0] - left_body[0]
        out[12:15] = delta
        out[15] = xy_distance(right_body[0], left_body[0])
        valid[12:16] = True

        for j, tip in enumerate(FINGERTIPS):
            out[16 + j] = xy_distance(
                right_body[tip],
                left_body[tip],
            )
        valid[16:21] = True

    return out, valid


def build_one_feature_file(row, norms, output_path):
    with np.load(row["feature_path"], allow_pickle=False) as data:
        left = data["left_hand"].astype(np.float32)
        right = data["right_hand"].astype(np.float32)
        pose = data["pose"].astype(np.float32)
        face = data["face_selected"].astype(np.float32)

        left_present = np.maximum(
            data["left_detected"],
            data["left_rescued"],
        ).astype(bool)

        right_present = np.maximum(
            data["right_detected"],
            data["right_rescued"],
        ).astype(bool)

        pose_present = data["pose_detected"].astype(bool)
        face_present = data["face_detected"].astype(bool)

    landmark_stream = np.zeros(
        (NUM_FRAMES, LANDMARK_DIM),
        dtype=np.float32,
    )

    geometry_raw = np.zeros(
        (NUM_FRAMES, GEOMETRY_DIM),
        dtype=np.float32,
    )

    geometry_valid = np.zeros(
        (NUM_FRAMES, GEOMETRY_DIM),
        dtype=bool,
    )

    availability = np.stack(
        [
            left_present,
            right_present,
            pose_present,
            face_present,
        ],
        axis=1,
    ).astype(np.float32)

    left_hc_seq = np.zeros((NUM_FRAMES, 21, 3), dtype=np.float32)
    right_hc_seq = np.zeros((NUM_FRAMES, 21, 3), dtype=np.float32)

    left_body_seq = np.zeros((NUM_FRAMES, 21, 3), dtype=np.float32)
    right_body_seq = np.zeros((NUM_FRAMES, 21, 3), dtype=np.float32)
    pose_body_seq = np.zeros((NUM_FRAMES, 33, 3), dtype=np.float32)

    # ------------------------------------------------------------------
    # A. Per-frame body normalization + landmark stream + static geometry
    # ------------------------------------------------------------------
    for t in range(NUM_FRAMES):
        origin, scale = body_reference(pose[t])

        left_body = np.zeros((21, 3), dtype=np.float32)
        right_body = np.zeros((21, 3), dtype=np.float32)
        pose_body = np.zeros((33, 3), dtype=np.float32)
        face_body = np.zeros((92, 3), dtype=np.float32)

        if left_present[t]:
            left_body = body_normalize(
                left[t], origin, scale, subtract_z_origin=False
            )
            left_body_seq[t] = left_body

        if right_present[t]:
            right_body = body_normalize(
                right[t], origin, scale, subtract_z_origin=False
            )
            right_body_seq[t] = right_body

        if pose_present[t]:
            pose_body = body_normalize(
                pose[t, :, :3],
                origin,
                scale,
                subtract_z_origin=True,
            )
            pose_body_seq[t] = pose_body

        if face_present[t]:
            face_body = body_normalize(
                face[t], origin, scale, subtract_z_origin=False
            )

        # -------- landmark stream (534 dims) --------
        cursor = 0

        if left_present[t]:
            v = zscore_valid(
                left_body,
                norms["left_mean"],
                norms["left_std"],
            ).reshape(-1)
            landmark_stream[t, cursor:cursor + 63] = v
        cursor += 63

        if right_present[t]:
            v = zscore_valid(
                right_body,
                norms["right_mean"],
                norms["right_std"],
            ).reshape(-1)
            landmark_stream[t, cursor:cursor + 63] = v
        cursor += 63

        # Pose: 99 xyz standardized + 33 visibility values = 132.
        if pose_present[t]:
            pose_xyz_std = zscore_valid(
                pose_body,
                norms["pose_mean"],
                norms["pose_std"],
            ).reshape(-1)

            pose_vis = pose[t, :, 3].astype(np.float32)

            pose_block = np.concatenate(
                [pose_xyz_std, pose_vis],
                axis=0,
            )
            landmark_stream[t, cursor:cursor + 132] = pose_block
        cursor += 132

        if face_present[t]:
            v = zscore_valid(
                face_body,
                norms["face_mean"],
                norms["face_std"],
            ).reshape(-1)
            landmark_stream[t, cursor:cursor + 276] = v

        # -------- hand-centric static geometry --------
        if left_present[t]:
            left_hc, _ = hand_centric(left[t])
            left_hc_seq[t] = left_hc

            geometry_raw[t, G_LEFT_COORD] = left_hc.reshape(-1)
            geometry_valid[t, G_LEFT_COORD] = True

            geometry_raw[t, G_LEFT_BONES] = bone_lengths(left_hc)
            geometry_valid[t, G_LEFT_BONES] = True

            geometry_raw[t, G_LEFT_ANGLES] = joint_angles(left_hc)
            geometry_valid[t, G_LEFT_ANGLES] = True

        if right_present[t]:
            right_hc, _ = hand_centric(right[t])
            right_hc_seq[t] = right_hc

            geometry_raw[t, G_RIGHT_COORD] = right_hc.reshape(-1)
            geometry_valid[t, G_RIGHT_COORD] = True

            geometry_raw[t, G_RIGHT_BONES] = bone_lengths(right_hc)
            geometry_valid[t, G_RIGHT_BONES] = True

            geometry_raw[t, G_RIGHT_ANGLES] = joint_angles(right_hc)
            geometry_valid[t, G_RIGHT_ANGLES] = True

        rel, rel_valid = relation_features(
            left_body,
            right_body,
            pose_body,
            bool(left_present[t]),
            bool(right_present[t]),
            bool(pose_present[t]),
        )
        geometry_raw[t, G_RELATIONS] = rel
        geometry_valid[t, G_RELATIONS] = rel_valid

    # ------------------------------------------------------------------
    # B. Temporal geometry: hand-centric velocity and acceleration
    # ------------------------------------------------------------------
    for t in range(1, NUM_FRAMES):
        if left_present[t] and left_present[t - 1]:
            vel = left_hc_seq[t] - left_hc_seq[t - 1]
            geometry_raw[t, G_LEFT_VEL] = vel.reshape(-1)
            geometry_valid[t, G_LEFT_VEL] = True

        if right_present[t] and right_present[t - 1]:
            vel = right_hc_seq[t] - right_hc_seq[t - 1]
            geometry_raw[t, G_RIGHT_VEL] = vel.reshape(-1)
            geometry_valid[t, G_RIGHT_VEL] = True

    for t in range(2, NUM_FRAMES):
        if (
            left_present[t]
            and left_present[t - 1]
            and left_present[t - 2]
        ):
            v_now = left_hc_seq[t] - left_hc_seq[t - 1]
            v_prev = left_hc_seq[t - 1] - left_hc_seq[t - 2]
            acc = v_now - v_prev

            geometry_raw[t, G_LEFT_ACC] = acc.reshape(-1)
            geometry_valid[t, G_LEFT_ACC] = True

        if (
            right_present[t]
            and right_present[t - 1]
            and right_present[t - 2]
        ):
            v_now = right_hc_seq[t] - right_hc_seq[t - 1]
            v_prev = right_hc_seq[t - 1] - right_hc_seq[t - 2]
            acc = v_now - v_prev

            geometry_raw[t, G_RIGHT_ACC] = acc.reshape(-1)
            geometry_valid[t, G_RIGHT_ACC] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        landmark_stream=landmark_stream,
        geometry_raw=geometry_raw,
        geometry_valid=geometry_valid.astype(np.uint8),
        availability_mask=availability,
        label=np.array(str(row["label"])),
        class_id=np.array(int(row["class_id"]), dtype=np.int32),
        split=np.array(str(row["split"])),
        source_index=np.array(int(row["source_index"]), dtype=np.int32),
    )

    return {
        "source_index": int(row["source_index"]),
        "video_name": row["video_name"],
        "label": row["label"],
        "class_id": int(row["class_id"]),
        "split": row["split"],
        "feature_path": str(output_path.resolve()),
        "landmark_dim": LANDMARK_DIM,
        "geometry_dim": GEOMETRY_DIM,
        "left_rate": float(left_present.mean()),
        "right_rate": float(right_present.mean()),
        "pose_rate": float(pose_present.mean()),
        "face_rate": float(face_present.mean()),
        "status": "raw_ready",
    }


def output_path_for(row):
    src = int(row["source_index"])
    name = f"{src:05d}_c{int(row['class_id']):03d}_{row['split']}.npz"
    return OUT_DIR / str(row["split"]) / name


def load_landmark_norm():
    with np.load(LANDMARK_NORM, allow_pickle=False) as n:
        return {
            "left_mean": n["left_mean"].astype(np.float32),
            "left_std": n["left_std"].astype(np.float32),
            "right_mean": n["right_mean"].astype(np.float32),
            "right_std": n["right_std"].astype(np.float32),
            "pose_mean": n["pose_mean"].astype(np.float32),
            "pose_std": n["pose_std"].astype(np.float32),
            "face_mean": n["face_mean"].astype(np.float32),
            "face_std": n["face_std"].astype(np.float32),
        }


def compute_geometry_train_stats(manifest):
    print("\n" + "=" * 88)
    print("PHASE B - COMPUTE TRAIN-ONLY GEOMETRY NORMALIZATION")
    print("=" * 88)

    train = manifest[
        (manifest["split"] == "train") &
        (manifest["status"].isin(["raw_ready", "ready"]))
    ].copy()

    if len(train) != 3121:
        raise RuntimeError(
            f"Expected 3121 TRAIN feature files, found {len(train)}."
        )

    count = np.zeros(GEOMETRY_DIM, dtype=np.int64)
    total = np.zeros(GEOMETRY_DIM, dtype=np.float64)
    total_sq = np.zeros(GEOMETRY_DIM, dtype=np.float64)

    for i, row in train.reset_index(drop=True).iterrows():
        with np.load(row["feature_path"], allow_pickle=False) as data:
            x = data["geometry_raw"].astype(np.float64)
            valid = data["geometry_valid"].astype(bool)

        count += valid.sum(axis=0)

        safe_x = np.where(valid, x, 0.0)
        total += safe_x.sum(axis=0)
        total_sq += np.square(safe_x).sum(axis=0)

        done = i + 1
        if done % 500 == 0 or done == len(train):
            print(f"Geometry stats TRAIN {done}/{len(train)}")

    mean = np.zeros(GEOMETRY_DIM, dtype=np.float32)
    std = np.ones(GEOMETRY_DIM, dtype=np.float32)

    valid_dims = count > 0

    mean64 = np.zeros(GEOMETRY_DIM, dtype=np.float64)
    mean64[valid_dims] = total[valid_dims] / count[valid_dims]

    var64 = np.zeros(GEOMETRY_DIM, dtype=np.float64)
    var64[valid_dims] = (
        total_sq[valid_dims] / count[valid_dims]
        - np.square(mean64[valid_dims])
    )
    var64 = np.maximum(var64, 1e-8)

    std64 = np.sqrt(var64)
    std64 = np.where(std64 < 1e-4, 1.0, std64)

    mean[:] = mean64.astype(np.float32)
    std[:] = std64.astype(np.float32)

    np.savez_compressed(
        GEOM_NORM,
        mean=mean,
        std=std,
        valid_count=count,
        train_video_count=np.array(len(train), dtype=np.int64),
        geometry_dim=np.array(GEOMETRY_DIM, dtype=np.int32),
        version=np.array("include262_geometry64_train_only_v1"),
    )

    print(f"Saved geometry normalization:\n{GEOM_NORM}")

    return mean, std


def add_standardized_geometry(manifest, mean, std):
    print("\n" + "=" * 88)
    print("PHASE C - ADD STANDARDIZED GEOMETRY STREAM")
    print("=" * 88)

    rows = []

    for i, row in manifest.reset_index(drop=True).iterrows():
        path = Path(str(row["feature_path"]))

        with np.load(path, allow_pickle=False) as data:
            landmark_stream = data["landmark_stream"]
            geometry_raw = data["geometry_raw"]
            geometry_valid = data["geometry_valid"].astype(bool)
            availability_mask = data["availability_mask"]

            label = data["label"]
            class_id = data["class_id"]
            split = data["split"]
            source_index = data["source_index"]

        geometry_stream = np.zeros_like(
            geometry_raw,
            dtype=np.float32,
        )

        standardized = (
            geometry_raw.astype(np.float32) - mean[None, :]
        ) / std[None, :]

        geometry_stream[geometry_valid] = standardized[geometry_valid]

        np.savez_compressed(
            path,
            landmark_stream=landmark_stream.astype(np.float32),
            geometry_stream=geometry_stream.astype(np.float32),
            geometry_raw=geometry_raw.astype(np.float32),
            geometry_valid=geometry_valid.astype(np.uint8),
            availability_mask=availability_mask.astype(np.float32),
            label=label,
            class_id=class_id,
            split=split,
            source_index=source_index,
        )

        rec = row.to_dict()
        rec["status"] = "ready"
        rows.append(rec)

        done = i + 1
        if done % 500 == 0 or done == len(manifest):
            print(f"Standardized {done}/{len(manifest)}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_MANIFEST, index=False)
    return out


def verify_final(manifest):
    print("\n" + "=" * 88)
    print("PHASE D - FINAL MODEL-FEATURE VERIFICATION")
    print("=" * 88)

    failures = []

    for i, row in manifest.reset_index(drop=True).iterrows():
        path = Path(str(row["feature_path"]))

        try:
            with np.load(path, allow_pickle=False) as data:
                landmark = data["landmark_stream"]
                geometry = data["geometry_stream"]
                geom_valid = data["geometry_valid"]
                avail = data["availability_mask"]

                ok = (
                    landmark.shape == (64, LANDMARK_DIM)
                    and geometry.shape == (64, GEOMETRY_DIM)
                    and geom_valid.shape == (64, GEOMETRY_DIM)
                    and avail.shape == (64, AVAILABILITY_DIM)
                    and np.isfinite(landmark).all()
                    and np.isfinite(geometry).all()
                )

                if not ok:
                    failures.append(str(path))

        except Exception:
            failures.append(str(path))

        done = i + 1
        if done % 750 == 0 or done == len(manifest):
            print(
                f"Verified {done}/{len(manifest)} | "
                f"Failed={len(failures)}"
            )

    return failures


def write_schema():
    schema = {
        "version": "include262_model_features64_v1",
        "num_frames": 64,
        "landmark_stream_dim": LANDMARK_DIM,
        "geometry_stream_dim": GEOMETRY_DIM,
        "availability_mask_dim": AVAILABILITY_DIM,
        "landmark_stream": {
            "0:63": "left hand body-normalized standardized xyz",
            "63:126": "right hand body-normalized standardized xyz",
            "126:225": "pose body-normalized standardized xyz",
            "225:258": "pose visibility",
            "258:534": "selected face body-normalized standardized xyz",
        },
        "geometry_stream": {
            "0:63": "left hand-centric xyz",
            "63:126": "right hand-centric xyz",
            "126:146": "left normalized bone lengths",
            "146:166": "right normalized bone lengths",
            "166:181": "left joint angles/pi",
            "181:196": "right joint angles/pi",
            "196:217": "hand-body/inter-hand relations",
            "217:280": "left hand-centric velocity",
            "280:343": "right hand-centric velocity",
            "343:406": "left hand-centric acceleration",
            "406:469": "right hand-centric acceleration",
        },
        "availability_mask": [
            "left_hand_present",
            "right_hand_present",
            "pose_present",
            "face_present",
        ],
        "normalization": {
            "landmark": "body-centric + TRAIN-only mean/std",
            "geometry": "hand/body scale-normalized + TRAIN-only per-dimension mean/std",
            "validation_test_statistics_used": False,
        },
    }

    SCHEMA_JSON.write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild raw model features even if they already exist.",
    )
    args = parser.parse_args()

    print("=" * 88)
    print("INCLUDE STEP 17 - BUILD LANDMARK + HAND-CENTRIC GEOMETRY + MOTION FEATURES")
    print("=" * 88)

    if not LANDMARK_MANIFEST.exists():
        raise FileNotFoundError(
            f"Landmark manifest not found:\n{LANDMARK_MANIFEST}"
        )

    if not LANDMARK_NORM.exists():
        raise FileNotFoundError(
            f"Landmark normalization not found:\n{LANDMARK_NORM}"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(LANDMARK_MANIFEST)

    source = source[source["status"] == "ok"].copy()

    if len(source) != 4284:
        raise RuntimeError(
            f"Expected 4284 landmark feature rows, found {len(source)}."
        )

    print(f"Input videos      : {len(source)}")
    print(f"Classes           : {source['class_id'].nunique()}")
    print(f"Splits            : {source['split'].value_counts().to_dict()}")
    print(f"Landmark dim      : {LANDMARK_DIM}")
    print(f"Geometry dim      : {GEOMETRY_DIM}")
    print(f"Frames/video      : {NUM_FRAMES}")
    print()

    norms = load_landmark_norm()

    # ------------------------------------------------------------------
    # Phase A: build raw model feature files (resumable)
    # ------------------------------------------------------------------
    print("=" * 88)
    print("PHASE A - BUILD RAW MODEL FEATURE FILES")
    print("=" * 88)

    old_manifest = pd.DataFrame()
    if OUT_MANIFEST.exists():
        try:
            old_manifest = pd.read_csv(OUT_MANIFEST)
        except Exception:
            old_manifest = pd.DataFrame()

    old_by_source = {}
    if not old_manifest.empty and "source_index" in old_manifest.columns:
        for _, r in old_manifest.iterrows():
            old_by_source[int(r["source_index"])] = r.to_dict()

    records = {}
    started = time.time()
    built_this_run = 0

    for i, row in source.reset_index(drop=True).iterrows():
        source_index = int(row["source_index"])
        out_path = output_path_for(row)

        existing = old_by_source.get(source_index)

        if (
            not args.rebuild
            and existing is not None
            and out_path.exists()
            and str(existing.get("status", "")) in {"raw_ready", "ready"}
        ):
            records[source_index] = existing
            continue

        rec = build_one_feature_file(
            row,
            norms,
            out_path,
        )

        records[source_index] = rec
        built_this_run += 1

        if built_this_run % 100 == 0 or built_this_run == 1:
            elapsed = (time.time() - started) / 60.0
            print(
                f"Built this run {built_this_run} | "
                f"Source progress {i+1}/{len(source)} | "
                f"Time={elapsed:.1f} min"
            )

        if built_this_run % 25 == 0:
            pd.DataFrame(
                [records[k] for k in sorted(records)]
            ).to_csv(OUT_MANIFEST, index=False)

    manifest = pd.DataFrame(
        [records[k] for k in sorted(records)]
    )

    # If there were skipped rows not yet added to records because of a prior
    # partial manifest mismatch, recover them from the source/output files.
    if len(manifest) != 4284:
        missing_sources = set(source["source_index"].astype(int)) - set(
            manifest["source_index"].astype(int)
        )

        for source_index in sorted(missing_sources):
            row = source[
                source["source_index"].astype(int) == source_index
            ].iloc[0]

            out_path = output_path_for(row)

            if not out_path.exists():
                rec = build_one_feature_file(
                    row,
                    norms,
                    out_path,
                )
            else:
                rec = {
                    "source_index": int(row["source_index"]),
                    "video_name": row["video_name"],
                    "label": row["label"],
                    "class_id": int(row["class_id"]),
                    "split": row["split"],
                    "feature_path": str(out_path.resolve()),
                    "landmark_dim": LANDMARK_DIM,
                    "geometry_dim": GEOMETRY_DIM,
                    "left_rate": float(row["left_final_rate"]),
                    "right_rate": float(row["right_final_rate"]),
                    "pose_rate": float(row["pose_rate"]),
                    "face_rate": float(row["face_rate"]),
                    "status": "raw_ready",
                }

            records[source_index] = rec

        manifest = pd.DataFrame(
            [records[k] for k in sorted(records)]
        )

    manifest.to_csv(OUT_MANIFEST, index=False)

    print(f"\nRaw model feature files ready: {len(manifest)}")

    if len(manifest) != 4284:
        raise RuntimeError(
            f"Expected 4284 raw model feature files, found {len(manifest)}."
        )

    # ------------------------------------------------------------------
    # Phase B: TRAIN-only geometry statistics
    # ------------------------------------------------------------------
    geom_mean, geom_std = compute_geometry_train_stats(manifest)

    # ------------------------------------------------------------------
    # Phase C: standardized geometry
    # ------------------------------------------------------------------
    manifest = add_standardized_geometry(
        manifest,
        geom_mean,
        geom_std,
    )

    # ------------------------------------------------------------------
    # Phase D: verification
    # ------------------------------------------------------------------
    failures = verify_final(manifest)

    write_schema()

    split_counts = manifest["split"].value_counts().to_dict()
    represented = manifest["class_id"].nunique()
    ready_count = int((manifest["status"] == "ready").sum())

    print("\n" + "=" * 88)
    print("FINAL MODEL FEATURE SUMMARY")
    print("=" * 88)
    print(f"Feature files ready : {ready_count} / 4284")
    print(f"Verification failures: {len(failures)}")
    print(f"Represented classes : {represented} / 262")
    print(f"Split counts        : {split_counts}")
    print(f"Landmark stream     : 64 x {LANDMARK_DIM}")
    print(f"Geometry stream     : 64 x {GEOMETRY_DIM}")
    print(f"Availability mask   : 64 x {AVAILABILITY_DIM}")
    print()
    print(f"Output folder:\n{OUT_DIR}")
    print(f"\nManifest:\n{OUT_MANIFEST}")
    print(f"\nGeometry normalization:\n{GEOM_NORM}")
    print(f"\nFeature schema:\n{SCHEMA_JSON}")

    success = (
        ready_count == 4284
        and len(failures) == 0
        and represented == 262
        and split_counts == {
            "train": 3121,
            "test": 816,
            "val": 347,
        }
    )

    summary_lines = [
        "=" * 88,
        "INCLUDE 262 MODEL FEATURES64 V1 SUMMARY",
        "=" * 88,
        "",
        f"Feature files ready: {ready_count} / 4284",
        f"Verification failures: {len(failures)}",
        f"Represented classes: {represented} / 262",
        f"Split counts: {split_counts}",
        "",
        f"Landmark stream shape: 64 x {LANDMARK_DIM}",
        f"Geometry stream shape: 64 x {GEOMETRY_DIM}",
        f"Availability mask shape: 64 x {AVAILABILITY_DIM}",
        "",
        "Landmark branch:",
        "  normalized hands + pose + selected face",
        "",
        "Geometry/motion branch:",
        "  hand-centric coordinates",
        "  normalized bone lengths",
        "  joint angles",
        "  hand-to-body relations",
        "  inter-hand relations",
        "  velocity",
        "  acceleration",
        "",
        "Leakage control:",
        "  landmark statistics = TRAIN only",
        "  geometry statistics = TRAIN only",
        "  validation/test statistics used = 0",
        "",
        f"Output folder: {OUT_DIR}",
        f"Manifest: {OUT_MANIFEST}",
        f"Geometry normalization: {GEOM_NORM}",
        f"Schema: {SCHEMA_JSON}",
        "",
        "STATUS:",
    ]

    if success:
        summary_lines.append("MODEL FEATURES READY ✅")
        summary_lines.append(
            "NEXT: train the two-stream Conv1D + HTMA INCLUDE262 baseline."
        )
    else:
        summary_lines.append("MODEL FEATURE BUILD FAILED ⚠️")

    SUMMARY.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n" + "\n".join(summary_lines))
    print("=" * 88)

    if not success:
        raise RuntimeError(
            "Model-feature build did not pass final validation."
        )


if __name__ == "__main__":
    main()