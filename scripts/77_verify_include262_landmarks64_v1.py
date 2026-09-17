from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
META = ROOT / "metadata"
RESULTS = ROOT / "results"

MANIFEST_CSV = META / "include262_landmarks64_manifest_v1.csv"
FINAL_META_CSV = META / "include262_official_split_final.csv"

OUTPUT_CSV = META / "include262_landmarks64_verification_v1.csv"
FAILED_CSV = META / "include262_landmarks64_verification_failed_v1.csv"
SUMMARY_TXT = RESULTS / "include262_landmarks64_verification_v1_summary.txt"

EXPECTED_FRAMES = 64
EXPECTED_LEFT_SHAPE = (64, 21, 3)
EXPECTED_RIGHT_SHAPE = (64, 21, 3)
EXPECTED_POSE_SHAPE = (64, 33, 4)
EXPECTED_FACE_POINTS = 92
EXPECTED_FACE_SHAPE = (64, EXPECTED_FACE_POINTS, 3)
EXPECTED_COMBINED_DIM = 534
EXPECTED_COMBINED_SHAPE = (64, EXPECTED_COMBINED_DIM)


def check_npz(row):
    feature_path = Path(str(row["feature_path"]))

    result = {
        "source_index": int(row["source_index"]),
        "video_name": row["video_name"],
        "label": row["label"],
        "class_id": int(row["class_id"]),
        "split": row["split"],
        "feature_path": str(feature_path),
        "exists": feature_path.exists(),
        "load_ok": False,
        "shape_ok": False,
        "finite_ok": False,
        "metadata_ok": False,
        "verification_ok": False,
        "reason": "",
        "left_nonzero_rate": np.nan,
        "right_nonzero_rate": np.nan,
        "pose_nonzero_rate": np.nan,
        "face_nonzero_rate": np.nan,
    }

    if not feature_path.exists():
        result["reason"] = "feature_file_missing"
        return result

    try:
        data = np.load(feature_path, allow_pickle=False)
    except Exception as e:
        result["reason"] = f"npz_load_failed:{type(e).__name__}"
        return result

    required = {
        "left_hand", "right_hand", "pose", "face_selected", "face_indices",
        "combined_raw", "left_detected", "right_detected", "left_rescued",
        "right_rescued", "pose_detected", "face_detected", "label",
        "class_id", "split", "source_index",
    }

    missing = sorted(required - set(data.files))
    if missing:
        result["reason"] = "missing_arrays:" + ",".join(missing)
        return result

    result["load_ok"] = True

    left = data["left_hand"]
    right = data["right_hand"]
    pose = data["pose"]
    face = data["face_selected"]
    face_ids = data["face_indices"]
    combined = data["combined_raw"]

    shapes_ok = (
        left.shape == EXPECTED_LEFT_SHAPE
        and right.shape == EXPECTED_RIGHT_SHAPE
        and pose.shape == EXPECTED_POSE_SHAPE
        and face.shape == EXPECTED_FACE_SHAPE
        and combined.shape == EXPECTED_COMBINED_SHAPE
        and face_ids.shape == (EXPECTED_FACE_POINTS,)
        and data["left_detected"].shape == (EXPECTED_FRAMES,)
        and data["right_detected"].shape == (EXPECTED_FRAMES,)
        and data["left_rescued"].shape == (EXPECTED_FRAMES,)
        and data["right_rescued"].shape == (EXPECTED_FRAMES,)
        and data["pose_detected"].shape == (EXPECTED_FRAMES,)
        and data["face_detected"].shape == (EXPECTED_FRAMES,)
    )
    result["shape_ok"] = bool(shapes_ok)

    arrays_to_check = [left, right, pose, face, combined]
    finite_ok = all(np.isfinite(a).all() for a in arrays_to_check)
    result["finite_ok"] = bool(finite_ok)

    stored_label = str(data["label"].item())
    stored_class = int(data["class_id"].item())
    stored_split = str(data["split"].item())
    stored_source = int(data["source_index"].item())

    metadata_ok = (
        stored_label == str(row["label"])
        and stored_class == int(row["class_id"])
        and stored_split == str(row["split"])
        and stored_source == int(row["source_index"])
    )
    result["metadata_ok"] = bool(metadata_ok)

    result["left_nonzero_rate"] = float(
        np.any(np.abs(left) > 1e-8, axis=(1, 2)).mean()
    )
    result["right_nonzero_rate"] = float(
        np.any(np.abs(right) > 1e-8, axis=(1, 2)).mean()
    )
    result["pose_nonzero_rate"] = float(
        np.any(np.abs(pose[:, :, :3]) > 1e-8, axis=(1, 2)).mean()
    )
    result["face_nonzero_rate"] = float(
        np.any(np.abs(face) > 1e-8, axis=(1, 2)).mean()
    )

    if not shapes_ok:
        result["reason"] = (
            f"shape_mismatch:left={left.shape},right={right.shape},"
            f"pose={pose.shape},face={face.shape},combined={combined.shape}"
        )
        return result

    if not finite_ok:
        result["reason"] = "nan_or_inf_detected"
        return result

    if not metadata_ok:
        result["reason"] = "stored_metadata_mismatch"
        return result

    result["verification_ok"] = True
    result["reason"] = "ok"
    return result


def main():
    print("=" * 86)
    print("INCLUDE STEP 15 - VERIFY ALL 64-FRAME MEDIAPIPE FEATURES")
    print("=" * 86)

    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Manifest not found:\n{MANIFEST_CSV}")

    if not FINAL_META_CSV.exists():
        raise FileNotFoundError(f"Final metadata not found:\n{FINAL_META_CSV}")

    RESULTS.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_CSV)
    final_meta = pd.read_csv(FINAL_META_CSV)

    print(f"Manifest rows      : {len(manifest)}")
    print(f"Final metadata rows: {len(final_meta)}")
    print(f"Manifest classes   : {manifest['class_id'].nunique()}")
    print(f"Manifest splits    : {manifest['split'].value_counts().to_dict()}")
    print()

    rows = []

    for i, row in manifest.iterrows():
        result = check_npz(row)
        rows.append(result)

        done = i + 1
        if done % 250 == 0 or done == len(manifest):
            good = sum(r["verification_ok"] for r in rows)
            bad = done - good
            print(f"Verified {done}/{len(manifest)} | OK={good} | Failed={bad}")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)

    failed = out[~out["verification_ok"]].copy()
    failed.to_csv(FAILED_CSV, index=False)

    valid = out[out["verification_ok"]].copy()

    total_ok = int(valid.shape[0])
    total_bad = int(failed.shape[0])

    represented_classes = valid["class_id"].nunique() if not valid.empty else 0
    split_counts = valid["split"].value_counts().to_dict() if not valid.empty else {}

    source_unique = int(manifest["source_index"].nunique())
    feature_path_unique = int(manifest["feature_path"].nunique())

    manifest_status_bad = int((manifest["status"] != "ok").sum())
    manifest_dup_source = int(manifest["source_index"].duplicated().sum())
    manifest_dup_feature = int(manifest["feature_path"].duplicated().sum())

    mean_left = float(valid["left_nonzero_rate"].mean()) if not valid.empty else 0.0
    mean_right = float(valid["right_nonzero_rate"].mean()) if not valid.empty else 0.0
    mean_pose = float(valid["pose_nonzero_rate"].mean()) if not valid.empty else 0.0
    mean_face = float(valid["face_nonzero_rate"].mean()) if not valid.empty else 0.0

    both_hand_per_video = np.minimum(
        valid["left_nonzero_rate"].to_numpy(),
        valid["right_nonzero_rate"].to_numpy(),
    ) if not valid.empty else np.array([])

    mean_both_proxy = float(both_hand_per_video.mean()) if len(both_hand_per_video) else 0.0

    zero_left_videos = int((valid["left_nonzero_rate"] == 0).sum()) if not valid.empty else 0
    zero_right_videos = int((valid["right_nonzero_rate"] == 0).sum()) if not valid.empty else 0
    zero_pose_videos = int((valid["pose_nonzero_rate"] == 0).sum()) if not valid.empty else 0
    zero_face_videos = int((valid["face_nonzero_rate"] == 0).sum()) if not valid.empty else 0

    print("\n" + "=" * 86)
    print("FINAL FEATURE VERIFICATION")
    print("=" * 86)
    print(f"Verified OK               : {total_ok}")
    print(f"Failed                    : {total_bad}")
    print(f"Represented classes       : {represented_classes} / 262")
    print(f"Split counts              : {split_counts}")
    print(f"Unique source_index       : {source_unique}")
    print(f"Unique feature paths      : {feature_path_unique}")
    print(f"Manifest non-ok rows      : {manifest_status_bad}")
    print(f"Duplicate source_index    : {manifest_dup_source}")
    print(f"Duplicate feature paths   : {manifest_dup_feature}")
    print()
    print(f"Mean left-hand frame rate : {mean_left:.3f}")
    print(f"Mean right-hand frame rate: {mean_right:.3f}")
    print(f"Mean both-hands proxy     : {mean_both_proxy:.3f}")
    print(f"Mean pose frame rate      : {mean_pose:.3f}")
    print(f"Mean face frame rate      : {mean_face:.3f}")
    print()
    print(f"Videos with zero left hand : {zero_left_videos}")
    print(f"Videos with zero right hand: {zero_right_videos}")
    print(f"Videos with zero pose      : {zero_pose_videos}")
    print(f"Videos with zero face      : {zero_face_videos}")

    expected_split_counts = {"train": 3121, "val": 347, "test": 816}

    success = (
        len(manifest) == 4284
        and len(final_meta) == 4284
        and total_ok == 4284
        and total_bad == 0
        and represented_classes == 262
        and split_counts == expected_split_counts
        and source_unique == 4284
        and feature_path_unique == 4284
        and manifest_status_bad == 0
        and manifest_dup_source == 0
        and manifest_dup_feature == 0
    )

    summary = [
        "=" * 86,
        "INCLUDE 262 LANDMARK64 V1 VERIFICATION SUMMARY",
        "=" * 86,
        "",
        f"Verified OK: {total_ok}",
        f"Failed: {total_bad}",
        f"Represented classes: {represented_classes} / 262",
        f"Split counts: {split_counts}",
        "",
        f"Mean left-hand frame rate: {mean_left:.3f}",
        f"Mean right-hand frame rate: {mean_right:.3f}",
        f"Mean both-hands proxy: {mean_both_proxy:.3f}",
        f"Mean pose frame rate: {mean_pose:.3f}",
        f"Mean face frame rate: {mean_face:.3f}",
        "",
        f"Videos with zero left hand: {zero_left_videos}",
        f"Videos with zero right hand: {zero_right_videos}",
        f"Videos with zero pose: {zero_pose_videos}",
        f"Videos with zero face: {zero_face_videos}",
        "",
        f"Verification report: {OUTPUT_CSV}",
        f"Failed-only report: {FAILED_CSV}",
        "",
        "STATUS:",
    ]

    if success:
        summary.append("ALL 4284 FEATURE FILES VERIFIED ✅")
        summary.append(
            "NEXT: compute TRAIN-ONLY normalization and build motion/geometry features."
        )
    else:
        summary.append("FEATURE VERIFICATION FAILED ⚠️")
        summary.append("Review failed-only CSV before continuing.")

    SUMMARY_TXT.write_text("\n".join(summary), encoding="utf-8")

    print("\n" + "\n".join(summary))
    print("=" * 86)


if __name__ == "__main__":
    main()