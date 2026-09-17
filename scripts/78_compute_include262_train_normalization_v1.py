from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
META = ROOT / "metadata"
RESULTS = ROOT / "results"

MANIFEST_CSV = META / "include262_landmarks64_manifest_v1.csv"
OUTPUT_NPZ = META / "include262_landmarks64_train_normalization_v1.npz"
SUMMARY_TXT = RESULTS / "include262_landmarks64_train_normalization_v1_summary.txt"

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

EPS = 1e-6


class RunningStats:
    def __init__(self, shape):
        self.count = 0
        self.sum = np.zeros(shape, dtype=np.float64)
        self.sumsq = np.zeros(shape, dtype=np.float64)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.sum.shape):
            x = x[None, ...]

        self.count += x.shape[0]
        self.sum += x.sum(axis=0)
        self.sumsq += np.square(x).sum(axis=0)

    def finalize(self):
        if self.count == 0:
            mean = np.zeros_like(self.sum, dtype=np.float32)
            std = np.ones_like(self.sum, dtype=np.float32)
            return mean, std

        mean = self.sum / self.count
        var = self.sumsq / self.count - np.square(mean)
        var = np.maximum(var, 1e-8)
        std = np.sqrt(var)

        # Avoid exploding values for nearly constant dimensions.
        std = np.where(std < 1e-4, 1.0, std)

        return mean.astype(np.float32), std.astype(np.float32)


def body_reference(pose_frame):
    """
    Returns:
      origin_xyz: shoulder midpoint in MediaPipe pose coordinate system
      scale: shoulder distance in normalized image coordinates
      used_fallback: whether hip/constant fallback was required
    """
    ls = pose_frame[LEFT_SHOULDER]
    rs = pose_frame[RIGHT_SHOULDER]

    shoulder_origin = (ls[:3] + rs[:3]) / 2.0

    dx = float(ls[0] - rs[0])
    dy = float(ls[1] - rs[1])
    scale = float(np.sqrt(dx * dx + dy * dy))

    if np.isfinite(scale) and scale > 0.03:
        return shoulder_origin.astype(np.float32), scale, False

    # Fallback: torso length from shoulder midpoint to hip midpoint.
    lh = pose_frame[LEFT_HIP]
    rh = pose_frame[RIGHT_HIP]
    hip_origin = (lh[:3] + rh[:3]) / 2.0

    dx = float(shoulder_origin[0] - hip_origin[0])
    dy = float(shoulder_origin[1] - hip_origin[1])
    torso_scale = float(np.sqrt(dx * dx + dy * dy))

    if np.isfinite(torso_scale) and torso_scale > 0.03:
        return shoulder_origin.astype(np.float32), torso_scale, True

    # Last-resort fallback. This should be rare because pose detection is 100%.
    return shoulder_origin.astype(np.float32), 0.25, True


def normalize_xy_and_scale_z(points, origin_xyz, scale, subtract_z_origin=False):
    """
    Body-centric normalization:
      x,y -> subtract shoulder midpoint and divide by shoulder scale
      z   -> divide by shoulder scale
    For pose only, z also subtracts the shoulder-midpoint pose z.
    """
    out = np.asarray(points, dtype=np.float32).copy()

    out[..., 0] = (out[..., 0] - origin_xyz[0]) / scale
    out[..., 1] = (out[..., 1] - origin_xyz[1]) / scale

    if subtract_z_origin:
        out[..., 2] = (out[..., 2] - origin_xyz[2]) / scale
    else:
        out[..., 2] = out[..., 2] / scale

    return out


def main():
    print("=" * 88)
    print("INCLUDE STEP 16 - COMPUTE TRAIN-ONLY BODY-CENTRIC NORMALIZATION")
    print("=" * 88)

    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Manifest not found:\n{MANIFEST_CSV}")

    RESULTS.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_CSV)

    train = manifest[
        (manifest["split"] == "train") &
        (manifest["status"] == "ok")
    ].copy()

    print(f"All manifest rows : {len(manifest)}")
    print(f"TRAIN rows used   : {len(train)}")
    print(f"TRAIN classes     : {train['class_id'].nunique()}")
    print("Validation/test rows used for statistics: 0")
    print()

    if len(train) != 3121:
        raise RuntimeError(
            f"Expected 3121 TRAIN videos, found {len(train)}."
        )

    left_stats = RunningStats((21, 3))
    right_stats = RunningStats((21, 3))
    pose_stats = RunningStats((33, 3))
    face_stats = RunningStats((92, 3))

    left_valid_frames = 0
    right_valid_frames = 0
    pose_valid_frames = 0
    face_valid_frames = 0

    total_frames = 0
    body_fallback_frames = 0

    body_scales = []

    for i, row in train.reset_index(drop=True).iterrows():
        feature_path = Path(str(row["feature_path"]))

        with np.load(feature_path, allow_pickle=False) as data:
            left = data["left_hand"].astype(np.float32)
            right = data["right_hand"].astype(np.float32)
            pose = data["pose"].astype(np.float32)
            face = data["face_selected"].astype(np.float32)

            left_detected = np.maximum(
                data["left_detected"],
                data["left_rescued"]
            ).astype(bool)

            right_detected = np.maximum(
                data["right_detected"],
                data["right_rescued"]
            ).astype(bool)

            pose_detected = data["pose_detected"].astype(bool)
            face_detected = data["face_detected"].astype(bool)

        for t in range(left.shape[0]):
            total_frames += 1

            origin, scale, fallback = body_reference(pose[t])

            body_scales.append(scale)

            if fallback:
                body_fallback_frames += 1

            if pose_detected[t]:
                pose_norm = normalize_xy_and_scale_z(
                    pose[t, :, :3],
                    origin,
                    scale,
                    subtract_z_origin=True,
                )
                pose_stats.update(pose_norm)
                pose_valid_frames += 1

            if face_detected[t]:
                face_norm = normalize_xy_and_scale_z(
                    face[t],
                    origin,
                    scale,
                    subtract_z_origin=False,
                )
                face_stats.update(face_norm)
                face_valid_frames += 1

            if left_detected[t]:
                left_norm = normalize_xy_and_scale_z(
                    left[t],
                    origin,
                    scale,
                    subtract_z_origin=False,
                )
                left_stats.update(left_norm)
                left_valid_frames += 1

            if right_detected[t]:
                right_norm = normalize_xy_and_scale_z(
                    right[t],
                    origin,
                    scale,
                    subtract_z_origin=False,
                )
                right_stats.update(right_norm)
                right_valid_frames += 1

        done = i + 1
        if done % 250 == 0 or done == len(train):
            print(
                f"Processed TRAIN {done}/{len(train)} | "
                f"Frames={total_frames}"
            )

    left_mean, left_std = left_stats.finalize()
    right_mean, right_std = right_stats.finalize()
    pose_mean, pose_std = pose_stats.finalize()
    face_mean, face_std = face_stats.finalize()

    body_scales = np.asarray(body_scales, dtype=np.float32)

    np.savez_compressed(
        OUTPUT_NPZ,
        left_mean=left_mean,
        left_std=left_std,
        right_mean=right_mean,
        right_std=right_std,
        pose_mean=pose_mean,
        pose_std=pose_std,
        face_mean=face_mean,
        face_std=face_std,
        left_valid_frames=np.array(left_valid_frames, dtype=np.int64),
        right_valid_frames=np.array(right_valid_frames, dtype=np.int64),
        pose_valid_frames=np.array(pose_valid_frames, dtype=np.int64),
        face_valid_frames=np.array(face_valid_frames, dtype=np.int64),
        total_train_frames=np.array(total_frames, dtype=np.int64),
        body_fallback_frames=np.array(body_fallback_frames, dtype=np.int64),
        body_scale_mean=np.array(body_scales.mean(), dtype=np.float32),
        body_scale_std=np.array(body_scales.std(), dtype=np.float32),
        body_scale_min=np.array(body_scales.min(), dtype=np.float32),
        body_scale_max=np.array(body_scales.max(), dtype=np.float32),
        train_video_count=np.array(len(train), dtype=np.int64),
        normalization_version=np.array("body_centric_train_only_v1"),
    )

    fallback_rate = body_fallback_frames / max(1, total_frames)

    print("\n" + "=" * 88)
    print("TRAIN-ONLY NORMALIZATION SUMMARY")
    print("=" * 88)
    print(f"TRAIN videos              : {len(train)}")
    print(f"TRAIN frames              : {total_frames}")
    print(f"Left valid frames         : {left_valid_frames}")
    print(f"Right valid frames        : {right_valid_frames}")
    print(f"Pose valid frames         : {pose_valid_frames}")
    print(f"Face valid frames         : {face_valid_frames}")
    print(f"Body-scale fallback frames: {body_fallback_frames}")
    print(f"Body-scale fallback rate  : {fallback_rate:.4f}")
    print(f"Mean body scale           : {body_scales.mean():.4f}")
    print(f"Std body scale            : {body_scales.std():.4f}")
    print()
    print(f"Saved normalization:\n{OUTPUT_NPZ}")

    summary = [
        "=" * 88,
        "INCLUDE 262 TRAIN-ONLY NORMALIZATION V1 SUMMARY",
        "=" * 88,
        "",
        "Statistics source: TRAIN SPLIT ONLY",
        f"Train videos: {len(train)}",
        f"Train frames: {total_frames}",
        "Validation videos used: 0",
        "Test videos used: 0",
        "",
        f"Left valid frames: {left_valid_frames}",
        f"Right valid frames: {right_valid_frames}",
        f"Pose valid frames: {pose_valid_frames}",
        f"Face valid frames: {face_valid_frames}",
        "",
        f"Body-scale fallback frames: {body_fallback_frames}",
        f"Body-scale fallback rate: {fallback_rate:.6f}",
        f"Mean body scale: {body_scales.mean():.6f}",
        f"Std body scale: {body_scales.std():.6f}",
        "",
        "Normalization:",
        "  x/y centered on shoulder midpoint",
        "  coordinates scaled by shoulder distance",
        "  pose z centered on shoulder-midpoint z",
        "  hand/face z scale-normalized",
        "  missing hand frames excluded from mean/std statistics",
        "",
        f"Output: {OUTPUT_NPZ}",
        "",
        "STATUS:",
        "TRAIN-ONLY NORMALIZATION READY ✅",
        "NEXT: build landmark + hand-centric geometry + motion feature sequences.",
    ]

    SUMMARY_TXT.write_text("\n".join(summary), encoding="utf-8")

    print("\n" + "\n".join(summary))
    print("=" * 88)


if __name__ == "__main__":
    main()