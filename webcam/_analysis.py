"""
Shared feature-analysis helpers for the webcam diagnostics.

Loads the backend modules (app/backend) so every path uses the exact same
extraction + feature build as serving. Read-only: never writes model/metadata.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))
sys.path.insert(0, str(ROOT))

import config           # noqa: E402
import preprocessing    # noqa: E402
import webcam_extract   # noqa: E402
import inference        # noqa: E402
import torch            # noqa: E402

# landmark_stream (534) block layout — from scripts/79 write_schema()
LANDMARK_BLOCKS = {
    "left_hand":  slice(0, 63),
    "right_hand": slice(63, 126),
    "pose_xyz":   slice(126, 225),
    "pose_vis":   slice(225, 258),
    "face":       slice(258, 534),
}

# geometry_stream (469) slice layout — from scripts/79 G_* constants
GEOMETRY_SLICES = {
    "left_coord":  slice(0, 63),
    "right_coord": slice(63, 126),
    "left_bones":  slice(126, 146),
    "right_bones": slice(146, 166),
    "left_angles": slice(166, 181),
    "right_angles": slice(181, 196),
    "relations":   slice(196, 217),
    "left_vel":    slice(217, 280),
    "right_vel":   slice(280, 343),
    "left_acc":    slice(343, 406),
    "right_acc":   slice(406, 469),
}


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def video_meta(path: str) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "path": os.path.abspath(path),
        "frame_count": n,
        "fps": round(fps, 3),
        "width": w,
        "height": h,
        "aspect": round(w / h, 4) if h else None,
        "duration_s": round(n / fps, 3) if fps > 1e-3 else None,
    }


def block_stats(arr: np.ndarray, blocks: dict) -> dict:
    """Per-block mean / std / % all-zero-rows / min / max for a (64, D) stream."""
    out = {}
    for name, sl in blocks.items():
        b = arr[:, sl]
        nonzero_rows = np.any(b != 0.0, axis=1)
        out[name] = {
            "mean": round(float(b[nonzero_rows].mean()) if nonzero_rows.any() else 0.0, 4),
            "std": round(float(b[nonzero_rows].std()) if nonzero_rows.any() else 0.0, 4),
            "abs_mean": round(float(np.abs(b[nonzero_rows]).mean()) if nonzero_rows.any() else 0.0, 4),
            "min": round(float(b.min()), 3),
            "max": round(float(b.max()), 3),
            "zero_row_frac": round(float(1.0 - nonzero_rows.mean()), 3),
        }
    return out


def analyse(video_path: str, trimmed: bool) -> dict:
    """Full stage-by-stage snapshot of one video through one path."""
    if trimmed:
        arrays, xstats = webcam_extract.extract_trimmed(video_path)
        status = xstats.get("status", "ok")
    else:
        arrays, xstats = preprocessing.extract_raw_arrays(video_path)
        if arrays is None:
            raise RuntimeError(f"frozen extraction failed: {xstats.get('status')}")
        status = xstats.get("status", "ok")

    feats = preprocessing.build_features_from_arrays(arrays)
    L, G, A = feats["landmark"], feats["geometry"], feats["availability"]

    # motion magnitudes from the geometry velocity/accel slices
    vel = np.abs(np.concatenate([G[:, GEOMETRY_SLICES["left_vel"]], G[:, GEOMETRY_SLICES["right_vel"]]], axis=1))
    acc = np.abs(np.concatenate([G[:, GEOMETRY_SLICES["left_acc"]], G[:, GEOMETRY_SLICES["right_acc"]]], axis=1))

    return {
        "path": os.path.abspath(video_path),
        "trimmed": trimmed,
        "video": video_meta(video_path),
        "extract_status": status,
        "extractor_rates": {
            k: round(float(xstats.get(k, 0.0)), 3)
            for k in ("left_detect_rate_raw", "right_detect_rate_raw", "left_rescue_rate",
                      "right_rescue_rate", "left_final_rate", "right_final_rate",
                      "both_hands_final_rate", "pose_rate", "face_rate")
        },
        "trim": {
            k: xstats.get(k) for k in
            ("trimmed_frame_range", "total_frames", "motion_score", "trim_fallback",
             "trimmed_duration_s", "original_duration_s", "scan_frames")
        } if trimmed else None,
        "availability_mean": {
            "left": round(float(A[:, 0].mean()), 3),
            "right": round(float(A[:, 1].mean()), 3),
            "pose": round(float(A[:, 2].mean()), 3),
            "face": round(float(A[:, 3].mean()), 3),
        },
        "landmark_blocks": block_stats(L, LANDMARK_BLOCKS),
        "geometry_slices": block_stats(G, GEOMETRY_SLICES),
        "motion": {
            "abs_velocity_mean": round(float(vel[vel > 0].mean()) if (vel > 0).any() else 0.0, 4),
            "abs_accel_mean": round(float(acc[acc > 0].mean()) if (acc > 0).any() else 0.0, 4),
        },
        "tensor": {
            "landmark_shape": list(L.shape), "geometry_shape": list(G.shape),
            "availability_shape": list(A.shape), "dtype": str(L.dtype),
        },
        "_feats": feats,  # kept for model scoring, not serialized
    }


def predict(feats: dict, topk: int = 5) -> list[dict]:
    model, _ = inference._load_model()
    cmap = inference._load_class_map()
    lm = torch.from_numpy(feats["landmark"]).unsqueeze(0)
    gm = torch.from_numpy(feats["geometry"]).unsqueeze(0)
    av = torch.from_numpy(feats["availability"]).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(lm, gm, av), dim=1)[0]
    p, idx = probs.topk(topk)
    return [{"class_id": int(i), "sign": cmap[int(i)], "prob": round(float(pp), 4)}
            for pp, i in zip(p, idx)]


def norm_file_info() -> dict:
    return {
        "landmark_norm": {
            "path": str(config.LANDMARK_NORM), "sha256_16": file_hash(config.LANDMARK_NORM),
        },
        "geometry_norm": {
            "path": str(config.GEOMETRY_NORM), "sha256_16": file_hash(config.GEOMETRY_NORM),
        },
        "mediapipe_version": _mp_version(),
        "holistic_kwargs": preprocessing.HOLISTIC_KWARGS,
        "hands_kwargs": preprocessing.HANDS_KWARGS,
    }


def _mp_version() -> str:
    try:
        import mediapipe
        return mediapipe.__version__
    except Exception:
        return "unknown"


def resolve_include(spec: str) -> tuple[str, str]:
    """spec is a file path or an integer source_index into the official split. Returns (path, label)."""
    import pandas as pd
    if os.path.isfile(spec):
        return spec, ""
    df = pd.read_csv(config.METADATA / "include262_official_split_final.csv")
    if spec.isdigit():
        row = df.iloc[int(spec)]
    else:
        m = df[df["label"] == spec]
        if m.empty:
            raise SystemExit(f"No INCLUDE video for label {spec!r}; pass a file path or source_index.")
        row = m.iloc[0]
    return row["video_path"], row["label"]
