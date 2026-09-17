"""
Raw video  ->  (landmark, geometry, availability) tensors for INCLUDE262.

This module does NOT re-implement the INCLUDE preprocessing. It drives the
existing project scripts:

  * scripts/76_extract_include262_mediapipe64_v1.py
        - sample_64_frames / process_video  (MediaPipe Holistic + pose-guided
          hand rescue -> raw 64-frame landmark arrays)

  * scripts/79_build_include262_model_features64_v1.py
        - load_landmark_norm
        - build_one_feature_file            (Phase A + Phase B:
          body-centric normalization, z-scored landmark stream,
          hand-centric geometry, bone lengths, joint angles, relations,
          velocity, acceleration)

Only Phase C (geometry standardization) is applied inline, byte-for-byte
identical to `add_standardized_geometry` in script 79 (lines ~599-608),
because that function is written as a manifest/file rewriter and cannot be
called on a single in-memory sample.

Nothing in datasets/ metadata/ processed_data/ models/ results/ scripts/ is
modified. All temp files live in a TemporaryDirectory that is deleted on exit.
"""

import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from model_loader import load_script_module  # noqa: E402

# ---------------------------------------------------------------------------
# MediaPipe configuration.
#
# Copied VERBATIM from scripts/76_extract_include262_mediapipe64_v1.py, main()
# (the `holistic_kwargs` / `hands_kwargs` dicts, lines ~400-417). These are
# the extractor's config constants and live inside main(), so they cannot be
# imported; every value below must stay identical to script 76.
# ---------------------------------------------------------------------------
HOLISTIC_KWARGS = dict(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45,
)

HANDS_KWARGS = dict(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.40,
    min_tracking_confidence=0.40,
)

# ---------------------------------------------------------------------------
# Import the existing scripts once.
# ---------------------------------------------------------------------------
_s76 = load_script_module(config.SCRIPT_76, "include262_extract_76")
_s79 = load_script_module(config.SCRIPT_79, "include262_build_79")

# Script 79 resolves its own paths from Path.cwd(); repoint the ones its
# helper functions read so this works regardless of the current directory.
_s79.ROOT = config.ROOT
_s79.META = config.METADATA
_s79.LANDMARK_NORM = str(config.LANDMARK_NORM)
_s79.GEOM_NORM = str(config.GEOMETRY_NORM)

_LANDMARK_NORMS = _s79.load_landmark_norm()  # dict: left/right/pose/face mean+std


def _standardize_geometry(geometry_raw: np.ndarray, geometry_valid: np.ndarray) -> np.ndarray:
    """
    Phase C - identical to scripts/79 add_standardized_geometry():
        standardized = (geometry_raw - mean) / std
        geometry_stream[valid] = standardized[valid]   (invalid entries stay 0)
    """
    with np.load(config.GEOMETRY_NORM, allow_pickle=False) as gn:
        mean = gn["mean"].astype(np.float32)
        std = gn["std"].astype(np.float32)

    standardized = (geometry_raw.astype(np.float32) - mean[None, :]) / std[None, :]
    geometry_stream = np.zeros_like(geometry_raw, dtype=np.float32)
    geometry_stream[geometry_valid.astype(bool)] = standardized[geometry_valid.astype(bool)]
    return geometry_stream


def build_features_from_arrays(arrays: dict) -> dict:
    """
    Phase A + B + C on already-extracted raw landmark arrays.

    `arrays` must be the exact dict that scripts/76 `process_video` returns
    (left_hand, right_hand, pose, face_selected, face_indices, *_detected,
    *_rescued, ...). This is the ONE code path used by every caller — the frozen
    upload flow (`preprocess_video`), the live webcam flow (`live_inference`) and
    the offline `webcam/preprocess_webcam.py` — so their features are guaranteed
    identical to training.

    Returns: {landmark (64,534), geometry (64,469), availability (64,4)} float32.
    """
    with tempfile.TemporaryDirectory(prefix="include262_infer_") as tmp:
        tmp = Path(tmp)

        # Write the raw arrays in the exact on-disk layout script 76 produces,
        # so build_one_feature_file() consumes them unchanged.
        raw_npz = tmp / "raw_landmarks.npz"
        np.savez_compressed(
            raw_npz,
            **arrays,
            label=np.array("infer"),
            class_id=np.array(0, dtype=np.int32),
            split=np.array("infer"),
            source_index=np.array(0, dtype=np.int32),
        )

        row = {
            "feature_path": str(raw_npz),
            "source_index": 0,
            "video_name": "infer",
            "label": "infer",
            "class_id": 0,
            "split": "infer",
        }

        feature_npz = tmp / "model_features.npz"
        _s79.build_one_feature_file(row, _LANDMARK_NORMS, feature_npz)

        with np.load(feature_npz, allow_pickle=False) as data:
            landmark_stream = data["landmark_stream"].astype(np.float32)
            geometry_raw = data["geometry_raw"].astype(np.float32)
            geometry_valid = data["geometry_valid"].astype(bool)
            availability = data["availability_mask"].astype(np.float32)

    # -- Phase C: geometry standardization (scripts/79 add_standardized_geometry)
    geometry_stream = _standardize_geometry(geometry_raw, geometry_valid)

    # -- Shape + finiteness contract --------------------------------------
    expected = {
        "landmark": (config.NUM_FRAMES, config.LANDMARK_DIM),
        "geometry": (config.NUM_FRAMES, config.GEOMETRY_DIM),
        "availability": (config.NUM_FRAMES, config.AVAILABILITY_DIM),
    }
    got = {
        "landmark": landmark_stream.shape,
        "geometry": geometry_stream.shape,
        "availability": availability.shape,
    }
    for k in expected:
        if tuple(got[k]) != expected[k]:
            raise RuntimeError(f"{k} shape {got[k]} != expected {expected[k]}")

    for name, arr in (("landmark", landmark_stream),
                      ("geometry", geometry_stream),
                      ("availability", availability)):
        if not np.isfinite(arr).all():
            raise RuntimeError(f"Non-finite values in {name} stream")

    return {
        "landmark": np.ascontiguousarray(landmark_stream),
        "geometry": np.ascontiguousarray(geometry_stream),
        "availability": np.ascontiguousarray(availability),
    }


def extract_raw_arrays(video_path: str):
    """scripts/76 raw extraction with the frozen MediaPipe config. Returns (arrays, stats)."""
    with _s76.mp_holistic.Holistic(**HOLISTIC_KWARGS) as holistic_model, \
         _s76.mp_hands.Hands(**HANDS_KWARGS) as hands_model:
        return _s76.process_video(video_path, holistic_model, hands_model)


def preprocess_video(video_path) -> dict:
    """
    Frozen upload path — behavior-identical to before the build_features refactor.

    Returns a dict with:
        landmark            np.float32 (64, 534)
        geometry            np.float32 (64, 469)
        availability        np.float32 (64, 4)
        modality_coverage   per-frame availability means + script-76 rates
        extraction_status   'ok' or a failure reason from script 76
        preprocess_seconds  wall-clock for the whole function
    """
    video_path = str(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    t0 = time.perf_counter()

    # -- Step 1: raw landmark extraction (scripts/76) --------------------------
    arrays, stats = extract_raw_arrays(video_path)

    if arrays is None:
        raise RuntimeError(
            f"MediaPipe extraction failed for {video_path!r}: {stats.get('status')}"
        )

    # -- Step 2 + 3: model features Phase A + B + C (scripts/79) --------------
    feats = build_features_from_arrays(arrays)
    landmark_stream = feats["landmark"]
    geometry_stream = feats["geometry"]
    availability = feats["availability"]

    elapsed = time.perf_counter() - t0

    return {
        "landmark": np.ascontiguousarray(landmark_stream),
        "geometry": np.ascontiguousarray(geometry_stream),
        "availability": np.ascontiguousarray(availability),
        "extraction_status": stats.get("status", "ok"),
        "modality_coverage": {
            "per_frame_availability": {
                "left_hand": float(availability[:, 0].mean()),
                "right_hand": float(availability[:, 1].mean()),
                "pose": float(availability[:, 2].mean()),
                "face": float(availability[:, 3].mean()),
            },
            "extractor_rates": {
                "left_final_rate": float(stats.get("left_final_rate", 0.0)),
                "right_final_rate": float(stats.get("right_final_rate", 0.0)),
                "both_hands_final_rate": float(stats.get("both_hands_final_rate", 0.0)),
                "left_rescue_rate": float(stats.get("left_rescue_rate", 0.0)),
                "right_rescue_rate": float(stats.get("right_rescue_rate", 0.0)),
                "pose_rate": float(stats.get("pose_rate", 0.0)),
                "face_rate": float(stats.get("face_rate", 0.0)),
            },
        },
        "preprocess_seconds": elapsed,
    }
