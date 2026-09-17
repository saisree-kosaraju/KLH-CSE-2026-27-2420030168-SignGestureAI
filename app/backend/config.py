"""
Central configuration for the standalone INCLUDE262 inference sanity check.

Nothing here is executable pipeline logic - only paths and the frozen
architectural constants that the existing INCLUDE262 scripts already use.

Existing project files are treated as READ-ONLY. This module never writes to
datasets/, metadata/, processed_data/, models/, results/ or scripts/.
"""

import os
from pathlib import Path

# app/backend/config.py -> app/backend -> app -> <project root>
ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = ROOT / "scripts"
METADATA = ROOT / "metadata"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# ---------------------------------------------------------------------------
# Webcam serving-layer paths (new, additive — nothing in scripts/ datasets/
# metadata/ processed_data/INCLUDE262/ models/ is written).
# ---------------------------------------------------------------------------
WEBCAM_DATA = ROOT / "webcam_data"                                   # collected clips + sidecars
WEBCAM_FEATURES = ROOT / "processed_data" / "WEBCAM" / "model_features64_v1"

# Dev-only webcam data collector endpoints are mounted only when this is set.
COLLECTOR_ENABLED = os.environ.get("INCLUDE262_COLLECTOR", "").strip() in {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# Existing scripts that are reused verbatim through importlib
# ---------------------------------------------------------------------------
SCRIPT_76 = SCRIPTS / "76_extract_include262_mediapipe64_v1.py"   # raw video -> landmarks
SCRIPT_79 = SCRIPTS / "79_build_include262_model_features64_v1.py" # landmarks -> model features
SCRIPT_82 = SCRIPTS / "82_train_include262_partaware_htma_v2.py"   # model architecture (V2)

# ---------------------------------------------------------------------------
# Frozen checkpoint (load ema_state only) - produced by script 82, verified by
# scripts 83 (validation) and 84 (final locked test: 98.04% top-1). NEVER overwritten.
# ---------------------------------------------------------------------------
CHECKPOINT = MODELS / "checkpoints" / "include262_conv1d_htma_v2_seed42_best.pt"

# Optional webcam-adapted checkpoint (produced by webcam/adapt_train.py).
WEBCAM_ADAPTED_CHECKPOINT = MODELS / "checkpoints" / "include262_webcam_adapted_v1.pt"

# Which checkpoint the LIVE camera path (/predict/live) uses. The Upload path
# (/predict) always uses the original. Default: original -> zero behavior change.
#   INCLUDE262_MODEL_VARIANT=original | webcam_adapted
MODEL_VARIANT = os.environ.get("INCLUDE262_MODEL_VARIANT", "original").strip().lower() or "original"


def checkpoint_for(variant: str) -> Path:
    """Resolve a variant name to a checkpoint path, falling back to the original."""
    if variant == "webcam_adapted" and WEBCAM_ADAPTED_CHECKPOINT.exists():
        return WEBCAM_ADAPTED_CHECKPOINT
    return CHECKPOINT


def active_live_variant() -> str:
    """The variant actually in effect for /predict/live (accounts for a missing file)."""
    if MODEL_VARIANT == "webcam_adapted" and WEBCAM_ADAPTED_CHECKPOINT.exists():
        return "webcam_adapted"
    return "original"

# ---------------------------------------------------------------------------
# The exact two TRAIN-only normalization files used to build model features
# ---------------------------------------------------------------------------
LANDMARK_NORM = METADATA / "include262_landmarks64_train_normalization_v1.npz"
GEOMETRY_NORM = METADATA / "include262_geometry64_train_normalization_v1.npz"

# ---------------------------------------------------------------------------
# class_id -> sign name (262 rows, class_id 0..261)
# ---------------------------------------------------------------------------
CLASS_MAP_CSV = METADATA / "include262_class_mapping_final.csv"

# ---------------------------------------------------------------------------
# Frozen tensor contract (see metadata/include262_model_features64_schema_v1.json)
# ---------------------------------------------------------------------------
NUM_FRAMES = 64
LANDMARK_DIM = 534
GEOMETRY_DIM = 469
AVAILABILITY_DIM = 4
NUM_CLASSES = 262

DEVICE = "cpu"  # checkpoint is CPU torch; MediaPipe extraction dominates runtime


def check_paths() -> None:
    """Raise early with a clear message if any required artifact is missing."""
    required = {
        "script 76": SCRIPT_76,
        "script 79": SCRIPT_79,
        "script 82": SCRIPT_82,
        "checkpoint": CHECKPOINT,
        "landmark normalization": LANDMARK_NORM,
        "geometry normalization": GEOMETRY_NORM,
        "class mapping": CLASS_MAP_CSV,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required INCLUDE262 artifacts:\n  " + "\n  ".join(missing)
        )
