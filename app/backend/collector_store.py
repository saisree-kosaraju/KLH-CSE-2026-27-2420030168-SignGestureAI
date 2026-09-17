"""
Dev-only webcam data collector storage (Module 10).

Saves human-labelled webcam clips + metadata under webcam_data/{train,val}/.
The HUMAN-selected label is ground truth — the classifier is never consulted here.

Saves are fast (no MediaPipe): the sidecar stores the browser-observed coverage
from the live overlay. Authoritative coverage is (re)computed later by
`webcam/preprocess_webcam.py` with the frozen Holistic config.

Only imported/mounted when config.COLLECTOR_ENABLED is true.
"""

from __future__ import annotations

import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import config

_ROOT = config.WEBCAM_DATA
_MANIFEST = _ROOT / "manifest.csv"
_SPLITS = ("train", "val")
_ALLOWED_EXT = {".mp4", ".mov", ".webm"}

_MANIFEST_FIELDS = [
    "filename", "split", "class_id", "label", "timestamp", "session_id",
    "background_id", "lighting", "position", "clothing", "camera_distance",
    "duration_s", "width", "height", "fps",
    "left_coverage", "right_coverage", "pose_coverage", "face_coverage",
    "coverage_source", "source", "notes", "app_version",
]


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "x"


def _load_vocab_label(class_id: int) -> str | None:
    import sys
    sys.path.insert(0, str(config.ROOT))
    from webcam.vocab import all_classes  # noqa: E402
    for r in all_classes():
        if r["class_id"] == class_id:
            return r["label"]
    return None


def save_sample(file_bytes: bytes, ext: str, meta: dict) -> dict:
    ext = ext.lower()
    if ext not in _ALLOWED_EXT:
        raise ValueError(f"unsupported extension {ext!r}")
    if not file_bytes:
        raise ValueError("empty upload")

    split = str(meta.get("split", "")).lower()
    if split not in _SPLITS:
        raise ValueError("split must be 'train' or 'val'")

    try:
        class_id = int(meta["class_id"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("class_id is required and must be an integer")

    true_label = _load_vocab_label(class_id)
    if true_label is None:
        raise ValueError(f"class_id {class_id} is not one of the 262 INCLUDE signs")

    given_label = str(meta.get("label", "")).strip()
    if given_label and given_label != true_label:
        raise ValueError(
            f"label {given_label!r} does not match class_id {class_id} ({true_label!r})"
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    session = _safe(meta.get("session_id") or "s0")
    filename = f"c{class_id:03d}_{_safe(true_label)}_{session}_{ts}_{int(time.time() * 1000) % 100000}{ext}"

    split_dir = _ROOT / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / filename).write_bytes(file_bytes)

    def _num(key, default=None):
        try:
            return float(meta[key])
        except (KeyError, TypeError, ValueError):
            return default

    record = {
        "filename": filename,
        "split": split,
        "class_id": class_id,
        "label": true_label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session,
        "background_id": _safe(meta.get("background_id") or "unset"),
        "lighting": _safe(meta.get("lighting") or "unset"),
        "position": _safe(meta.get("position") or "center"),
        "clothing": _safe(meta.get("clothing") or "unset"),
        "camera_distance": _safe(meta.get("camera_distance") or "normal"),
        "duration_s": _num("duration_s"),
        "width": _num("width"),
        "height": _num("height"),
        "fps": _num("fps"),
        "left_coverage": _num("left_coverage"),
        "right_coverage": _num("right_coverage"),
        "pose_coverage": _num("pose_coverage"),
        "face_coverage": _num("face_coverage"),
        "coverage_source": "browser",
        "source": "webcam",
        "notes": str(meta.get("notes") or "").replace("\n", " ")[:300],
        "app_version": str(meta.get("app_version") or "signspeak-ai"),
    }

    (split_dir / f"{filename}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    _append_manifest(record)

    return {
        "saved": True,
        "filename": filename,
        "split": split,
        "class_id": class_id,
        "label": true_label,
        "coverage": {
            "left": record["left_coverage"] or 0.0,
            "right": record["right_coverage"] or 0.0,
            "pose": record["pose_coverage"] or 0.0,
            "face": record["face_coverage"] or 0.0,
        },
        "counts": counts()["by_class"].get(str(class_id), {"train": 0, "val": 0}),
        "message": f"Saved {true_label} to {split}.",
    }


def _append_manifest(record: dict) -> None:
    _ROOT.mkdir(parents=True, exist_ok=True)
    new = not _MANIFEST.exists()
    with open(_MANIFEST, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: record.get(k, "") for k in _MANIFEST_FIELDS})


def counts() -> dict:
    by_split = {s: 0 for s in _SPLITS}
    by_class: dict[str, dict] = {}
    total = 0
    if _MANIFEST.exists():
        with open(_MANIFEST, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                s = row.get("split", "")
                cid = row.get("class_id", "")
                if s not in _SPLITS:
                    continue
                # only count rows whose clip still exists on disk
                if not (_ROOT / s / row.get("filename", "")).exists():
                    continue
                total += 1
                by_split[s] += 1
                slot = by_class.setdefault(cid, {"label": row.get("label", ""), "train": 0, "val": 0})
                slot[s] += 1
    return {"total": total, "by_split": by_split, "by_class": by_class}
