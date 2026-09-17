"""
Batch-preprocess collected webcam clips into model features.

    webcam_data/{train,val}/*.{mp4,mov,webm}  (+ <clip>.json sidecars)
        -> webcam_extract.extract_trimmed  (active-sign trim, frozen MediaPipe config)
        -> preprocessing.build_features_from_arrays  (identical to training: scripts/79)
        -> processed_data/WEBCAM/model_features64_v1/<split>/<clip>.npz
        + webcam_data/model_features_manifest.csv   (incl. authoritative coverage)

Uses the FROZEN INCLUDE TRAIN normalization files only — never fits new statistics.

    python -m webcam.preprocess_webcam                 # all splits, resumable
    python -m webcam.preprocess_webcam --split val
    python -m webcam.preprocess_webcam --no-trim       # ablation: skip trimming
    python -m webcam.preprocess_webcam --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from webcam import _analysis as A

config = A.config
webcam_extract = A.webcam_extract
preprocessing = A.preprocessing

VIDEO_EXT = {".mp4", ".mov", ".webm", ".MOV", ".MP4"}
MANIFEST_FIELDS = [
    "clip", "split", "class_id", "label", "feature_path", "status",
    "orig_duration_s", "trimmed_duration_s", "trimmed_frame_range", "motion_score", "trim_fallback",
    "left_final_rate", "right_final_rate", "both_hands_final_rate", "pose_rate", "face_rate",
    "session_id", "background_id", "lighting", "camera_distance", "position", "clothing", "trimmed",
]


def _iter_clips(split_dir: Path):
    for p in sorted(split_dir.iterdir()):
        if p.suffix in VIDEO_EXT and p.is_file():
            yield p


def _sidecar(clip: Path) -> dict:
    j = clip.with_suffix(clip.suffix + ".json")
    if j.exists():
        try:
            return json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _feature_out(split: str, clip: Path) -> Path:
    return config.WEBCAM_FEATURES / split / (clip.stem + ".npz")


def process_clip(clip: Path, split: str, disable_trim: bool) -> dict:
    meta = _sidecar(clip)
    class_id = meta.get("class_id")
    label = meta.get("label", "")
    if class_id is None:
        return {"clip": clip.name, "split": split, "status": "no_sidecar_class_id"}

    arrays, xstats = webcam_extract.extract_trimmed(str(clip), disable_trim=disable_trim)
    feats = preprocessing.build_features_from_arrays(arrays)

    out = _feature_out(split, clip)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        landmark_stream=feats["landmark"],
        geometry_stream=feats["geometry"],
        availability_mask=feats["availability"],
        class_id=np.array(int(class_id), dtype=np.int32),
        label=np.array(str(label)),
        split=np.array(str(split)),
        source=np.array("webcam"),
    )

    return {
        "clip": clip.name, "split": split, "class_id": int(class_id), "label": label,
        "feature_path": str(out), "status": "ready",
        "orig_duration_s": xstats.get("original_duration_s"),
        "trimmed_duration_s": xstats.get("trimmed_duration_s"),
        "trimmed_frame_range": xstats.get("trimmed_frame_range"),
        "motion_score": xstats.get("motion_score"),
        "trim_fallback": xstats.get("trim_fallback"),
        "trimmed": xstats.get("trimmed") and not disable_trim,
        "left_final_rate": round(float(xstats.get("left_final_rate", 0.0)), 3),
        "right_final_rate": round(float(xstats.get("right_final_rate", 0.0)), 3),
        "both_hands_final_rate": round(float(xstats.get("both_hands_final_rate", 0.0)), 3),
        "pose_rate": round(float(xstats.get("pose_rate", 0.0)), 3),
        "face_rate": round(float(xstats.get("face_rate", 0.0)), 3),
        "session_id": meta.get("session_id", ""),
        "background_id": meta.get("background_id", ""),
        "lighting": meta.get("lighting", ""),
        "camera_distance": meta.get("camera_distance", ""),
        "position": meta.get("position", ""),
        "clothing": meta.get("clothing", ""),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "all"], default="all")
    ap.add_argument("--no-trim", dest="no_trim", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    if not config.WEBCAM_DATA.exists():
        raise SystemExit(f"No webcam_data/ yet. Collect clips first (see webcam/README.md).\n  {config.WEBCAM_DATA}")

    splits = ["train", "val"] if args.split == "all" else [args.split]
    manifest_path = config.WEBCAM_DATA / "model_features_manifest.csv"
    existing = {}
    if manifest_path.exists() and not args.overwrite:
        with open(manifest_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                existing[(row["split"], row["clip"])] = row

    rows = dict(existing)
    started = time.time()
    n_done = 0
    for split in splits:
        sd = config.WEBCAM_DATA / split
        if not sd.exists():
            print(f"(no {split}/ dir, skipping)")
            continue
        clips = list(_iter_clips(sd))
        print(f"{split}: {len(clips)} clips")
        for clip in clips:
            key = (split, clip.name)
            out = _feature_out(split, clip)
            if not args.overwrite and key in existing and out.exists():
                continue
            try:
                rec = process_clip(clip, split, disable_trim=args.no_trim)
            except Exception as exc:  # noqa: BLE001
                rec = {"clip": clip.name, "split": split, "status": f"error:{exc}"}
            rows[key] = rec
            n_done += 1
            print(f"  [{n_done}] {clip.name} -> {rec.get('status')}  "
                  f"(L/R/P {rec.get('left_final_rate')}/{rec.get('right_final_rate')}/{rec.get('pose_rate')})")

    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for rec in rows.values():
            w.writerow({k: rec.get(k, "") for k in MANIFEST_FIELDS})

    ready = sum(1 for r in rows.values() if r.get("status") == "ready")
    print(f"\ndone in {(time.time()-started)/60:.1f} min  |  {ready} feature files ready")
    print(f"features : {config.WEBCAM_FEATURES}")
    print(f"manifest : {manifest_path}")


if __name__ == "__main__":
    main()
