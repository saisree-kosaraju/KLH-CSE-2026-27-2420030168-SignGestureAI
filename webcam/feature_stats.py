"""
Phase 9 — feature distribution comparison: INCLUDE TRAIN vs WEBCAM TRAIN.

Aggregates the model-feature npz files for both sources over shared labels and
compares landmark / geometry / motion / availability distributions, so failures
can be attributed to (A) MediaPipe extraction, (B) temporal segmentation,
(C) handedness/orientation, (D) geometry distribution shift, or (E) classifier shift.

Never reads the INCLUDE locked TEST split.

    python -m webcam.feature_stats
    python -m webcam.feature_stats --include-split train --webcam-split train

Requires `python -m webcam.preprocess_webcam` first.
Writes results/webcam_feature_stats_<ts>.txt
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from webcam import _analysis as A

config = A.config
INCLUDE_FEATURES = config.ROOT / "processed_data" / "INCLUDE262" / "model_features64_v1"


def _agg(paths, key_land, key_geom):
    """Mean abs value per landmark-block and geometry-slice + availability, over a set of npz files."""
    land = defaultdict(list)
    geom = defaultdict(list)
    avail = []
    vel, acc = [], []
    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            L = d[key_land].astype(np.float32)
            G = d[key_geom].astype(np.float32) if key_geom in d else d["geometry_raw"].astype(np.float32)
            Av = d["availability_mask"].astype(np.float32)
        for name, sl in A.LANDMARK_BLOCKS.items():
            b = L[:, sl]
            nz = np.any(b != 0, axis=1)
            land[name].append(float(np.abs(b[nz]).mean()) if nz.any() else 0.0)
        for name, sl in A.GEOMETRY_SLICES.items():
            b = G[:, sl]
            nz = np.any(b != 0, axis=1)
            geom[name].append(float(np.abs(b[nz]).mean()) if nz.any() else 0.0)
        avail.append(Av.mean(axis=0))
        v = np.abs(np.concatenate([G[:, A.GEOMETRY_SLICES["left_vel"]], G[:, A.GEOMETRY_SLICES["right_vel"]]], axis=1))
        a = np.abs(np.concatenate([G[:, A.GEOMETRY_SLICES["left_acc"]], G[:, A.GEOMETRY_SLICES["right_acc"]]], axis=1))
        vel.append(float(v[v > 0].mean()) if (v > 0).any() else 0.0)
        acc.append(float(a[a > 0].mean()) if (a > 0).any() else 0.0)
    return {
        "n": len(paths),
        "land": {k: float(np.mean(v)) for k, v in land.items()},
        "geom": {k: float(np.mean(v)) for k, v in geom.items()},
        "avail": np.mean(avail, axis=0) if avail else np.zeros(4),
        "vel": float(np.mean(vel)) if vel else 0.0,
        "acc": float(np.mean(acc)) if acc else 0.0,
    }


def _include_paths(split: str, labels: set[int]) -> list[Path]:
    import pandas as pd
    man = config.METADATA / "include262_model_features64_manifest_v1.csv"
    df = pd.read_csv(man)
    df = df[(df["split"] == split) & (df["class_id"].isin(labels))]
    return [Path(p) for p in df["feature_path"] if Path(p).exists()]


def _webcam_paths(split: str) -> tuple[list[Path], set[int]]:
    man = config.WEBCAM_DATA / "model_features_manifest.csv"
    if not man.exists():
        raise SystemExit("Run `python -m webcam.preprocess_webcam` first.")
    paths, labels = [], set()
    with open(man, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == split and r["status"] == "ready" and Path(r["feature_path"]).exists():
                paths.append(Path(r["feature_path"]))
                labels.add(int(r["class_id"]))
    return paths, labels


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-split", default="train", choices=["train", "val"])
    ap.add_argument("--webcam-split", default="train", choices=["train", "val"])
    args = ap.parse_args(argv)

    cam_paths, labels = _webcam_paths(args.webcam_split)
    if not cam_paths:
        print("No webcam features yet. Run: python -m webcam.preprocess_webcam")
        return
    inc_paths = _include_paths(args.include_split, labels)

    cam = _agg(cam_paths, "landmark_stream", "geometry_stream")
    inc = _agg(inc_paths, "landmark_stream", "geometry_stream")

    ts = time.strftime("%Y%m%d_%H%M%S")
    lines = ["=" * 82,
             f"FEATURE DISTRIBUTION: INCLUDE {args.include_split} vs WEBCAM {args.webcam_split}   {ts}",
             "=" * 82,
             f"shared classes: {len(labels)}   include clips: {inc['n']}   webcam clips: {cam['n']}",
             "",
             f"{'stream / slice':<22}{'INCLUDE':>10}{'WEBCAM':>10}{'ratio':>9}   flag",
             "-" * 82]
    def row(name, a, b):
        ratio = (b / a) if a > 1e-9 else float("nan")
        flag = "  <-- SHIFT" if (ratio == ratio and (ratio > 1.6 or ratio < 0.6)) else ""
        lines.append(f"{name:<22}{a:>10.4f}{b:>10.4f}{ratio:>9.2f}{flag}")

    lines.append("[availability]")
    for i, k in enumerate(("left", "right", "pose", "face")):
        row(f"  {k}", float(inc['avail'][i]), float(cam['avail'][i]))
    lines.append("[landmark blocks |mean|]")
    for k in A.LANDMARK_BLOCKS:
        row(f"  {k}", inc["land"][k], cam["land"][k])
    lines.append("[geometry slices |mean|]")
    for k in A.GEOMETRY_SLICES:
        row(f"  {k}", inc["geom"][k], cam["geom"][k])
    lines.append("[motion]")
    row("  abs_velocity", inc["vel"], cam["vel"])
    row("  abs_accel", inc["acc"], cam["acc"])

    lines += ["", "Interpretation guide:",
              "  availability shift        -> (A) MediaPipe extraction under webcam conditions",
              "  left/right coord swap      -> (C) handedness/orientation",
              "  bones/angles shift         -> (D) geometry distribution (signer scale / distance)",
              "  velocity/accel shift       -> (B) segmentation / signing-speed mismatch",
              "  all small, accuracy low    -> (E) classifier domain shift -> consider adaptation",
              "=" * 82]

    text = "\n".join(lines)
    print(text)
    out = Path(config.RESULTS) / f"webcam_feature_stats_{ts}.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
