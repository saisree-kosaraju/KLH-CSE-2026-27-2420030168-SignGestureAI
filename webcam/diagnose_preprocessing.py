"""
Phase 2 — webcam vs INCLUDE preprocessing equivalence diagnostic.

Runs a known-good INCLUDE clip and a webcam clip of the same sign through BOTH the
frozen path and the trimmed webcam path, and prints a stage-by-stage comparison so
you can see exactly where webcam input diverges from what the model was trained on.

    python -m webcam.diagnose_preprocessing --include water --webcam webcam_data/val/cXXX.mp4
    python -m webcam.diagnose_preprocessing --include 41 --webcam clip.mp4 --label water

--include accepts a file path, a label, or an integer source_index into
metadata/include262_official_split_final.csv.

Writes results/webcam_preprocessing_diagnostic_<timestamp>.txt and .json.
Never reads the INCLUDE locked TEST split for tuning; this is a diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from webcam import _analysis as A

FLAG = 0.35  # relative-difference threshold for the delta report


def _fmt_topk(items):
    return "  ".join(f"{d['sign']}={d['prob']}" for d in items)


def _delta_lines(inc: dict, cam: dict) -> list[str]:
    lines = []
    # availability
    for k in ("left", "right", "pose", "face"):
        a, b = inc["availability_mean"][k], cam["availability_mean"][k]
        mark = "  <-- FLAG" if abs(a - b) > 0.25 else ""
        lines.append(f"  availability.{k:<6} include={a:<6} webcam={b:<6} d={b - a:+.3f}{mark}")
    # landmark blocks (abs_mean, zero_row_frac)
    for name in A.LANDMARK_BLOCKS:
        ia, ca = inc["landmark_blocks"][name], cam["landmark_blocks"][name]
        d_abs = ca["abs_mean"] - ia["abs_mean"]
        rel = abs(d_abs) / (abs(ia["abs_mean"]) + 1e-6)
        mark = "  <-- FLAG" if rel > FLAG else ""
        lines.append(
            f"  L[{name:<10}] abs_mean inc={ia['abs_mean']:<7} cam={ca['abs_mean']:<7} "
            f"d={d_abs:+.3f}  zero_rows inc={ia['zero_row_frac']} cam={ca['zero_row_frac']}{mark}"
        )
    # geometry slices (abs_mean)
    for name in A.GEOMETRY_SLICES:
        ig, cg = inc["geometry_slices"][name], cam["geometry_slices"][name]
        d_abs = cg["abs_mean"] - ig["abs_mean"]
        rel = abs(d_abs) / (abs(ig["abs_mean"]) + 1e-6)
        mark = "  <-- FLAG" if rel > FLAG else ""
        lines.append(
            f"  G[{name:<11}] abs_mean inc={ig['abs_mean']:<7} cam={cg['abs_mean']:<7} d={d_abs:+.3f}{mark}"
        )
    # motion
    for k in ("abs_velocity_mean", "abs_accel_mean"):
        a, b = inc["motion"][k], cam["motion"][k]
        rel = abs(b - a) / (abs(a) + 1e-6)
        mark = "  <-- FLAG" if rel > FLAG else ""
        lines.append(f"  motion.{k:<18} include={a:<8} webcam={b:<8} d={b - a:+.4f}{mark}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", required=True, help="INCLUDE clip: path | label | source_index")
    ap.add_argument("--webcam", required=True, help="webcam clip path (mp4/mov/webm decodable by OpenCV)")
    ap.add_argument("--label", default="", help="ground-truth sign (for the report header)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    inc_path, inc_label = A.resolve_include(args.include)
    label = args.label or inc_label
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out or A.config.RESULTS / f"webcam_preprocessing_diagnostic_{ts}.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"INCLUDE : {inc_path}")
    print(f"webcam  : {args.webcam}")
    print(f"label   : {label or '(unspecified)'}\n")

    report: dict = {"label": label, "include_path": inc_path, "webcam_path": args.webcam,
                    "normalization": A.norm_file_info(), "runs": {}}

    runs = {}
    for tag, path, trimmed in (
        ("include_frozen", inc_path, False),
        ("include_trimmed", inc_path, True),
        ("webcam_frozen", args.webcam, False),
        ("webcam_trimmed", args.webcam, True),
    ):
        print(f"  analysing {tag} ...")
        snap = A.analyse(path, trimmed=trimmed)
        snap["prediction"] = A.predict(snap.pop("_feats"))
        runs[tag] = snap
    report["runs"] = runs

    lines = []
    P = lines.append
    P("=" * 96)
    P(f"WEBCAM vs INCLUDE PREPROCESSING DIAGNOSTIC   label={label or '(unspecified)'}   {ts}")
    P("=" * 96)
    P(f"MediaPipe: {report['normalization']['mediapipe_version']}   (frozen path + trimmed path use identical config)")
    P(f"landmark norm sha={report['normalization']['landmark_norm']['sha256_16']}   "
      f"geometry norm sha={report['normalization']['geometry_norm']['sha256_16']}")
    P("")
    for tag, snap in runs.items():
        v = snap["video"]
        t = snap["trim"]
        P(f"[{tag}]")
        P(f"   video     : {v['frame_count']} frames  {v['fps']} fps  {v['width']}x{v['height']}  "
          f"aspect {v['aspect']}  dur {v['duration_s']}s")
        if t:
            P(f"   trim      : frames {t['trimmed_frame_range']} of {t['total_frames']}  "
              f"motion={t['motion_score']}  fallback={t['trim_fallback']}")
        P(f"   avail L/R/P/F : {snap['availability_mean']['left']} / {snap['availability_mean']['right']} / "
          f"{snap['availability_mean']['pose']} / {snap['availability_mean']['face']}")
        P(f"   motion |v|/|a| : {snap['motion']['abs_velocity_mean']} / {snap['motion']['abs_accel_mean']}")
        P(f"   tensor shapes : {snap['tensor']['landmark_shape']} {snap['tensor']['geometry_shape']} "
          f"{snap['tensor']['availability_shape']} {snap['tensor']['dtype']}")
        P(f"   prediction    : {_fmt_topk(snap['prediction'])}")
        P("")

    P("-" * 96)
    P("DELTA REPORT  (webcam_trimmed  vs  include_frozen)   FLAG = relative diff > "
      f"{FLAG:.0%} or availability gap > 0.25")
    P("-" * 96)
    for ln in _delta_lines(runs["include_frozen"], runs["webcam_trimmed"]):
        P(ln)
    P("")
    P("-" * 96)
    P("PREDICTION SUMMARY")
    P("-" * 96)
    for tag, snap in runs.items():
        top = snap["prediction"][0]
        hit = "OK " if (label and top["sign"] == label) else ("   " if not label else "MISS")
        P(f"   {hit} {tag:<16} -> {top['sign']} ({top['prob']})")
    P("=" * 96)

    text = "\n".join(lines)
    print("\n" + text)
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten: {out}\n         {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
