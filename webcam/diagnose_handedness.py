"""
Phase 3 — mirroring / handedness diagnosis.

The live PREVIEW is mirrored for natural UX (CSS scaleX(-1)), but the recorded clip
that reaches the backend is the RAW, un-mirrored getUserMedia stream (MediaRecorder
captures the track, not the CSS transform). So webcam inference should already use the
same orientation as training. This script verifies that empirically:

  A. raw clip                 (what the backend actually receives)
  B. horizontally flipped clip (cv2.flip) — the "wrong" orientation

For each: left/right hand final-detection rate, which hand carries the motion, mean
wrist-x, and the model's top-5. If B predicts *better* and the L/R rates swap, there is
an orientation mismatch to fix in webcam_extract.py (never with per-sign hacks).

    python -m webcam.diagnose_handedness --webcam clip.mp4 [--include water] [--label water]

Writes results/webcam_handedness_diagnostic_<timestamp>.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from webcam import _analysis as A


def _flip_to_temp(video_path: str) -> str:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = str(Path(A.config.RESULTS) / f"_hflip_tmp_{int(time.time())}.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(cv2.flip(fr, 1))
    cap.release()
    vw.release()
    return tmp


def _hand_motion_split(arrays: dict) -> dict:
    """Total frame-to-frame wrist travel per hand + mean wrist-x, over present frames."""
    out = {}
    for side, hk, dk, rk in (("left", "left_hand", "left_detected", "left_rescued"),
                             ("right", "right_hand", "right_detected", "right_rescued")):
        hand = arrays[hk]
        present = np.maximum(arrays[dk], arrays[rk]).astype(bool)
        wrist = hand[:, 0, :2]
        travel = 0.0
        for t in range(1, len(wrist)):
            if present[t] and present[t - 1]:
                travel += float(np.linalg.norm(wrist[t] - wrist[t - 1]))
        xs = wrist[present, 0]
        out[side] = {
            "final_rate": round(float(present.mean()), 3),
            "wrist_travel": round(travel, 3),
            "mean_wrist_x": round(float(xs.mean()), 3) if xs.size else None,
        }
    return out


def _run(path: str, tag: str) -> dict:
    arrays, xstats = A.preprocessing.extract_raw_arrays(path)
    if arrays is None:
        raise SystemExit(f"{tag}: extraction failed ({xstats.get('status')})")
    feats = A.preprocessing.build_features_from_arrays(arrays)
    return {"tag": tag, "split": _hand_motion_split(arrays), "pred": A.predict(feats)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam", required=True)
    ap.add_argument("--include", default=None, help="optional INCLUDE reference: path | label | source_index")
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(A.config.RESULTS) / f"webcam_handedness_diagnostic_{ts}.txt"
    lines: list[str] = []
    P = lines.append

    P("=" * 92)
    P(f"MIRRORING / HANDEDNESS DIAGNOSTIC   label={args.label or '(unspecified)'}   {ts}")
    P("=" * 92)
    P("CONVENTIONS")
    P("  training            : MediaPipe Holistic on UN-flipped INCLUDE videos.")
    P("                        left_hand_landmarks = signer's anatomical LEFT hand")
    P("                        (associated with pose LEFT_WRIST). landmark_stream 0:63 = left, 63:126 = right.")
    P("  webcam raw          : getUserMedia returns UN-mirrored sensor frames; MediaRecorder")
    P("                        captures the track (NOT the CSS scaleX(-1) preview). Uploaded clip = un-mirrored.")
    P("  MediaPipe handedness: Holistic assigns L/R by pose-wrist association (orientation-robust for")
    P("                        frontal views). Browser Tasks HandLandmarker (overlay only) uses image-space")
    P("                        handedness and is a SEPARATE concern from inference.")
    P("  final inference     : feed the RAW (un-flipped) clip. NO flip in webcam_extract.py.")
    P("")

    runs = [_run(args.webcam, "webcam_raw")]
    flipped = _flip_to_temp(args.webcam)
    try:
        runs.append(_run(flipped, "webcam_hflip"))
    finally:
        try:
            Path(flipped).unlink()
        except OSError:
            pass
    if args.include:
        inc_path, inc_label = A.resolve_include(args.include)
        if not args.label:
            args.label = inc_label
        runs.append(_run(inc_path, "include_ref"))

    for r in runs:
        s = r["split"]
        P(f"[{r['tag']}]")
        P(f"   left  : final_rate={s['left']['final_rate']:<6} travel={s['left']['wrist_travel']:<7} "
          f"mean_x={s['left']['mean_wrist_x']}")
        P(f"   right : final_rate={s['right']['final_rate']:<6} travel={s['right']['wrist_travel']:<7} "
          f"mean_x={s['right']['mean_wrist_x']}")
        P(f"   top5  : " + "  ".join(f"{d['sign']}={d['prob']}" for d in r["pred"]))
        P("")

    raw = next(r for r in runs if r["tag"] == "webcam_raw")
    flp = next(r for r in runs if r["tag"] == "webcam_hflip")
    raw_top, flp_top = raw["pred"][0], flp["pred"][0]

    P("-" * 92)
    P("VERDICT")
    P("-" * 92)
    rates_swapped = (
        abs(raw["split"]["left"]["final_rate"] - flp["split"]["right"]["final_rate"]) < 0.15
        and abs(raw["split"]["right"]["final_rate"] - flp["split"]["left"]["final_rate"]) < 0.15
        and abs(raw["split"]["left"]["final_rate"] - raw["split"]["right"]["final_rate"]) > 0.2
    )
    if args.label:
        raw_ok = raw_top["sign"] == args.label
        flp_ok = flp_top["sign"] == args.label
        if flp_ok and not raw_ok and rates_swapped:
            P("  !! ORIENTATION MISMATCH: the horizontally-flipped clip predicts the correct sign and the")
            P("     hand streams swap. Fix by flipping frames inside webcam_extract.py BEFORE extraction")
            P("     (visualization mirroring stays separate). Do NOT special-case individual signs.")
        elif raw_ok and not flp_ok:
            P("  OK: raw orientation predicts correctly and flipping breaks it -> convention already matches training.")
        else:
            P(f"  INCONCLUSIVE from one clip: raw -> {raw_top['sign']} ({raw_top['prob']}), "
              f"hflip -> {flp_top['sign']} ({flp_top['prob']}). Repeat with more clips / clearer signs.")
    else:
        P(f"  raw   -> {raw_top['sign']} ({raw_top['prob']})")
        P(f"  hflip -> {flp_top['sign']} ({flp_top['prob']})")
        P(f"  hand-rate swap on flip: {rates_swapped}")
        P("  (pass --label to get an automatic verdict)")
    P("=" * 92)

    text = "\n".join(lines)
    print(text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
