"""
Step 9 (OPTIONAL) — pre-MediaPipe visual robustness augmentation.

Only useful if diagnostics show MediaPipe landmark extraction itself degrades under
lighting/background changes (compare `left/right/pose/face` coverage across
backgrounds in the webcam baseline). It creates MILD transformed copies of the
webcam TRAIN clips so the landmark extractor sees varied appearance.

    ./venv/Scripts/python.exe -m webcam.visual_augment --variants 2
    ./venv/Scripts/python.exe -m webcam.visual_augment --variants 2 --effects brightness,contrast,gamma

Writes  webcam_data/train/<clip>__augK.mp4  + sidecars  (augmented: true).
NEVER touches webcam_data/val. After running, re-run:
    ./venv/Scripts/python.exe -m webcam.preprocess_webcam
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from webcam import _analysis as A

config = A.config
TRAIN_DIR = config.WEBCAM_DATA / "train"
VIDEO_EXT = {".mp4", ".mov", ".webm"}

# conservative ranges — never extreme
_EFFECTS = {
    "brightness": lambda f, r: np.clip(f.astype(np.float32) * (1.0 + r * 0.22), 0, 255).astype(np.uint8),
    "contrast": lambda f, r: np.clip((f.astype(np.float32) - 128) * (1.0 + r * 0.20) + 128, 0, 255).astype(np.uint8),
    "gamma": lambda f, r: (((f / 255.0) ** (1.0 / (1.0 + r * 0.30))) * 255).astype(np.uint8),
    "blur": lambda f, r: cv2.GaussianBlur(f, (0, 0), 0.6 + abs(r) * 1.1),
    "noise": lambda f, r: np.clip(
        f.astype(np.float32) + np.random.normal(0, 3.0 + abs(r) * 4.0, f.shape), 0, 255
    ).astype(np.uint8),
}


def _augment_clip(src: Path, out: Path, effects: list[str], rng: np.random.Generator):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    chosen = [(e, float(rng.uniform(-1, 1))) for e in effects]
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        for name, r in chosen:
            fr = _EFFECTS[name](fr, r)
        vw.write(fr)
    cap.release()
    vw.release()
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=2, help="augmented copies per train clip")
    ap.add_argument("--effects", default="brightness,contrast,gamma,blur,noise")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    effects = [e.strip() for e in args.effects.split(",") if e.strip() in _EFFECTS]
    if not effects:
        raise SystemExit(f"no valid effects; choose from {sorted(_EFFECTS)}")
    if not TRAIN_DIR.exists():
        raise SystemExit(f"No webcam train clips: {TRAIN_DIR}")

    rng = np.random.default_rng(args.seed)
    originals = [p for p in sorted(TRAIN_DIR.iterdir())
                 if p.suffix.lower() in VIDEO_EXT and "__aug" not in p.stem]
    print(f"{len(originals)} original train clips -> {args.variants} variant(s) each  effects={effects}")

    made = 0
    for src in originals:
        sidecar = src.with_suffix(src.suffix + ".json")
        meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        for k in range(1, args.variants + 1):
            out = src.with_name(f"{src.stem}__aug{k}.mp4")
            if out.exists() and not args.overwrite:
                continue
            chosen = _augment_clip(src, out, rng.choice(effects, size=min(2, len(effects)), replace=False).tolist(), rng)
            if not chosen:
                continue
            child = dict(meta)
            child.update({
                "filename": out.name, "augmented": True, "augmented_from": src.name,
                "aug_effects": [f"{n}:{round(r, 2)}" for n, r in chosen],
                "coverage_source": "browser_parent", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            (out.with_suffix(out.suffix + ".json")).write_text(json.dumps(child, indent=2), encoding="utf-8")
            made += 1
            print(f"  {out.name}  ({', '.join(f'{n}' for n, _ in chosen)})")

    print(f"\ncreated {made} augmented train clips. Now re-run:")
    print("  ./venv/Scripts/python.exe -m webcam.preprocess_webcam")


if __name__ == "__main__":
    main()
