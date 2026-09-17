"""
Phase 17 / Module 15 — WEBCAM VALIDATION BASELINE for the ORIGINAL checkpoint.

Evaluates the frozen INCLUDE262 model (ema_state) on the collected webcam VAL set.
This is REAL-WORLD WEBCAM VALIDATION ACCURACY. It is NOT the locked INCLUDE test
result (98.04%) and must never be reported as such.

    python -m webcam.evaluate_webcam_baseline                 # val split, original ckpt
    python -m webcam.evaluate_webcam_baseline --split train   # sanity check on the train clips
    python -m webcam.evaluate_webcam_baseline --checkpoint models/checkpoints/include262_webcam_adapted_v1.pt

Requires `python -m webcam.preprocess_webcam` to have been run first.
Writes results/webcam_baseline_<ts>_{summary.txt,predictions.csv,per_class.csv,confusions.csv,metrics.json}.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from webcam import _analysis as A

config = A.config


def _load_manifest(split: str) -> list[dict]:
    mp = config.WEBCAM_DATA / "model_features_manifest.csv"
    if not mp.exists():
        raise SystemExit(
            "No webcam feature manifest. Run:  python -m webcam.preprocess_webcam\n"
            f"(expected {mp})"
        )
    rows = []
    with open(mp, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == split and r["status"] == "ready" and Path(r["feature_path"]).exists():
                rows.append(r)
    return rows


def _macro_f1(y_true, y_pred) -> float:
    classes = sorted(set(y_true))
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--checkpoint", default=None, help="override checkpoint (default: frozen original)")
    ap.add_argument("--min-per-class-f1", type=int, default=3)
    args = ap.parse_args(argv)

    rows = _load_manifest(args.split)
    if not rows:
        print(f"No usable webcam {args.split} clips yet.")
        print("Collect >= 3 clips for each of ~20-30 supported signs, then:")
        print("  python -m webcam.preprocess_webcam")
        print("  python -m webcam.evaluate_webcam_baseline")
        return

    device = torch.device(config.DEVICE)
    if args.checkpoint:
        from model_loader import load_script_module
        s82 = load_script_module(config.SCRIPT_82, "include262_v2_eval")
        model = s82.IncludePartAwareConvHTMAV2().to(device)
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state.get("ema_state") or state.get("model_state") or state, strict=True)
        variant = Path(args.checkpoint).name
    else:
        model, _ = A.inference._load_model("original")
        variant = config.CHECKPOINT.name
    cmap = A.inference._load_class_map()
    model.eval()

    import rejection

    ts = time.strftime("%Y%m%d_%H%M%S")
    preds = []
    y_true, y_pred = [], []
    top5_hits = 0
    confidences = []
    accepted_correct = accepted_total = 0
    by_group = defaultdict(lambda: [0, 0])  # key -> [correct, total]

    for r in rows:
        with np.load(r["feature_path"], allow_pickle=False) as d:
            lm = torch.from_numpy(d["landmark_stream"].astype(np.float32))[None].to(device)
            gm = torch.from_numpy(d["geometry_stream"].astype(np.float32))[None].to(device)
            av = torch.from_numpy(d["availability_mask"].astype(np.float32))[None].to(device)
        with torch.no_grad():
            probs = torch.softmax(model(lm, gm, av), dim=1)[0]
        p5, i5 = probs.topk(5)
        i5 = [int(x) for x in i5]
        p5 = [float(x) for x in p5]
        true_c = int(r["class_id"])
        pred_c = i5[0]
        top5 = [{"class_id": c, "sign": cmap[c], "probability": pr} for c, pr in zip(i5, p5)]
        verdict = rejection.evaluate(top5)

        y_true.append(true_c)
        y_pred.append(pred_c)
        top5_hits += int(true_c in i5)
        confidences.append(p5[0])
        if verdict["accepted"]:
            accepted_total += 1
            accepted_correct += int(pred_c == true_c)
        for gk in ("background_id", "lighting", "camera_distance", "session_id"):
            g = r.get(gk) or "unset"
            by_group[f"{gk}={g}"][1] += 1
            by_group[f"{gk}={g}"][0] += int(pred_c == true_c)

        preds.append({
            "clip": r["clip"], "true": cmap[true_c], "pred": cmap[pred_c],
            "correct": int(pred_c == true_c), "confidence": round(p5[0], 4),
            "top5": "|".join(f"{cmap[c]}:{pr:.3f}" for c, pr in zip(i5, p5)),
            "accepted": verdict["accepted"], "rejection_reason": verdict["rejection_reason"] or "",
            "background_id": r.get("background_id", ""), "lighting": r.get("lighting", ""),
            "camera_distance": r.get("camera_distance", ""), "session_id": r.get("session_id", ""),
        })

    n = len(rows)
    top1 = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    top5_acc = top5_hits / n
    per_class_n = Counter(y_true)
    enough = [c for c, k in per_class_n.items() if k >= args.min_per_class_f1]
    macro_f1 = _macro_f1(
        [t for t in y_true if t in enough], [p for t, p in zip(y_true, y_pred) if t in enough]
    ) if enough else None

    per_class = []
    for c in sorted(per_class_n):
        idx = [i for i, t in enumerate(y_true) if t == c]
        acc = sum(y_pred[i] == c for i in idx) / len(idx)
        per_class.append({"class_id": c, "sign": cmap[c], "n": len(idx), "accuracy": round(acc, 3)})

    conf_pairs = Counter(
        (cmap[t], cmap[p]) for t, p in zip(y_true, y_pred) if t != p
    ).most_common(20)

    metrics = {
        "label": "WEBCAM VALIDATION ACCURACY (real webcam clips) - NOT the INCLUDE locked test result",
        "checkpoint": variant,
        "split": args.split,
        "n_clips": n,
        "n_classes": len(per_class_n),
        "top1": round(top1, 4),
        "top5": round(top5_acc, 4),
        "macro_f1_ge_%d_per_class" % args.min_per_class_f1: round(macro_f1, 4) if macro_f1 is not None else None,
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "rejection_rate": round(1 - accepted_total / n, 4),
        "accepted_prediction_accuracy": round(accepted_correct / accepted_total, 4) if accepted_total else None,
        "include_locked_test_top1_reference": 0.9804,
        "by_group": {k: {"acc": round(v[0] / v[1], 3), "n": v[1]} for k, v in sorted(by_group.items())},
    }

    base = Path(config.RESULTS) / f"webcam_baseline_{ts}"
    base.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{base}_predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(preds[0].keys()))
        w.writeheader(); w.writerows(preds)
    with open(f"{base}_per_class.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["class_id", "sign", "n", "accuracy"])
        w.writeheader(); w.writerows(per_class)
    with open(f"{base}_confusions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["true", "pred", "count"])
        for (t, p), c in conf_pairs:
            w.writerow([t, p, c])
    Path(f"{base}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        "=" * 80,
        "WEBCAM VALIDATION BASELINE  (original INCLUDE262 checkpoint)",
        "=" * 80,
        "NOTE: this is real-world webcam validation accuracy. It is a SEPARATE metric",
        "      from the locked INCLUDE262 test result (Top-1 = 98.04%).",
        "",
        f"checkpoint            : {variant}",
        f"split                 : {args.split}",
        f"clips / classes       : {n} / {len(per_class_n)}",
        f"Webcam Top-1          : {top1:.4f}",
        f"Webcam Top-5          : {top5_acc:.4f}",
        f"macro-F1 (>={args.min_per_class_f1}/class) : {macro_f1:.4f}" if macro_f1 is not None else
        f"macro-F1              : (need >= {args.min_per_class_f1} clips per class)",
        f"mean confidence       : {np.mean(confidences):.4f}",
        f"rejection rate        : {metrics['rejection_rate']:.4f}",
        f"accepted-pred accuracy: {metrics['accepted_prediction_accuracy']}",
        "",
        "by group (background / lighting / session):",
    ]
    for k, v in metrics["by_group"].items():
        lines.append(f"   {k:<28} acc={v['acc']:<6} n={v['n']}")
    lines += ["", "top confusion pairs (true -> pred):"]
    for (t, p), c in conf_pairs[:12]:
        lines.append(f"   {t:<18} -> {p:<18} x{c}")
    lines += ["", f"artifacts: {base}_*.csv/.json", "=" * 80]

    text = "\n".join(lines)
    print(text)
    Path(f"{base}_summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
