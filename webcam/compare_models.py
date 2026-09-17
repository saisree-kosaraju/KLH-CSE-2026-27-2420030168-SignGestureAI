"""
Step 11 + 12 — ORIGINAL vs WEBCAM-ADAPTED, on the SAME webcam VAL and INCLUDE VAL.

    ./venv/Scripts/python.exe -m webcam.compare_models
    ./venv/Scripts/python.exe -m webcam.compare_models --adapted models/checkpoints/include262_webcam_adapted_v1.pt

Reports for each model: webcam Top-1/Top-5/macro-F1, rejection rate,
accepted-prediction accuracy, INCLUDE VAL Top-1/Top-5, and webcam performance
sliced by background / lighting / camera_distance / session.

Writes results/webcam_model_comparison_<ts>.txt + .json. Never touches INCLUDE TEST.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from webcam import _analysis as A

config = A.config
INCLUDE_MANIFEST = config.METADATA / "include262_model_features64_manifest_v1.csv"
WEBCAM_MANIFEST = config.WEBCAM_DATA / "model_features_manifest.csv"
DEFAULT_ADAPTED = config.MODELS / "checkpoints" / "include262_webcam_adapted_v1.pt"

from model_loader import load_script_module  # noqa: E402
import rejection  # noqa: E402

_s82 = load_script_module(config.SCRIPT_82, "include262_v2_compare")


def _webcam_rows():
    rows = []
    if not WEBCAM_MANIFEST.exists():
        return rows
    with open(WEBCAM_MANIFEST, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "val" and r["status"] == "ready" and Path(r["feature_path"]).exists():
                rows.append(r)
    return rows


def _include_val_rows():
    import pandas as pd
    m = pd.read_csv(INCLUDE_MANIFEST)
    m = m[(m["status"] == "ready") & (m["split"] == "val")]
    return [{"feature_path": p, "class_id": int(c)} for p, c in zip(m["feature_path"], m["class_id"])
            if Path(p).exists()]


def _load(variant_path):
    model = _s82.IncludePartAwareConvHTMAV2()
    ck = torch.load(variant_path, map_location="cpu", weights_only=False)
    state = ck.get("ema_state") or ck.get("model_state") or ck
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def _score(model, rows, cmap):
    out = []
    for r in rows:
        with np.load(r["feature_path"], allow_pickle=False) as d:
            lm = torch.from_numpy(d["landmark_stream"].astype(np.float32))[None]
            gm = torch.from_numpy(d["geometry_stream"].astype(np.float32))[None]
            av = torch.from_numpy(d["availability_mask"].astype(np.float32))[None]
        probs = torch.softmax(model(lm, gm, av), dim=1)[0]
        p5, i5 = probs.topk(5)
        i5 = [int(x) for x in i5]
        top5 = [{"class_id": c, "sign": cmap[c], "probability": float(x)} for c, x in zip(i5, p5)]
        v = rejection.evaluate(top5)
        out.append({"true": int(r["class_id"]), "pred": i5[0], "top5": i5,
                    "accepted": v["accepted"], **{k: r.get(k, "") for k in
                    ("background_id", "lighting", "camera_distance", "session_id")}})
    return out


def _agg(scored, n_classes_hint=None):
    n = len(scored)
    if n == 0:
        return {"n": 0}
    top1 = sum(s["pred"] == s["true"] for s in scored) / n
    top5 = sum(s["true"] in s["top5"] for s in scored) / n
    classes = sorted({s["true"] for s in scored})
    f1s = []
    for c in classes:
        tp = sum(s["true"] == c and s["pred"] == c for s in scored)
        fp = sum(s["true"] != c and s["pred"] == c for s in scored)
        fn = sum(s["true"] == c and s["pred"] != c for s in scored)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    acc_n = sum(s["accepted"] for s in scored)
    acc_correct = sum(s["accepted"] and s["pred"] == s["true"] for s in scored)
    return {
        "n": n, "top1": round(100 * top1, 2), "top5": round(100 * top5, 2),
        "macro_f1": round(100 * float(np.mean(f1s)), 2) if f1s else None,
        "rejection_rate": round(1 - acc_n / n, 3),
        "accepted_pred_acc": round(acc_correct / acc_n, 3) if acc_n else None,
    }


def _by(scored, key):
    groups = defaultdict(list)
    for s in scored:
        groups[s.get(key) or "unset"].append(s)
    return {g: _agg(v) for g, v in sorted(groups.items())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default=str(config.CHECKPOINT))
    ap.add_argument("--adapted", default=str(DEFAULT_ADAPTED))
    args = ap.parse_args(argv)

    web_rows = _webcam_rows()
    inc_rows = _include_val_rows()
    if not web_rows:
        raise SystemExit("No webcam VAL features. Run webcam.preprocess_webcam first.")
    if not Path(args.adapted).exists():
        raise SystemExit(f"Adapted checkpoint not found: {args.adapted}\nRun webcam.adapt_train first.")

    cmap = A.inference._load_class_map()
    ts = time.strftime("%Y%m%d_%H%M%S")

    report = {}
    for name, path in (("original", args.original), ("adapted", args.adapted)):
        model = _load(path)
        web = _score(model, web_rows, cmap)
        inc = _score(model, inc_rows, cmap)
        report[name] = {
            "checkpoint": Path(path).name,
            "webcam_val": _agg(web),
            "include_val": _agg(inc),
            "webcam_by_background": _by(web, "background_id"),
            "webcam_by_lighting": _by(web, "lighting"),
            "webcam_by_distance": _by(web, "camera_distance"),
            "webcam_by_session": _by(web, "session_id"),
        }

    o, a = report["original"], report["adapted"]
    L = []
    P = L.append
    P("=" * 88)
    P(f"ORIGINAL vs WEBCAM-ADAPTED   {ts}")
    P("=" * 88)
    P(f"webcam VAL clips: {o['webcam_val']['n']}   INCLUDE VAL clips: {o['include_val']['n']}")
    P("")
    P(f"{'metric':<26}{'ORIGINAL':>12}{'ADAPTED':>12}{'delta':>10}")
    P("-" * 88)

    def cmp(label, ov, av):
        d = (av - ov) if (ov is not None and av is not None) else None
        P(f"{label:<26}{('%.2f' % ov) if ov is not None else '-':>12}"
          f"{('%.2f' % av) if av is not None else '-':>12}"
          f"{('%+.2f' % d) if d is not None else '-':>10}")

    cmp("webcam Top-1", o["webcam_val"]["top1"], a["webcam_val"]["top1"])
    cmp("webcam Top-5", o["webcam_val"]["top5"], a["webcam_val"]["top5"])
    cmp("webcam macro-F1", o["webcam_val"]["macro_f1"], a["webcam_val"]["macro_f1"])
    cmp("webcam rejection rate", o["webcam_val"]["rejection_rate"] * 100, a["webcam_val"]["rejection_rate"] * 100)
    acc_o = o["webcam_val"]["accepted_pred_acc"]
    acc_a = a["webcam_val"]["accepted_pred_acc"]
    cmp("webcam accepted-pred acc", acc_o * 100 if acc_o else None, acc_a * 100 if acc_a else None)
    cmp("INCLUDE val Top-1", o["include_val"]["top1"], a["include_val"]["top1"])
    cmp("INCLUDE val Top-5", o["include_val"]["top5"], a["include_val"]["top5"])
    P("")

    for slice_key, title in (("webcam_by_background", "BY BACKGROUND"),
                             ("webcam_by_lighting", "BY LIGHTING"),
                             ("webcam_by_distance", "BY CAMERA DISTANCE"),
                             ("webcam_by_session", "BY SESSION")):
        P(f"-- {title} (webcam Top-1, n) --")
        keys = sorted(set(o[slice_key]) | set(a[slice_key]))
        for k in keys:
            ov = o[slice_key].get(k, {})
            av = a[slice_key].get(k, {})
            P(f"   {k:<22} original {ov.get('top1', '-')!s:>7} (n={ov.get('n', 0)})   "
              f"adapted {av.get('top1', '-')!s:>7} (n={av.get('n', 0)})")
        P("")

    verdict = "IMPROVED" if (a["webcam_val"]["top1"] or 0) > (o["webcam_val"]["top1"] or 0) else "NOT IMPROVED"
    inc_drop = (o["include_val"]["top1"] or 0) - (a["include_val"]["top1"] or 0)
    P("=" * 88)
    P(f"VERDICT: webcam validation {verdict}   "
      f"(INCLUDE val {'preserved' if inc_drop <= 2 else f'DROPPED {inc_drop:.2f} pts'})")
    P("Deploy the adapted checkpoint only if webcam validation clearly improved and")
    P("INCLUDE val did not drop more than ~2 points.")
    P("=" * 88)

    text = "\n".join(L)
    print(text)
    base = config.RESULTS / f"webcam_model_comparison_{ts}"
    base.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{base}.txt").write_text(text, encoding="utf-8")
    Path(f"{base}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {base}.txt / .json")


if __name__ == "__main__":
    main()
