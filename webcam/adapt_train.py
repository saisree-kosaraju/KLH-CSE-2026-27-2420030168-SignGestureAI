"""
Webcam DOMAIN ADAPTATION of the frozen INCLUDE262 model.

    init  : models/checkpoints/include262_conv1d_htma_v2_seed42_best.pt  (ema_state)
    train : INCLUDE TRAIN features  +  WEBCAM TRAIN features   (balanced sampling)
    val   : INCLUDE VAL  and  WEBCAM VAL   (reported separately; TEST never touched)
    out   : models/checkpoints/include262_webcam_adapted_v1.pt   (original never overwritten)

Two conservative stages:
    A  freeze the 5 part encoders, train fusion/gates/temporal/classifier   (low LR, few epochs)
    B  unfreeze everything                                                  (even lower LR, early stop)

Checkpoint selection: best WEBCAM VAL top-1, subject to
    INCLUDE VAL top-1 >= (original INCLUDE VAL top-1) - include_tolerance
so the adapted model cannot catastrophically forget the original task.

Usage:
    ./venv/Scripts/python.exe -m webcam.adapt_train                    # full run
    ./venv/Scripts/python.exe -m webcam.adapt_train --smoke            # 2+2 epochs, tiny subset
    ./venv/Scripts/python.exe -m webcam.adapt_train --webcam-frac 0.5 --epochs-a 8 --epochs-b 14

Requires  `python -m webcam.preprocess_webcam`  to have produced webcam features.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from webcam import _analysis as A
from webcam import augment as aug

config = A.config
ROOT = config.ROOT
CKPT_ORIGINAL = config.CHECKPOINT
CKPT_ADAPTED = config.MODELS / "checkpoints" / "include262_webcam_adapted_v1.pt"
INCLUDE_MANIFEST = config.METADATA / "include262_model_features64_manifest_v1.csv"
WEBCAM_MANIFEST = config.WEBCAM_DATA / "model_features_manifest.csv"
RESULTS = config.RESULTS

NUM_CLASSES = config.NUM_CLASSES
PART_ENCODERS = ("left_encoder", "right_encoder", "pose_encoder", "face_encoder", "geometry_encoder")

# architecture + EMA + metrics come from the frozen training script, exactly like
# the backend model_loader does — never re-declared here.
from model_loader import load_script_module  # on sys.path via webcam._analysis

_s82 = load_script_module(config.SCRIPT_82, "include262_train_v2_adapt")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class FeatureDataset(Dataset):
    def __init__(self, rows: list[dict], training: bool, aug_cfg: aug.AugConfig):
        self.rows = rows
        self.training = training
        self.aug_cfg = aug_cfg if training else aug.NO_AUG

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        with np.load(r["feature_path"], allow_pickle=False) as d:
            lm = d["landmark_stream"].astype(np.float32)
            gm = d["geometry_stream"].astype(np.float32)
            av = d["availability_mask"].astype(np.float32)
        if self.training and self.aug_cfg.enabled:
            lm, gm, av = aug.augment(lm, gm, av, self.aug_cfg)
        return (
            torch.from_numpy(np.ascontiguousarray(lm)),
            torch.from_numpy(np.ascontiguousarray(gm)),
            torch.from_numpy(np.ascontiguousarray(av)),
            torch.tensor(int(r["class_id"]), dtype=torch.long),
        )


def _include_rows(split: str) -> list[dict]:
    m = pd.read_csv(INCLUDE_MANIFEST)
    m = m[(m["status"] == "ready") & (m["split"] == split)]
    return [{"feature_path": p, "class_id": int(c), "source": "include",
             "background_id": "include", "lighting": "include", "session_id": "include"}
            for p, c in zip(m["feature_path"], m["class_id"]) if Path(p).exists()]


def _webcam_rows(split: str) -> list[dict]:
    if not WEBCAM_MANIFEST.exists():
        return []
    out = []
    with open(WEBCAM_MANIFEST, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == split and r["status"] == "ready" and Path(r["feature_path"]).exists():
                out.append({"feature_path": r["feature_path"], "class_id": int(r["class_id"]),
                            "source": "webcam", "background_id": r.get("background_id", ""),
                            "lighting": r.get("lighting", ""), "session_id": r.get("session_id", "")})
    return out


def _balanced_sampler(rows: list[dict], webcam_frac: float) -> WeightedRandomSampler:
    """Per-sample weights so a drawn batch is ~webcam_frac webcam, class-balanced within each source."""
    src = np.array([1 if r["source"] == "webcam" else 0 for r in rows])
    n_web, n_inc = int(src.sum()), int((src == 0).sum())
    w = np.zeros(len(rows), dtype=np.float64)
    if n_web:
        w[src == 1] = webcam_frac / n_web
    if n_inc:
        w[src == 0] = (1.0 - webcam_frac) / n_inc if n_web else 1.0 / n_inc
    # gentle within-source class balancing
    cls = np.array([r["class_id"] for r in rows])
    for s in (0, 1):
        mask = src == s
        if not mask.any():
            continue
        cc = pd.Series(cls[mask]).value_counts()
        w[mask] *= (1.0 / np.sqrt(cc.reindex(cls[mask]).to_numpy()))
    w = w / w.sum()
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(rows), replacement=True)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, rows: list[dict], device, batch_size=64) -> dict:
    if not rows:
        return {"n": 0, "top1": float("nan"), "top5": float("nan"), "macro_f1": float("nan")}
    model.eval()
    ds = FeatureDataset(rows, training=False, aug_cfg=aug.NO_AUG)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    yt, yp = [], []
    c1 = c5 = 0
    for lm, gm, av, y in loader:
        logits = model(lm.to(device), gm.to(device), av.to(device))
        b1, b5 = _s82.topk_counts(logits, y.to(device))
        c1 += b1
        c5 += b5
        yt.extend(y.tolist())
        yp.extend(logits.argmax(1).cpu().tolist())
    n = len(rows)
    return {
        "n": n,
        "top1": round(100.0 * c1 / n, 2),
        "top5": round(100.0 * c5 / n, 2),
        "macro_f1": round(100.0 * _s82.macro_f1(yt, yp), 2),
    }


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def set_freeze(model, freeze_encoders: bool):
    for name in PART_ENCODERS:
        for p in getattr(model, name).parameters():
            p.requires_grad_(not freeze_encoders)


def cosine_lr(optimizer, epoch, epochs, base_lr, min_lr, warmup):
    if epoch <= warmup:
        lr = base_lr * epoch / max(1, warmup)
    else:
        prog = (epoch - warmup) / max(1, epochs - warmup)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def run_stage(stage, model, ema, train_rows, inc_val, web_val, device, args, aug_cfg,
              inc_val_floor, history, best):
    freeze = stage == "A"
    set_freeze(model, freeze_encoders=freeze)
    epochs = args.epochs_a if freeze else args.epochs_b
    base_lr = args.lr_a if freeze else args.lr_b
    warmup = 2 if freeze else 1

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss(label_smoothing=0.08)

    sampler = _balanced_sampler(train_rows, args.webcam_frac)
    loader = DataLoader(FeatureDataset(train_rows, True, aug_cfg), batch_size=args.batch_size,
                        sampler=sampler, num_workers=0)

    bad = 0
    for ep in range(1, epochs + 1):
        lr = cosine_lr(opt, ep, epochs, base_lr, args.min_lr, warmup)
        model.train()
        loss_sum = seen = 0
        for lm, gm, av, y in loader:
            lm, gm, av, y = lm.to(device), gm.to(device), av.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(lm, gm, av), y)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ema.update(model)
            loss_sum += loss.item() * len(y)
            seen += len(y)

        ev_inc = evaluate(ema.model, inc_val, device)
        ev_web = evaluate(ema.model, web_val, device)
        keep = (ev_web["top1"] == ev_web["top1"]) and ev_web["top1"] > best["webcam_top1"] and \
               (not (ev_inc["top1"] == ev_inc["top1"]) or ev_inc["top1"] >= inc_val_floor)
        row = {"stage": stage, "epoch": ep, "lr": round(lr, 8),
               "train_loss": round(loss_sum / max(seen, 1), 4),
               "include_val_top1": ev_inc["top1"], "include_val_top5": ev_inc["top5"],
               "webcam_val_top1": ev_web["top1"], "webcam_val_top5": ev_web["top5"],
               "webcam_val_macro_f1": ev_web["macro_f1"], "kept": keep}
        history.append(row)
        print(f"  [{stage}{ep:02d}] loss {row['train_loss']:.3f}  lr {lr:.2e}  "
              f"INCLUDE val {ev_inc['top1']}/{ev_inc['top5']}  "
              f"WEBCAM val {ev_web['top1']}/{ev_web['top5']} (F1 {ev_web['macro_f1']})"
              f"{'   <- best' if keep else ''}")
        if keep:
            best.update({"webcam_top1": ev_web["top1"], "webcam_top5": ev_web["top5"],
                         "webcam_macro_f1": ev_web["macro_f1"], "include_top1": ev_inc["top1"],
                         "include_top5": ev_inc["top5"], "stage": stage, "epoch": ep,
                         "ema_state": copy.deepcopy(ema.model.state_dict())})
            bad = 0
        else:
            bad += 1
            if not freeze and bad >= args.patience:
                print(f"  early stop (no webcam-val improvement for {args.patience} epochs)")
                break


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam-frac", type=float, default=0.5, help="target webcam fraction per batch")
    ap.add_argument("--epochs-a", type=int, default=8)
    ap.add_argument("--epochs-b", type=int, default=14)
    ap.add_argument("--lr-a", type=float, default=3e-4)
    ap.add_argument("--lr-b", type=float, default=4e-5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--include-tolerance", type=float, default=2.0,
                    help="max allowed INCLUDE-val top-1 drop vs the original checkpoint")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(CKPT_ADAPTED))
    args = ap.parse_args(argv)

    _s82.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.out)
    if args.smoke and out_path == CKPT_ADAPTED:
        out_path = CKPT_ADAPTED.with_name("include262_webcam_adapted_smoke.pt")

    inc_train = _include_rows("train")
    inc_val = _include_rows("val")
    web_train = _webcam_rows("train")
    web_val = _webcam_rows("val")

    if not web_train or not web_val:
        raise SystemExit(
            "No webcam features. Collect clips with the Data collection tab, then:\n"
            "  ./venv/Scripts/python.exe -m webcam.preprocess_webcam\n"
            f"(expected {WEBCAM_MANIFEST})"
        )

    if args.smoke:
        args.epochs_a, args.epochs_b, args.patience = 2, 2, 2
        rng = np.random.default_rng(args.seed)
        inc_train = list(rng.choice(inc_train, size=min(400, len(inc_train)), replace=False))
        inc_val = list(rng.choice(inc_val, size=min(160, len(inc_val)), replace=False))

    train_rows = inc_train + web_train
    web_classes = sorted({r["class_id"] for r in web_train})
    sessions = sorted({r["session_id"] for r in web_train if r["session_id"]})
    backgrounds = sorted({r["background_id"] for r in web_train if r["background_id"]})

    print("=" * 82)
    print("WEBCAM DOMAIN ADAPTATION")
    print("=" * 82)
    print(f"init checkpoint : {CKPT_ORIGINAL.name} (ema_state)")
    print(f"INCLUDE train/val: {len(inc_train)} / {len(inc_val)}")
    print(f"WEBCAM  train/val: {len(web_train)} / {len(web_val)}   classes={len(web_classes)}")
    print(f"sessions={sessions}  backgrounds={backgrounds}")
    print(f"device={device}  webcam_frac={args.webcam_frac}  aug={'off' if args.no_aug else 'on'}")

    # --- model init from the frozen original ---
    model = _s82.IncludePartAwareConvHTMAV2().to(device)
    ckpt0 = torch.load(CKPT_ORIGINAL, map_location=device, weights_only=False)
    model.load_state_dict(ckpt0["ema_state"], strict=True)
    ema = _s82.EMA(model, decay=0.995)

    orig = _s82.IncludePartAwareConvHTMAV2().to(device)
    orig.load_state_dict(ckpt0["ema_state"], strict=True)
    base_inc = evaluate(orig, inc_val, device)
    base_web = evaluate(orig, web_val, device)
    inc_floor = (base_inc["top1"] - args.include_tolerance) if base_inc["top1"] == base_inc["top1"] else -1
    print(f"\nBASELINE (original checkpoint)")
    print(f"  INCLUDE val : top1 {base_inc['top1']}  top5 {base_inc['top5']}")
    print(f"  WEBCAM  val : top1 {base_web['top1']}  top5 {base_web['top5']}  F1 {base_web['macro_f1']}")
    print(f"  INCLUDE-val floor for a kept checkpoint: {inc_floor}\n")

    aug_cfg = aug.NO_AUG if args.no_aug else aug.DEFAULT_AUG
    history: list[dict] = []
    best = {"webcam_top1": -1.0, "ema_state": None}

    t0 = time.time()
    print("STAGE A - part encoders frozen")
    run_stage("A", model, ema, train_rows, inc_val, web_val, device, args, aug_cfg, inc_floor, history, best)
    print("STAGE B - all layers unfrozen")
    run_stage("B", model, ema, train_rows, inc_val, web_val, device, args, aug_cfg, inc_floor, history, best)
    mins = (time.time() - t0) / 60

    if best["ema_state"] is None:
        print("\nNo checkpoint beat the baseline under the INCLUDE-val guard. Nothing saved.")
        _write_history(history, base_inc, base_web, None, args, mins)
        return

    meta = {
        "source_checkpoint": str(CKPT_ORIGINAL),
        "webcam_train_clips": len(web_train), "webcam_val_clips": len(web_val),
        "classes_represented": web_classes, "num_classes_webcam": len(web_classes),
        "sessions": sessions, "backgrounds": backgrounds,
        "webcam_frac": args.webcam_frac, "lr_a": args.lr_a, "lr_b": args.lr_b,
        "epochs_a": args.epochs_a, "epochs_b": args.epochs_b,
        "augmentation": "off" if args.no_aug else "feature-space (webcam.augment.DEFAULT_AUG)",
        "best_webcam_val_top1": best["webcam_top1"], "best_webcam_val_top5": best.get("webcam_top5"),
        "best_webcam_val_macro_f1": best.get("webcam_macro_f1"),
        "include_val_top1_at_best": best.get("include_top1"),
        "baseline_webcam_val_top1": base_web["top1"], "baseline_include_val_top1": base_inc["top1"],
        "selected_stage": best.get("stage"), "selected_epoch": best.get("epoch"),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema_state": best["ema_state"], "model_state": best["ema_state"],
                "epoch": best.get("epoch", -1), "best_top1": best["webcam_top1"],
                "metadata": meta}, out_path)
    print(f"\nsaved adapted checkpoint: {out_path}")
    print(f"  webcam val  : top1 {base_web['top1']} -> {best['webcam_top1']}")
    print(f"  INCLUDE val : top1 {base_inc['top1']} -> {best.get('include_top1')}")
    _write_history(history, base_inc, base_web, meta, args, mins, out_path)
    print(f"\nnext: compare -> ./venv/Scripts/python.exe -m webcam.compare_models --adapted {out_path}")


def _write_history(history, base_inc, base_web, meta, args, mins, out_path=None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    hp = RESULTS / f"webcam_adapt_v1_history_{ts}.csv"
    if history:
        with open(hp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
    (RESULTS / f"webcam_adapt_v1_config_{ts}.json").write_text(
        json.dumps({"args": vars(args), "baseline_include_val": base_inc,
                    "baseline_webcam_val": base_web, "metadata": meta, "minutes": round(mins, 1),
                    "checkpoint": str(out_path) if out_path else None}, indent=2),
        encoding="utf-8")
    print(f"history: {hp}")


if __name__ == "__main__":
    main()
