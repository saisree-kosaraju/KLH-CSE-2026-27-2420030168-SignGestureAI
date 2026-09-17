from pathlib import Path
import argparse
import math
import random
import time
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


ROOT = Path.cwd()
META = ROOT / "metadata"
MODELS = ROOT / "models" / "checkpoints"
RESULTS = ROOT / "results"

MANIFEST_CSV = META / "include262_model_features64_manifest_v1.csv"
SCHEMA_JSON = META / "include262_model_features64_schema_v1.json"

NUM_CLASSES = 262
NUM_FRAMES = 64
LANDMARK_DIM = 534
GEOMETRY_DIM = 469
AVAIL_DIM = 4


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Keep deterministic behavior where practical.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def macro_f1_from_predictions(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    f1s = []

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1s.append(f1)

    return float(np.mean(f1s))


def accuracy_topk(logits, targets, ks=(1, 5)):
    max_k = max(ks)
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()

    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    out = []
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum()
        out.append(float(correct_k.item()))

    return out


# =============================================================================
# Dataset
# =============================================================================

class IncludeFeatureDataset(Dataset):
    def __init__(
        self,
        dataframe,
        training=False,
        feature_noise_std=0.01,
        frame_dropout_prob=0.05,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.training = training
        self.feature_noise_std = feature_noise_std
        self.frame_dropout_prob = frame_dropout_prob

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        with np.load(row["feature_path"], allow_pickle=False) as data:
            landmark = data["landmark_stream"].astype(np.float32)
            geometry = data["geometry_stream"].astype(np.float32)
            availability = data["availability_mask"].astype(np.float32)

        if landmark.shape != (NUM_FRAMES, LANDMARK_DIM):
            raise RuntimeError(
                f"Bad landmark shape {landmark.shape} in {row['feature_path']}"
            )

        if geometry.shape != (NUM_FRAMES, GEOMETRY_DIM):
            raise RuntimeError(
                f"Bad geometry shape {geometry.shape} in {row['feature_path']}"
            )

        if availability.shape != (NUM_FRAMES, AVAIL_DIM):
            raise RuntimeError(
                f"Bad availability shape {availability.shape} in {row['feature_path']}"
            )

        if self.training:
            # Small Gaussian feature noise improves robustness without changing
            # the underlying sign identity.
            if self.feature_noise_std > 0:
                landmark = landmark + np.random.normal(
                    0.0,
                    self.feature_noise_std,
                    landmark.shape,
                ).astype(np.float32)

                geometry = geometry + np.random.normal(
                    0.0,
                    self.feature_noise_std,
                    geometry.shape,
                ).astype(np.float32)

            # Randomly suppress a few complete temporal tokens during training.
            # Availability is also zeroed so the model knows the token was dropped.
            if self.frame_dropout_prob > 0:
                drop = (
                    np.random.rand(NUM_FRAMES)
                    < self.frame_dropout_prob
                )

                # Never drop every frame.
                if drop.all():
                    drop[np.random.randint(0, NUM_FRAMES)] = False

                landmark[drop] = 0.0
                geometry[drop] = 0.0
                availability[drop] = 0.0

        return (
            torch.from_numpy(landmark),
            torch.from_numpy(geometry),
            torch.from_numpy(availability),
            torch.tensor(int(row["class_id"]), dtype=torch.long),
        )


# =============================================================================
# Model blocks
# =============================================================================

class TemporalConvStem(nn.Module):
    """
    Per-stream temporal Conv1D stem.
    Input:  [B, T, D]
    Output: [B, T, C]
    """

    def __init__(self, input_dim, channels=128, dropout=0.15):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, channels)

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
        )
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_proj(x)

        residual = x

        x = x.transpose(1, 2)  # B,C,T

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)

        x = x.transpose(1, 2)  # B,T,C
        x = self.dropout(x)

        return x + residual


class HeadTokenMixingAttention(nn.Module):
    """
    HTMA-style attention:
      1. multi-head token attention across the temporal sequence
      2. explicit learned mixing across the attention-head dimension

    This is deliberately implemented explicitly rather than using
    nn.MultiheadAttention so the head-mixing operation is visible.
    """

    def __init__(
        self,
        d_model=256,
        num_heads=8,
        attn_dropout=0.10,
        proj_dropout=0.10,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)

        # Learned mixing across H attention heads for every token/head-channel.
        self.head_mixer = nn.Linear(num_heads, num_heads, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        # x: [B,T,C]
        b, t, c = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            b,
            t,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3,B,H,T,Dh

        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * self.scale

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        y = torch.matmul(attn, v)  # B,H,T,Dh

        # Explicit head mixing.
        y = y.permute(0, 2, 3, 1)  # B,T,Dh,H
        y = self.head_mixer(y)
        y = y.permute(0, 3, 1, 2)  # B,H,T,Dh

        y = y.transpose(1, 2).reshape(b, t, c)

        y = self.out_proj(y)
        y = self.proj_dropout(y)

        return y


class HTMABlock(nn.Module):
    def __init__(
        self,
        d_model=256,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.15,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attn = HeadTokenMixingAttention(
            d_model=d_model,
            num_heads=num_heads,
            attn_dropout=dropout,
            proj_dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(d_model)

        hidden = int(d_model * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class IncludeTwoStreamConvHTMA(nn.Module):
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        branch_dim=128,
        d_model=256,
        num_heads=8,
        num_layers=3,
        dropout=0.15,
    ):
        super().__init__()

        self.landmark_stem = TemporalConvStem(
            LANDMARK_DIM,
            channels=branch_dim,
            dropout=dropout,
        )

        self.geometry_stem = TemporalConvStem(
            GEOMETRY_DIM,
            channels=branch_dim,
            dropout=dropout,
        )

        self.availability_proj = nn.Sequential(
            nn.Linear(AVAIL_DIM, 16),
            nn.GELU(),
        )

        fusion_dim = branch_dim * 2 + 16

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, NUM_FRAMES + 1, d_model)
        )

        self.blocks = nn.ModuleList(
            [
                HTMABlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(
            self.pos_embed,
            std=0.02,
        )
        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(
                    m.weight,
                    std=0.02,
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, landmark, geometry, availability):
        l = self.landmark_stem(landmark)
        g = self.geometry_stem(geometry)
        a = self.availability_proj(availability)

        x = torch.cat([l, g, a], dim=-1)
        x = self.fusion(x)

        b = x.size(0)

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed[:, :x.size(1)]

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        cls_out = x[:, 0]
        logits = self.classifier(cls_out)

        return logits


# =============================================================================
# Training / evaluation
# =============================================================================

def build_class_weights(train_df):
    counts = (
        train_df["class_id"]
        .value_counts()
        .reindex(range(NUM_CLASSES), fill_value=0)
        .to_numpy(dtype=np.float32)
    )

    if (counts == 0).any():
        missing = np.where(counts == 0)[0].tolist()
        raise RuntimeError(
            f"Training split has classes with zero examples: {missing}"
        )

    # Gentle balancing: inverse square-root frequency.
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    use_amp=False,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    top1_correct = 0.0
    top5_correct = 0.0
    total = 0

    y_true = []
    y_pred = []

    for landmark, geometry, availability, targets in loader:
        landmark = landmark.to(
            device,
            non_blocking=True,
        )
        geometry = geometry.to(
            device,
            non_blocking=True,
        )
        availability = availability.to(
            device,
            non_blocking=True,
        )
        targets = targets.to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(use_amp and device.type == "cuda"),
            ):
                logits = model(
                    landmark,
                    geometry,
                    availability,
                )

                loss = criterion(
                    logits,
                    targets,
                )

            if training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                    )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                    )

                    optimizer.step()

        batch_size = targets.size(0)
        total += batch_size
        running_loss += float(loss.item()) * batch_size

        t1, t5 = accuracy_topk(
            logits.detach(),
            targets,
            ks=(1, 5),
        )

        top1_correct += t1
        top5_correct += t5

        pred = logits.argmax(dim=1)

        y_true.extend(
            targets.detach().cpu().numpy().tolist()
        )
        y_pred.extend(
            pred.detach().cpu().numpy().tolist()
        )

    loss_avg = running_loss / max(1, total)
    top1 = 100.0 * top1_correct / max(1, total)
    top5 = 100.0 * top5_correct / max(1, total)
    macro_f1 = 100.0 * macro_f1_from_predictions(
        y_true,
        y_pred,
        NUM_CLASSES,
    )

    return {
        "loss": loss_avg,
        "top1": top1,
        "top5": top5,
        "macro_f1": macro_f1,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Keep 0 on Windows unless you know multiprocessing works.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="include262_conv1d_htma_baseline_v1",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Tiny pipeline test: 2 epochs and a small subset.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"Model-feature manifest not found:\n{MANIFEST_CSV}"
        )

    manifest = pd.read_csv(MANIFEST_CSV)

    ready = manifest[
        manifest["status"] == "ready"
    ].copy()

    train_df = ready[
        ready["split"] == "train"
    ].copy()

    val_df = ready[
        ready["split"] == "val"
    ].copy()

    test_df = ready[
        ready["split"] == "test"
    ].copy()

    if len(train_df) != 3121:
        raise RuntimeError(
            f"Expected 3121 train rows, found {len(train_df)}."
        )

    if len(val_df) != 347:
        raise RuntimeError(
            f"Expected 347 val rows, found {len(val_df)}."
        )

    if len(test_df) != 816:
        raise RuntimeError(
            f"Expected 816 test rows, found {len(test_df)}."
        )

    # IMPORTANT: TEST is intentionally not loaded into a DataLoader.
    # It remains untouched until model design is frozen.
    del test_df

    if args.smoke_test:
        args.epochs = 2
        args.patience = 2

        # Smoke test must still contain ALL 262 classes because the loss uses
        # a 262-class weight vector. First take one example from every class,
        # then fill the remaining slots randomly.
        smoke_n = min(384, len(train_df))

        one_per_class = (
            train_df.groupby("class_id", group_keys=False)
            .sample(n=1, random_state=args.seed)
        )

        remaining_n = smoke_n - len(one_per_class)

        if remaining_n > 0:
            remaining_pool = train_df.drop(index=one_per_class.index)
            extra = remaining_pool.sample(
                n=min(remaining_n, len(remaining_pool)),
                random_state=args.seed,
            )
            train_df = pd.concat(
                [one_per_class, extra],
                axis=0,
            ).sample(
                frac=1.0,
                random_state=args.seed,
            ).reset_index(drop=True)
        else:
            train_df = one_per_class.reset_index(drop=True)

        # Validation is only for checking that forward/evaluation code works.
        # It does not need to contain all 262 classes during a smoke test.
        val_df = val_df.sample(
            n=min(160, len(val_df)),
            random_state=args.seed,
        ).reset_index(drop=True)

        smoke_classes = train_df["class_id"].nunique()
        if smoke_classes != NUM_CLASSES:
            raise RuntimeError(
                f"Smoke-test train subset contains {smoke_classes} classes; "
                f"expected {NUM_CLASSES}."
            )

        if args.run_name == "include262_conv1d_htma_baseline_v1":
            args.run_name = "include262_conv1d_htma_smoke"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = device.type == "cuda"

    print("=" * 92)
    print("INCLUDE STEP 18 - TWO-STREAM Conv1D + HTMA BASELINE")
    print("=" * 92)
    print(f"Device        : {device}")
    print(
        f"CUDA available: {torch.cuda.is_available()}"
    )
    if torch.cuda.is_available():
        print(
            f"GPU           : {torch.cuda.get_device_name(0)}"
        )
    print(f"Train videos  : {len(train_df)}")
    print(f"Val videos    : {len(val_df)}")
    print("Test videos   : LOCKED / NOT USED")
    print(f"Classes       : {NUM_CLASSES}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Learning rate : {args.lr}")
    print(f"Run name      : {args.run_name}")
    print()

    train_ds = IncludeFeatureDataset(
        train_df,
        training=True,
        feature_noise_std=0.01,
        frame_dropout_prob=0.05,
    )

    val_ds = IncludeFeatureDataset(
        val_df,
        training=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = IncludeTwoStreamConvHTMA(
        num_classes=NUM_CLASSES,
        branch_dim=128,
        d_model=256,
        num_heads=8,
        num_layers=3,
        dropout=0.15,
    ).to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {params / 1e6:.2f} M"
    )

    class_weights = build_class_weights(train_df).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.10,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=1e-6,
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if use_amp
        else None
    )

    checkpoint_path = MODELS / f"{args.run_name}_best.pt"
    last_checkpoint_path = MODELS / f"{args.run_name}_last.pt"
    history_path = RESULTS / f"{args.run_name}_history.csv"
    summary_path = RESULTS / f"{args.run_name}_summary.txt"
    config_path = RESULTS / f"{args.run_name}_config.json"

    start_epoch = 1
    best_val_top1 = -1.0
    best_val_f1 = -1.0
    bad_epochs = 0

    history = []

    if args.resume and last_checkpoint_path.exists():
        ckpt = torch.load(
            last_checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            ckpt["model_state"]
        )
        optimizer.load_state_dict(
            ckpt["optimizer_state"]
        )
        scheduler.load_state_dict(
            ckpt["scheduler_state"]
        )

        start_epoch = int(ckpt["epoch"]) + 1
        best_val_top1 = float(
            ckpt.get("best_val_top1", -1.0)
        )
        best_val_f1 = float(
            ckpt.get("best_val_f1", -1.0)
        )
        bad_epochs = int(
            ckpt.get("bad_epochs", 0)
        )

        if history_path.exists():
            history = pd.read_csv(
                history_path
            ).to_dict("records")

        print(
            f"Resumed from epoch {start_epoch - 1}"
        )

    config = {
        "architecture": "TwoStream Conv1D + HTMA",
        "num_classes": NUM_CLASSES,
        "num_frames": NUM_FRAMES,
        "landmark_dim": LANDMARK_DIM,
        "geometry_dim": GEOMETRY_DIM,
        "availability_dim": AVAIL_DIM,
        "branch_dim": 128,
        "d_model": 256,
        "num_heads": 8,
        "num_htma_layers": 3,
        "dropout": 0.15,
        "label_smoothing": 0.10,
        "class_weighting": "inverse_sqrt_frequency",
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "test_used_during_training": False,
    }

    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    started = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
        )

        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        improved = (
            val_metrics["top1"] > best_val_top1
            or (
                math.isclose(
                    val_metrics["top1"],
                    best_val_top1,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                and val_metrics["macro_f1"] > best_val_f1
            )
        )

        if improved:
            best_val_top1 = val_metrics["top1"]
            best_val_f1 = val_metrics["macro_f1"]
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": config,
                    "best_val_top1": best_val_top1,
                    "best_val_f1": best_val_f1,
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "val_macro_f1": val_metrics["macro_f1"],
            "best_val_top1": best_val_top1,
            "best_val_macro_f1": best_val_f1,
            "bad_epochs": bad_epochs,
            "epoch_minutes": (time.time() - epoch_start) / 60.0,
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_top1": best_val_top1,
                "best_val_f1": best_val_f1,
                "bad_epochs": bad_epochs,
                "config": config,
            },
            last_checkpoint_path,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss={train_metrics['loss']:.4f} "
            f"Top1={train_metrics['top1']:.2f}% "
            f"Top5={train_metrics['top5']:.2f}% | "
            f"Val Loss={val_metrics['loss']:.4f} "
            f"Top1={val_metrics['top1']:.2f}% "
            f"Top5={val_metrics['top5']:.2f}% "
            f"F1={val_metrics['macro_f1']:.2f}% | "
            f"Best={best_val_top1:.2f}% | "
            f"{row['epoch_minutes']:.1f} min"
        )

        if bad_epochs >= args.patience:
            print(
                f"\nEarly stopping: no validation Top1 improvement "
                f"for {args.patience} epochs."
            )
            break

    total_minutes = (time.time() - started) / 60.0

    summary = [
        "=" * 92,
        "INCLUDE 262 TWO-STREAM Conv1D + HTMA BASELINE SUMMARY",
        "=" * 92,
        "",
        f"Run name: {args.run_name}",
        f"Device: {device}",
        f"Best validation Top1: {best_val_top1:.2f}%",
        f"Best validation Macro-F1: {best_val_f1:.2f}%",
        f"Total training time: {total_minutes:.1f} min",
        "",
        f"Best checkpoint: {checkpoint_path}",
        f"Last checkpoint: {last_checkpoint_path}",
        f"History: {history_path}",
        f"Config: {config_path}",
        "",
        "TEST SPLIT STATUS:",
        "LOCKED - NOT EVALUATED",
        "",
        "NEXT:",
        "Analyze validation confusions and ablations before touching the test split.",
    ]

    summary_path.write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("\n" + "\n".join(summary))
    print("=" * 92)


if __name__ == "__main__":
    main()