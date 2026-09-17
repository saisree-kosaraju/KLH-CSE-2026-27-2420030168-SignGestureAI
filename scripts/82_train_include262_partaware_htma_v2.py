from pathlib import Path
import argparse
import copy
import importlib.util
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path.cwd()
SCRIPTS = ROOT / "scripts"
META = ROOT / "metadata"
MODELS = ROOT / "models" / "checkpoints"
RESULTS = ROOT / "results"

V1_SCRIPT = SCRIPTS / "80_train_include262_conv1d_htma_baseline_v1_fixed.py"
MANIFEST_CSV = META / "include262_model_features64_manifest_v1.csv"

NUM_CLASSES = 262
NUM_FRAMES = 64

LEFT_DIM = 63
RIGHT_DIM = 63
POSE_DIM = 132
FACE_DIM = 276
GEOMETRY_DIM = 469
AVAIL_DIM = 4


def load_v1_module():
    spec = importlib.util.spec_from_file_location("include_v1", V1_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def macro_f1(y_true, y_pred, num_classes=NUM_CLASSES):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    values = []
    for c in range(num_classes):
        true_mask = y_true == c
        # Only average classes actually present in this evaluation split.
        if true_mask.sum() == 0:
            continue

        pred_mask = y_pred == c
        tp = np.sum(true_mask & pred_mask)
        fp = np.sum(~true_mask & pred_mask)
        fn = np.sum(true_mask & ~pred_mask)

        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        values.append(f1)

    return float(np.mean(values)) if values else 0.0


def shift_no_wrap(x, shift):
    if shift == 0:
        return x

    out = torch.zeros_like(x)

    if shift > 0:
        out[shift:] = x[:-shift]
    else:
        s = -shift
        out[:-s] = x[s:]

    return out


class V2Dataset:
    """
    Uses the exact same verified feature files as V1, but adds slightly
    stronger temporal regularization during training.
    """

    def __init__(self, base_dataset, training=False):
        self.base = base_dataset
        self.training = training

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        landmark, geometry, availability, target = self.base[idx]

        if not self.training:
            return landmark, geometry, availability, target

        # Small temporal boundary jitter. No circular wrapping.
        if random.random() < 0.50:
            shift = random.choice([-2, -1, 0, 1, 2])
            landmark = shift_no_wrap(landmark, shift)
            geometry = shift_no_wrap(geometry, shift)
            availability = shift_no_wrap(availability, shift)

        # Short contiguous temporal mask.
        if random.random() < 0.15:
            width = random.randint(2, 4)
            start = random.randint(0, NUM_FRAMES - width)
            landmark[start:start + width] = 0
            geometry[start:start + width] = 0
            availability[start:start + width] = 0

        # Very light modality dropout to reduce over-reliance on face/geometry.
        if random.random() < 0.05:
            landmark[:, 258:534] = 0
            availability[:, 3] = 0

        if random.random() < 0.03:
            geometry[:] = 0

        return landmark, geometry, availability, target


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels, dilation=1, dropout=0.12):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pw1 = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.pw2 = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        y = self.norm(x).transpose(1, 2)
        y = self.dw(y)
        y = F.gelu(y)
        y = self.pw1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.pw2(y)
        y = y.transpose(1, 2)
        y = self.dropout(y)
        return residual + y


class PartEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.12):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        self.blocks = nn.Sequential(
            ResidualTemporalBlock(output_dim, dilation=1, dropout=dropout),
            ResidualTemporalBlock(output_dim, dilation=2, dropout=dropout),
        )

    def forward(self, x):
        x = self.norm(x)
        x = F.gelu(self.proj(x))
        return self.blocks(x)


class HeadTokenMixingAttention(nn.Module):
    def __init__(
        self,
        d_model=320,
        num_heads=8,
        attn_dropout=0.10,
        proj_dropout=0.10,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.head_mixer = nn.Linear(num_heads, num_heads, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        b, t, c = x.shape

        qkv = self.qkv(x).reshape(
            b, t, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v  # B,H,T,Dh

        # Explicit learned mixing across attention heads.
        y = y.permute(0, 2, 3, 1)  # B,T,Dh,H
        y = self.head_mixer(y)
        y = y.permute(0, 3, 1, 2)  # B,H,T,Dh
        y = y.transpose(1, 2).reshape(b, t, c)

        y = self.proj(y)
        return self.proj_drop(y)


class HTMABlock(nn.Module):
    def __init__(self, d_model=320, num_heads=8, dropout=0.12):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = HeadTokenMixingAttention(
            d_model=d_model,
            num_heads=num_heads,
            attn_dropout=dropout,
            proj_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class IncludePartAwareConvHTMAV2(nn.Module):
    """
    V2 changes versus V1:
      - part-aware landmark encoding (left/right/pose/face separately)
      - multi-scale residual temporal Conv1D blocks
      - dynamic frame-wise branch gates
      - deeper/wider HTMA encoder
      - CLS + learned attention pooling instead of CLS-only
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        d_model=320,
        num_heads=8,
        num_layers=4,
        dropout=0.12,
    ):
        super().__init__()

        self.left_encoder = PartEncoder(LEFT_DIM, 64, dropout)
        self.right_encoder = PartEncoder(RIGHT_DIM, 64, dropout)
        self.pose_encoder = PartEncoder(POSE_DIM, 64, dropout)
        self.face_encoder = PartEncoder(FACE_DIM, 64, dropout)
        self.geometry_encoder = PartEncoder(GEOMETRY_DIM, 128, dropout)

        self.avail_proj = nn.Sequential(
            nn.Linear(AVAIL_DIM, 16),
            nn.GELU(),
        )

        branch_concat_dim = 64 + 64 + 64 + 64 + 128
        fusion_input_dim = branch_concat_dim + 16

        self.gate_net = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, 96),
            nn.GELU(),
            nn.Linear(96, 5),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, NUM_FRAMES + 1, d_model)
        )

        self.blocks = nn.ModuleList([
            HTMABlock(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

        self.pool_fuse = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.20),
        )

        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, landmark, geometry, availability):
        left = landmark[:, :, 0:63]
        right = landmark[:, :, 63:126]
        pose = landmark[:, :, 126:258]
        face = landmark[:, :, 258:534]

        l = self.left_encoder(left)
        r = self.right_encoder(right)
        p = self.pose_encoder(pose)
        f = self.face_encoder(face)
        g = self.geometry_encoder(geometry)
        a = self.avail_proj(availability)

        raw = torch.cat([l, r, p, f, g, a], dim=-1)
        gates = self.gate_net(raw)

        l = l * gates[:, :, 0:1]
        r = r * gates[:, :, 1:2]
        p = p * gates[:, :, 2:3]
        f = f * gates[:, :, 3:4]
        g = g * gates[:, :, 4:5]

        x = torch.cat([l, r, p, f, g, a], dim=-1)
        x = self.fusion(x)

        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.shape[1]]

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        cls_out = x[:, 0]
        tokens = x[:, 1:]

        scores = self.attn_pool(tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        attn_out = torch.sum(tokens * weights.unsqueeze(-1), dim=1)

        pooled = self.pool_fuse(
            torch.cat([cls_out, attn_out], dim=-1)
        )

        return self.classifier(pooled)


class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        src = model.state_dict()
        dst = self.model.state_dict()

        for k, v in dst.items():
            s = src[k].detach()

            if torch.is_floating_point(v):
                v.mul_(self.decay).add_(
                    s,
                    alpha=1.0 - self.decay,
                )
            else:
                v.copy_(s)


def class_weights(train_df):
    counts = (
        train_df["class_id"]
        .value_counts()
        .reindex(range(NUM_CLASSES), fill_value=0)
        .to_numpy(dtype=np.float32)
    )

    if (counts == 0).any():
        raise RuntimeError("Training split contains an empty class.")

    weights = 1.0 / np.sqrt(counts)
    weights /= weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def topk_counts(logits, target):
    _, pred = logits.topk(5, dim=1)
    correct1 = (pred[:, 0] == target).sum().item()
    correct5 = (pred == target[:, None]).any(dim=1).sum().item()
    return correct1, correct5


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    ema=None,
    scaler=None,
):
    training = optimizer is not None
    model.train(training)

    total = 0
    loss_sum = 0.0
    c1 = 0
    c5 = 0
    y_true = []
    y_pred = []

    use_amp = device.type == "cuda"

    for landmark, geometry, availability, target in loader:
        landmark = landmark.to(device, non_blocking=True)
        geometry = geometry.to(device, non_blocking=True)
        availability = availability.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(
                    landmark,
                    geometry,
                    availability,
                )
                loss = criterion(logits, target)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                if ema is not None:
                    ema.update(model)

        batch = target.size(0)
        total += batch
        loss_sum += float(loss.item()) * batch

        b1, b5 = topk_counts(logits.detach(), target)
        c1 += b1
        c5 += b5

        y_true.extend(target.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(1).detach().cpu().numpy().tolist())

    return {
        "loss": loss_sum / total,
        "top1": 100.0 * c1 / total,
        "top5": 100.0 * c5 / total,
        "macro_f1_supported": 100.0 * macro_f1(y_true, y_pred),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total = 0
    loss_sum = 0.0
    c1 = 0
    c5 = 0
    y_true = []
    y_pred = []

    use_amp = device.type == "cuda"

    for landmark, geometry, availability, target in loader:
        landmark = landmark.to(device)
        geometry = geometry.to(device)
        availability = availability.to(device)
        target = target.to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(
                landmark,
                geometry,
                availability,
            )
            loss = criterion(logits, target)

        batch = target.size(0)
        total += batch
        loss_sum += float(loss.item()) * batch

        b1, b5 = topk_counts(logits, target)
        c1 += b1
        c5 += b5

        y_true.extend(target.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    return {
        "loss": loss_sum / total,
        "top1": 100.0 * c1 / total,
        "top5": 100.0 * c5 / total,
        "macro_f1_supported": 100.0 * macro_f1(y_true, y_pred),
    }


def set_epoch_lr(optimizer, epoch, epochs, base_lr, min_lr, warmup):
    if epoch <= warmup:
        lr = base_lr * epoch / warmup
    else:
        progress = (epoch - warmup) / max(1, epochs - warmup)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


def make_smoke_subset(train_df, seed, n=384):
    one_per_class = (
        train_df.groupby("class_id", group_keys=False)
        .sample(n=1, random_state=seed)
    )

    remaining = n - len(one_per_class)

    if remaining > 0:
        pool = train_df.drop(index=one_per_class.index)
        extra = pool.sample(
            n=min(remaining, len(pool)),
            random_state=seed,
        )

        out = pd.concat([one_per_class, extra])
    else:
        out = one_per_class

    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--run-name",
        type=str,
        default="include262_conv1d_htma_v2_seed42",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    if not V1_SCRIPT.exists():
        raise FileNotFoundError(V1_SCRIPT)

    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(MANIFEST_CSV)

    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    v1 = load_v1_module()

    manifest = pd.read_csv(MANIFEST_CSV)
    ready = manifest[manifest["status"] == "ready"].copy()

    train_df = ready[ready["split"] == "train"].copy()
    val_df = ready[ready["split"] == "val"].copy()

    # Test stays locked. We intentionally never construct a test DataLoader.
    test_count = int((ready["split"] == "test").sum())

    if len(train_df) != 3121 or len(val_df) != 347 or test_count != 816:
        raise RuntimeError("Unexpected official split sizes.")

    if args.smoke_test:
        args.epochs = 2
        args.patience = 2
        train_df = make_smoke_subset(train_df, args.seed, n=384)
        val_df = val_df.sample(
            n=min(160, len(val_df)),
            random_state=args.seed,
        ).reset_index(drop=True)

        if args.run_name == "include262_conv1d_htma_v2_seed42":
            args.run_name = "include262_conv1d_htma_v2_smoke"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    base_train = v1.IncludeFeatureDataset(
        train_df,
        training=True,
        feature_noise_std=0.008,
        frame_dropout_prob=0.04,
    )
    base_val = v1.IncludeFeatureDataset(
        val_df,
        training=False,
    )

    train_ds = V2Dataset(base_train, training=True)
    val_ds = V2Dataset(base_val, training=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = IncludePartAwareConvHTMAV2().to(device)
    ema = EMA(model, decay=0.995)

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    weights = class_weights(train_df).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.08,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda"
        else None
    )

    best_path = MODELS / f"{args.run_name}_best.pt"
    last_path = MODELS / f"{args.run_name}_last.pt"
    history_path = RESULTS / f"{args.run_name}_history.csv"
    summary_path = RESULTS / f"{args.run_name}_summary.txt"
    config_path = RESULTS / f"{args.run_name}_config.json"

    start_epoch = 1
    best_top1 = -1.0
    best_f1 = -1.0
    bad_epochs = 0
    history = []

    if args.resume and last_path.exists():
        ckpt = torch.load(
            last_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(ckpt["model_state"])
        ema.model.load_state_dict(ckpt["ema_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

        start_epoch = int(ckpt["epoch"]) + 1
        best_top1 = float(ckpt["best_top1"])
        best_f1 = float(ckpt["best_f1"])
        bad_epochs = int(ckpt["bad_epochs"])

        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")

        print(f"Resumed from epoch {start_epoch - 1}")

    config = {
        "architecture": "Part-aware multi-scale Conv1D + gated fusion + HTMA + attention pooling",
        "num_classes": NUM_CLASSES,
        "frames": NUM_FRAMES,
        "d_model": 320,
        "heads": 8,
        "htma_layers": 4,
        "ema_decay": 0.995,
        "label_smoothing": 0.08,
        "optimizer": "AdamW",
        "base_lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "test_used": False,
    }

    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("=" * 96)
    print("INCLUDE STEP 20 - PART-AWARE Conv1D + HTMA V2")
    print("=" * 96)
    print(f"Device         : {device}")
    print(f"Train videos   : {len(train_df)}")
    print(f"Val videos     : {len(val_df)}")
    print(f"Test videos    : LOCKED / NOT USED ({test_count})")
    print(f"Classes        : {NUM_CLASSES}")
    print(f"Parameters     : {params / 1e6:.2f} M")
    print(f"Epochs         : {args.epochs}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Base LR        : {args.lr}")
    print(f"Warmup epochs  : {args.warmup}")
    print(f"EMA decay      : 0.995")
    print(f"Run name       : {args.run_name}")
    print()

    started = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        lr_now = set_epoch_lr(
            optimizer,
            epoch,
            args.epochs,
            args.lr,
            args.min_lr,
            args.warmup,
        )

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            ema=ema,
            scaler=scaler,
        )

        val_metrics = evaluate(
            ema.model,
            val_loader,
            criterion,
            device,
        )

        improved = (
            val_metrics["top1"] > best_top1
            or (
                math.isclose(
                    val_metrics["top1"],
                    best_top1,
                    abs_tol=1e-8,
                )
                and val_metrics["macro_f1_supported"] > best_f1
            )
        )

        if improved:
            best_top1 = val_metrics["top1"]
            best_f1 = val_metrics["macro_f1_supported"]
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "ema_state": ema.model.state_dict(),
                    "model_state": model.state_dict(),
                    "best_top1": best_top1,
                    "best_f1_supported": best_f1,
                    "config": config,
                },
                best_path,
            )
        else:
            bad_epochs += 1

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "train_f1_supported": train_metrics["macro_f1_supported"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "val_f1_supported": val_metrics["macro_f1_supported"],
            "best_val_top1": best_top1,
            "best_val_f1_supported": best_f1,
            "bad_epochs": bad_epochs,
            "epoch_minutes": (time.time() - epoch_start) / 60.0,
        }

        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": ema.model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_top1": best_top1,
                "best_f1": best_f1,
                "bad_epochs": bad_epochs,
                "config": config,
            },
            last_path,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"LR={lr_now:.6f} | "
            f"Train Loss={train_metrics['loss']:.4f} "
            f"Top1={train_metrics['top1']:.2f}% "
            f"Top5={train_metrics['top5']:.2f}% | "
            f"EMA Val Loss={val_metrics['loss']:.4f} "
            f"Top1={val_metrics['top1']:.2f}% "
            f"Top5={val_metrics['top5']:.2f}% "
            f"F1(supported)={val_metrics['macro_f1_supported']:.2f}% | "
            f"Best={best_top1:.2f}% | "
            f"{row['epoch_minutes']:.1f} min"
        )

        if bad_epochs >= args.patience:
            print(
                f"\nEarly stopping after {args.patience} "
                f"epochs without validation Top1 improvement."
            )
            break

    total_minutes = (time.time() - started) / 60.0

    summary = [
        "=" * 96,
        "INCLUDE262 PART-AWARE Conv1D + HTMA V2 SUMMARY",
        "=" * 96,
        "",
        f"Run name: {args.run_name}",
        f"Best validation Top1: {best_top1:.2f}%",
        f"Best validation Macro-F1 (supported classes only): {best_f1:.2f}%",
        f"Training time: {total_minutes:.1f} min",
        "",
        f"Best checkpoint: {best_path}",
        f"Last checkpoint: {last_path}",
        f"History: {history_path}",
        "",
        "TEST SPLIT:",
        "LOCKED - NOT EVALUATED",
        "",
        "NEXT:",
        "Compare V2 against 92.80% V1.",
        "If V2 reaches ~94%+, train additional seeds and ensemble validation logits.",
    ]

    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print("\n" + "\n".join(summary))
    print("=" * 96)


if __name__ == "__main__":
    main()