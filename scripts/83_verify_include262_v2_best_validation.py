from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path.cwd()
SCRIPTS = ROOT / "scripts"
META = ROOT / "metadata"
MODELS = ROOT / "models" / "checkpoints"
RESULTS = ROOT / "results"

V1_SCRIPT = SCRIPTS / "80_train_include262_conv1d_htma_baseline_v1_fixed.py"
V2_SCRIPT = SCRIPTS / "82_train_include262_partaware_htma_v2.py"
MANIFEST_CSV = META / "include262_model_features64_manifest_v1.csv"
CHECKPOINT = MODELS / "include262_conv1d_htma_v2_seed42_best.pt"

OUT_PRED = RESULTS / "include262_v2_seed42_validation_verified_predictions.csv"
OUT_SUMMARY = RESULTS / "include262_v2_seed42_validation_verification_summary.txt"

NUM_CLASSES = 262


def load_module(path, name):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def supported_macro_f1(y_true, y_pred, num_classes=NUM_CLASSES):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    f1s = []

    for c in range(num_classes):
        true_mask = y_true == c
        if true_mask.sum() == 0:
            continue

        pred_mask = y_pred == c
        tp = np.sum(true_mask & pred_mask)
        fp = np.sum((~true_mask) & pred_mask)
        fn = np.sum(true_mask & (~pred_mask))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        if precision + recall:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1s.append(f1)

    return float(np.mean(f1s)) if f1s else 0.0


def main():
    print("=" * 96)
    print("INCLUDE STEP 21 - INDEPENDENT V2 BEST-CHECKPOINT VALIDATION VERIFICATION")
    print("=" * 96)
    print("TEST SPLIT: LOCKED / NOT LOADED / NOT EVALUATED")
    print()

    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(MANIFEST_CSV)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    RESULTS.mkdir(parents=True, exist_ok=True)

    v1 = load_module(V1_SCRIPT, "include_v1")
    v2 = load_module(V2_SCRIPT, "include_v2")

    manifest = pd.read_csv(MANIFEST_CSV)
    ready = manifest[manifest["status"] == "ready"].copy()

    val_df = ready[ready["split"] == "val"].copy().reset_index(drop=True)
    test_count = int((ready["split"] == "test").sum())

    if len(val_df) != 347:
        raise RuntimeError(f"Expected 347 validation videos, found {len(val_df)}.")
    if test_count != 816:
        raise RuntimeError(f"Expected 816 locked test videos, found {test_count}.")

    class_map_df = (
        ready[["class_id", "label"]]
        .drop_duplicates()
        .sort_values("class_id")
    )
    class_names = {
        int(r["class_id"]): str(r["label"])
        for _, r in class_map_df.iterrows()
    }

    if len(class_names) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} classes, found {len(class_names)}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device            : {device}")
    print(f"Validation videos : {len(val_df)}")
    print(f"Classes           : {NUM_CLASSES}")
    print(f"Locked test videos: {test_count}")
    print()

    base_val = v1.IncludeFeatureDataset(val_df, training=False)
    val_ds = v2.V2Dataset(base_val, training=False)

    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = v2.IncludePartAwareConvHTMAV2().to(device)

    ckpt = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if "ema_state" not in ckpt:
        raise RuntimeError("Best checkpoint does not contain ema_state.")

    model.load_state_dict(ckpt["ema_state"])
    model.eval()

    checkpoint_epoch = int(ckpt.get("epoch", -1))
    stored_best_top1 = float(ckpt.get("best_top1", np.nan))
    stored_best_f1 = float(ckpt.get("best_f1_supported", np.nan))

    print(f"Best checkpoint epoch : {checkpoint_epoch}")
    print(f"Stored best Top1      : {stored_best_top1:.2f}%")
    print()

    y_true = []
    y_pred = []
    top5_all = []
    prob5_all = []

    with torch.no_grad():
        for batch_idx, (landmark, geometry, availability, target) in enumerate(val_loader, start=1):
            landmark = landmark.to(device)
            geometry = geometry.to(device)
            availability = availability.to(device)

            logits = model(landmark, geometry, availability)
            probs = torch.softmax(logits, dim=1)
            prob5, ids5 = probs.topk(5, dim=1, largest=True, sorted=True)

            y_true.extend(target.numpy().tolist())
            y_pred.extend(ids5[:, 0].cpu().numpy().tolist())
            top5_all.extend(ids5.cpu().numpy().tolist())
            prob5_all.extend(prob5.cpu().numpy().tolist())

            if batch_idx % 5 == 0 or batch_idx == len(val_loader):
                done = min(batch_idx * val_loader.batch_size, len(val_ds))
                print(f"Verified {done}/{len(val_ds)} validation samples")

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    top5 = np.asarray(top5_all, dtype=np.int64)
    prob5 = np.asarray(prob5_all, dtype=np.float32)

    correct_top1 = y_true == y_pred
    correct_top5 = np.asarray(
        [int(t in ids) for t, ids in zip(y_true, top5)],
        dtype=bool,
    )

    top1_count = int(correct_top1.sum())
    top5_count = int(correct_top5.sum())

    top1 = 100.0 * top1_count / len(y_true)
    top5_acc = 100.0 * top5_count / len(y_true)
    f1 = 100.0 * supported_macro_f1(y_true, y_pred)

    prediction_rows = []
    for i, row in val_df.iterrows():
        rec = {
            "source_index": int(row["source_index"]),
            "video_name": row["video_name"],
            "true_class_id": int(y_true[i]),
            "true_label": class_names[int(y_true[i])],
            "pred_class_id": int(y_pred[i]),
            "pred_label": class_names[int(y_pred[i])],
            "correct_top1": int(correct_top1[i]),
            "correct_top5": int(correct_top5[i]),
        }

        for rank in range(5):
            cid = int(top5[i, rank])
            rec[f"top{rank+1}_class_id"] = cid
            rec[f"top{rank+1}_label"] = class_names[cid]
            rec[f"top{rank+1}_probability"] = float(prob5[i, rank])

        prediction_rows.append(rec)

    pd.DataFrame(prediction_rows).to_csv(OUT_PRED, index=False)

    top1_matches = abs(top1 - stored_best_top1) < 1e-6
    f1_matches = (
        np.isnan(stored_best_f1)
        or abs(f1 - stored_best_f1) < 1e-6
    )

    verified = (
        len(y_true) == 347
        and top1_matches
        and f1_matches
    )

    print("\n" + "=" * 96)
    print("INDEPENDENT VALIDATION VERIFICATION RESULT")
    print("=" * 96)
    print(f"Checkpoint epoch          : {checkpoint_epoch}")
    print(f"Correct Top1              : {top1_count} / 347")
    print(f"Wrong Top1                : {347 - top1_count}")
    print(f"Verified Validation Top1  : {top1:.2f}%")
    print(f"Verified Validation Top5  : {top5_acc:.2f}%")
    print(f"Verified F1 (supported)   : {f1:.2f}%")
    print(f"Stored Validation Top1    : {stored_best_top1:.2f}%")
    if not np.isnan(stored_best_f1):
        print(f"Stored F1 (supported)     : {stored_best_f1:.2f}%")
    print(f"Top1 exact match          : {top1_matches}")
    print(f"F1 exact match            : {f1_matches}")
    print()
    print("TEST SPLIT STATUS:")
    print("LOCKED - NOT LOADED - NOT EVALUATED")

    summary = [
        "=" * 96,
        "INCLUDE262 V2 BEST-CHECKPOINT VALIDATION VERIFICATION",
        "=" * 96,
        "",
        f"Checkpoint epoch: {checkpoint_epoch}",
        f"Correct Top1: {top1_count} / 347",
        f"Wrong Top1: {347 - top1_count}",
        f"Verified Validation Top1: {top1:.6f}%",
        f"Verified Validation Top5: {top5_acc:.6f}%",
        f"Verified Macro-F1 (supported classes only): {f1:.6f}%",
        f"Stored Validation Top1: {stored_best_top1:.6f}%",
        (
            f"Stored Macro-F1 (supported classes only): {stored_best_f1:.6f}%"
            if not np.isnan(stored_best_f1)
            else "Stored Macro-F1: unavailable"
        ),
        f"Top1 exact match: {top1_matches}",
        f"F1 exact match: {f1_matches}",
        "",
        "TEST SPLIT:",
        "LOCKED - NOT LOADED - NOT EVALUATED",
        "",
        f"Predictions: {OUT_PRED}",
        "",
        "STATUS:",
    ]

    if verified:
        summary.extend(
            [
                "BEST V2 VALIDATION CHECKPOINT VERIFIED ✅",
                "ARCHITECTURE CAN NOW BE FROZEN.",
                "NEXT: one-time evaluation on the untouched 816-video test split.",
            ]
        )
    else:
        summary.extend(
            [
                "VALIDATION VERIFICATION MISMATCH ⚠️",
                "DO NOT TOUCH THE TEST SPLIT.",
                "Investigate checkpoint/evaluation mismatch first.",
            ]
        )

    OUT_SUMMARY.write_text("\n".join(summary), encoding="utf-8")

    print("\n" + "\n".join(summary))
    print("=" * 96)

    if not verified:
        raise RuntimeError(
            "Independent validation verification did not exactly match the saved best checkpoint."
        )


if __name__ == "__main__":
    main()