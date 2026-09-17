from pathlib import Path
import importlib.util
import json
import time

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

PREDICTIONS_CSV = RESULTS / "include262_v2_seed42_FINAL_TEST_predictions.csv"
PER_CLASS_CSV = RESULTS / "include262_v2_seed42_FINAL_TEST_per_class_metrics.csv"
CONFUSION_CSV = RESULTS / "include262_v2_seed42_FINAL_TEST_confusion_matrix.csv"
TOP_CONFUSIONS_CSV = RESULTS / "include262_v2_seed42_FINAL_TEST_top_confusions.csv"
SUMMARY_TXT = RESULTS / "include262_v2_seed42_FINAL_TEST_summary.txt"
FINAL_JSON = RESULTS / "include262_v2_seed42_FINAL_TEST_metrics.json"

NUM_CLASSES = 262


def load_module(path, name):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_div(a, b):
    return float(a / b) if b else 0.0


def compute_metrics(y_true, y_pred, top5_ids, class_names):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class_rows = []
    supported_f1 = []
    supported_recalls = []

    for c in range(NUM_CLASSES):
        tp = int(cm[c, c])
        support = int(cm[c].sum())
        pred_count = int(cm[:, c].sum())

        fp = pred_count - tp
        fn = support - tp

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)

        if precision + recall:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        idxs = np.where(y_true == c)[0]
        if len(idxs):
            top5_recall = float(
                np.mean([c in top5_ids[i] for i in idxs])
            )
            supported_f1.append(f1)
            supported_recalls.append(recall)
        else:
            top5_recall = np.nan

        row = cm[c].copy()
        row[c] = 0

        wrong_id = int(np.argmax(row)) if row.sum() else -1
        wrong_count = int(row[wrong_id]) if wrong_id >= 0 else 0

        per_class_rows.append({
            "class_id": c,
            "label": class_names[c],
            "support": support,
            "correct": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "top5_recall": top5_recall,
            "most_confused_with_id": wrong_id,
            "most_confused_with_label": class_names.get(wrong_id, ""),
            "confusion_count": wrong_count,
        })

    top1_count = int((y_true == y_pred).sum())
    top5_count = int(
        np.sum([t in ids for t, ids in zip(y_true, top5_ids)])
    )

    top1 = 100.0 * top1_count / len(y_true)
    top5 = 100.0 * top5_count / len(y_true)

    macro_f1_supported = (
        100.0 * float(np.mean(supported_f1))
        if supported_f1 else 0.0
    )

    balanced_accuracy_supported = (
        100.0 * float(np.mean(supported_recalls))
        if supported_recalls else 0.0
    )

    per_class = pd.DataFrame(per_class_rows)

    confusion_rows = []

    for true_id in range(NUM_CLASSES):
        support = int(cm[true_id].sum())

        for pred_id in range(NUM_CLASSES):
            if true_id == pred_id:
                continue

            count = int(cm[true_id, pred_id])

            if count > 0:
                confusion_rows.append({
                    "true_class_id": true_id,
                    "true_label": class_names[true_id],
                    "pred_class_id": pred_id,
                    "pred_label": class_names[pred_id],
                    "count": count,
                    "true_class_support": support,
                    "within_true_class_rate": safe_div(count, support),
                })

    top_confusions = pd.DataFrame(confusion_rows)

    if not top_confusions.empty:
        top_confusions = top_confusions.sort_values(
            ["count", "within_true_class_rate"],
            ascending=[False, False],
        ).reset_index(drop=True)

    return {
        "cm": cm,
        "per_class": per_class,
        "top_confusions": top_confusions,
        "top1_count": top1_count,
        "top5_count": top5_count,
        "top1": top1,
        "top5": top5,
        "macro_f1_supported": macro_f1_supported,
        "balanced_accuracy_supported": balanced_accuracy_supported,
    }


def main():
    print("=" * 100)
    print("INCLUDE STEP 22 - FINAL ONE-TIME EVALUATION ON UNTOUCHED TEST SPLIT")
    print("=" * 100)
    print("MODEL IS FROZEN. NO TRAINING. NO HYPERPARAMETER TUNING.")
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

    test_df = ready[
        ready["split"] == "test"
    ].copy().reset_index(drop=True)

    if len(test_df) != 816:
        raise RuntimeError(
            f"Expected 816 untouched test videos, found {len(test_df)}."
        )

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
        raise RuntimeError(
            f"Expected {NUM_CLASSES} classes, found {len(class_names)}."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device          : {device}")
    print(f"Test videos     : {len(test_df)}")
    print(f"Classes         : {NUM_CLASSES}")
    print(f"Checkpoint      : {CHECKPOINT.name}")
    print()

    base_test = v1.IncludeFeatureDataset(
        test_df,
        training=False,
    )

    test_ds = v2.V2Dataset(
        base_test,
        training=False,
    )

    test_loader = DataLoader(
        test_ds,
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
        raise RuntimeError(
            "Best V2 checkpoint does not contain ema_state."
        )

    model.load_state_dict(ckpt["ema_state"])
    model.eval()

    checkpoint_epoch = int(
        ckpt.get("epoch", -1)
    )

    print(f"Frozen checkpoint epoch: {checkpoint_epoch}")
    print()

    y_true = []
    y_pred = []
    top5_all = []
    prob5_all = []

    started = time.time()

    with torch.no_grad():
        for batch_idx, (
            landmark,
            geometry,
            availability,
            target,
        ) in enumerate(test_loader, start=1):

            landmark = landmark.to(device)
            geometry = geometry.to(device)
            availability = availability.to(device)

            logits = model(
                landmark,
                geometry,
                availability,
            )

            probs = torch.softmax(
                logits,
                dim=1,
            )

            prob5, ids5 = probs.topk(
                5,
                dim=1,
                largest=True,
                sorted=True,
            )

            y_true.extend(
                target.numpy().tolist()
            )

            y_pred.extend(
                ids5[:, 0]
                .cpu()
                .numpy()
                .tolist()
            )

            top5_all.extend(
                ids5.cpu()
                .numpy()
                .tolist()
            )

            prob5_all.extend(
                prob5.cpu()
                .numpy()
                .tolist()
            )

            if (
                batch_idx % 5 == 0
                or batch_idx == len(test_loader)
            ):
                done = min(
                    batch_idx * test_loader.batch_size,
                    len(test_ds),
                )

                print(
                    f"Evaluated {done}/{len(test_ds)} test samples"
                )

    elapsed = time.time() - started

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    top5_ids = np.asarray(
        top5_all,
        dtype=np.int64,
    )

    top5_probs = np.asarray(
        prob5_all,
        dtype=np.float32,
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
        top5_ids,
        class_names,
    )

    prediction_rows = []

    for i, row in test_df.iterrows():
        rec = {
            "source_index": int(row["source_index"]),
            "video_name": row["video_name"],
            "true_class_id": int(y_true[i]),
            "true_label": class_names[int(y_true[i])],
            "pred_class_id": int(y_pred[i]),
            "pred_label": class_names[int(y_pred[i])],
            "correct_top1": int(
                y_true[i] == y_pred[i]
            ),
            "correct_top5": int(
                y_true[i] in top5_ids[i]
            ),
        }

        for rank in range(5):
            cid = int(
                top5_ids[i, rank]
            )

            rec[f"top{rank+1}_class_id"] = cid
            rec[f"top{rank+1}_label"] = class_names[cid]
            rec[f"top{rank+1}_probability"] = float(
                top5_probs[i, rank]
            )

        prediction_rows.append(rec)

    predictions = pd.DataFrame(
        prediction_rows
    )

    predictions.to_csv(
        PREDICTIONS_CSV,
        index=False,
    )

    metrics["per_class"].sort_values(
        ["support", "recall", "f1"],
        ascending=[False, True, True],
    ).to_csv(
        PER_CLASS_CSV,
        index=False,
    )

    pd.DataFrame(
        metrics["cm"],
        index=[
            f"{c}:{class_names[c]}"
            for c in range(NUM_CLASSES)
        ],
        columns=[
            f"{c}:{class_names[c]}"
            for c in range(NUM_CLASSES)
        ],
    ).to_csv(
        CONFUSION_CSV
    )

    metrics["top_confusions"].head(
        100
    ).to_csv(
        TOP_CONFUSIONS_CSV,
        index=False,
    )

    supported_classes = int(
        (
            metrics["per_class"]["support"] > 0
        ).sum()
    )

    wrong_count = (
        len(y_true)
        - metrics["top1_count"]
    )

    print("\n" + "=" * 100)
    print("FINAL INCLUDE262 TEST RESULT")
    print("=" * 100)
    print(
        f"Frozen checkpoint epoch       : {checkpoint_epoch}"
    )
    print(
        f"Correct Top1                  : "
        f"{metrics['top1_count']} / 816"
    )
    print(
        f"Wrong Top1                    : "
        f"{wrong_count}"
    )
    print(
        f"FINAL TEST TOP1 ACCURACY      : "
        f"{metrics['top1']:.2f}%"
    )
    print(
        f"FINAL TEST TOP5 ACCURACY      : "
        f"{metrics['top5']:.2f}%"
    )
    print(
        f"FINAL TEST MACRO-F1           : "
        f"{metrics['macro_f1_supported']:.2f}% "
        f"(supported classes only)"
    )
    print(
        f"Balanced accuracy             : "
        f"{metrics['balanced_accuracy_supported']:.2f}% "
        f"(supported classes only)"
    )
    print(
        f"Test classes represented      : "
        f"{supported_classes} / {NUM_CLASSES}"
    )
    print(
        f"Total evaluation time         : "
        f"{elapsed:.2f} sec"
    )

    top_confusions = metrics[
        "top_confusions"
    ]

    if not top_confusions.empty:
        print("\nTOP 20 TEST CONFUSIONS")
        show = top_confusions.head(20)[
            [
                "true_label",
                "pred_label",
                "count",
                "true_class_support",
                "within_true_class_rate",
            ]
        ].copy()

        show[
            "within_true_class_rate"
        ] = (
            100.0
            * show[
                "within_true_class_rate"
            ]
        ).round(1)

        print(
            show.to_string(
                index=False
            )
        )

    final_metrics = {
        "dataset": "INCLUDE",
        "available_classes": NUM_CLASSES,
        "test_videos": 816,
        "frozen_checkpoint_epoch": checkpoint_epoch,
        "correct_top1": metrics["top1_count"],
        "wrong_top1": wrong_count,
        "test_top1_accuracy_percent": metrics["top1"],
        "test_top5_accuracy_percent": metrics["top5"],
        "test_macro_f1_supported_percent": metrics["macro_f1_supported"],
        "test_balanced_accuracy_supported_percent": metrics["balanced_accuracy_supported"],
        "test_supported_classes": supported_classes,
        "evaluation_seconds": elapsed,
        "checkpoint": str(CHECKPOINT),
        "model": "Part-aware multi-scale Conv1D + gated fusion + HTMA + attention pooling + EMA",
        "test_tuning_after_evaluation_allowed": False,
    }

    FINAL_JSON.write_text(
        json.dumps(
            final_metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = [
        "=" * 100,
        "FINAL INCLUDE262 FROZEN-MODEL TEST RESULT",
        "=" * 100,
        "",
        f"Frozen checkpoint epoch: {checkpoint_epoch}",
        f"Test videos: 816",
        f"Classes: {NUM_CLASSES}",
        f"Correct Top1: {metrics['top1_count']} / 816",
        f"Wrong Top1: {wrong_count}",
        f"FINAL TEST TOP1 ACCURACY: {metrics['top1']:.6f}%",
        f"FINAL TEST TOP5 ACCURACY: {metrics['top5']:.6f}%",
        (
            "FINAL TEST MACRO-F1 "
            f"(supported classes only): "
            f"{metrics['macro_f1_supported']:.6f}%"
        ),
        (
            "Balanced accuracy "
            f"(supported classes only): "
            f"{metrics['balanced_accuracy_supported']:.6f}%"
        ),
        f"Supported test classes: {supported_classes} / {NUM_CLASSES}",
        f"Evaluation time: {elapsed:.2f} sec",
        "",
        "MODEL:",
        (
            "Part-aware multi-scale Conv1D + gated fusion "
            "+ HTMA + attention pooling + EMA"
        ),
        "",
        "CHECKPOINT:",
        str(CHECKPOINT),
        "",
        "IMPORTANT:",
        "This is the final one-time frozen-model test result.",
        "Do not tune INCLUDE architecture/hyperparameters using this test result.",
        "",
        f"Predictions: {PREDICTIONS_CSV}",
        f"Per-class metrics: {PER_CLASS_CSV}",
        f"Confusion matrix: {CONFUSION_CSV}",
        f"Top confusions: {TOP_CONFUSIONS_CSV}",
        f"Metrics JSON: {FINAL_JSON}",
        "",
        "STATUS:",
        "INCLUDE MODEL EVALUATION COMPLETE ✅",
        "NEXT PROJECT PHASE: rebuild WLASL from scratch.",
    ]

    SUMMARY_TXT.write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("\n" + "\n".join(summary))
    print("=" * 100)


if __name__ == "__main__":
    main()