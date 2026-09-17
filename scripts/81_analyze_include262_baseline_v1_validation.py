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

TRAIN_SCRIPT = SCRIPTS / "80_train_include262_conv1d_htma_baseline_v1_fixed.py"
MANIFEST_CSV = META / "include262_model_features64_manifest_v1.csv"
CHECKPOINT = MODELS / "include262_conv1d_htma_baseline_v1_best.pt"
PREFIX = RESULTS / "include262_conv1d_htma_baseline_v1"
NUM_CLASSES = 262

def load_train_module():
    spec = importlib.util.spec_from_file_location("train_v1", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def div(a, b):
    return float(a / b) if b else 0.0

def main():
    print("=" * 92)
    print("INCLUDE STEP 19 - VALIDATION CONFUSION ANALYSIS")
    print("=" * 92)
    print("TEST SPLIT: LOCKED / NOT USED\n")

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(TRAIN_SCRIPT)
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(MANIFEST_CSV)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    RESULTS.mkdir(parents=True, exist_ok=True)

    train_mod = load_train_module()
    manifest = pd.read_csv(MANIFEST_CSV)
    ready = manifest[manifest["status"] == "ready"].copy()
    val_df = ready[ready["split"] == "val"].reset_index(drop=True)

    if len(val_df) != 347:
        raise RuntimeError(f"Expected 347 val rows, found {len(val_df)}")

    class_map = (
        ready[["class_id", "label"]]
        .drop_duplicates()
        .sort_values("class_id")
    )
    names = {int(r.class_id): str(r.label) for _, r in class_map.iterrows()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")
    print(f"Validation rows : {len(val_df)}")
    print(f"Classes         : {len(names)}\n")

    ds = train_mod.IncludeFeatureDataset(val_df, training=False)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = train_mod.IncludeTwoStreamConvHTMA(
        num_classes=NUM_CLASSES,
        branch_dim=128,
        d_model=256,
        num_heads=8,
        num_layers=3,
        dropout=0.15,
    ).to(device)

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    true_all, pred_all, top5_all, conf_all = [], [], [], []

    with torch.no_grad():
        for bi, (lm, geo, avail, y) in enumerate(dl, 1):
            lm = lm.to(device)
            geo = geo.to(device)
            avail = avail.to(device)

            logits = model(lm, geo, avail)
            probs = torch.softmax(logits, dim=1)
            confs, ids = probs.topk(5, dim=1)

            true_all.extend(y.numpy().tolist())
            pred_all.extend(ids[:, 0].cpu().numpy().tolist())
            top5_all.extend(ids.cpu().numpy().tolist())
            conf_all.extend(confs.cpu().numpy().tolist())

            if bi % 5 == 0 or bi == len(dl):
                print(f"Evaluated {min(bi*32, len(ds))}/{len(ds)}")

    y_true = np.array(true_all, dtype=int)
    y_pred = np.array(pred_all, dtype=int)
    top5 = np.array(top5_all, dtype=int)
    confs = np.array(conf_all, dtype=float)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_rows = []
    f1s, recalls = [], []

    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        support = cm[c].sum()
        pred_count = cm[:, c].sum()
        fp = pred_count - tp
        fn = support - tp

        precision = div(tp, tp + fp)
        recall = div(tp, tp + fn)
        f1 = div(2 * precision * recall, precision + recall)

        class_idx = np.where(y_true == c)[0]
        top5_recall = (
            np.mean([c in top5[i] for i in class_idx])
            if len(class_idx) else 0.0
        )

        row = cm[c].copy()
        row[c] = 0
        wrong_id = int(np.argmax(row)) if row.sum() else -1
        wrong_count = int(row[wrong_id]) if wrong_id >= 0 else 0

        per_rows.append({
            "class_id": c,
            "label": names[c],
            "support": int(support),
            "correct": int(tp),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "top5_recall": float(top5_recall),
            "most_confused_with_id": wrong_id,
            "most_confused_with_label": names.get(wrong_id, ""),
            "confusion_count": wrong_count,
        })
        f1s.append(f1)
        recalls.append(recall)

    per_class = pd.DataFrame(per_rows)

    confusion_rows = []
    for t in range(NUM_CLASSES):
        support = cm[t].sum()
        for p in range(NUM_CLASSES):
            if t != p and cm[t, p] > 0:
                confusion_rows.append({
                    "true_class_id": t,
                    "true_label": names[t],
                    "pred_class_id": p,
                    "pred_label": names[p],
                    "count": int(cm[t, p]),
                    "true_class_support": int(support),
                    "within_true_class_rate": div(cm[t, p], support),
                })

    top_confusions = pd.DataFrame(confusion_rows)
    if not top_confusions.empty:
        top_confusions = top_confusions.sort_values(
            ["count", "within_true_class_rate"],
            ascending=[False, False]
        ).reset_index(drop=True)

    pred_rows = []
    for i, row in val_df.iterrows():
        rec = {
            "source_index": int(row["source_index"]),
            "video_name": row["video_name"],
            "true_class_id": int(y_true[i]),
            "true_label": names[int(y_true[i])],
            "pred_class_id": int(y_pred[i]),
            "pred_label": names[int(y_pred[i])],
            "correct_top1": int(y_true[i] == y_pred[i]),
            "correct_top5": int(y_true[i] in top5[i]),
            "top1_confidence": float(confs[i, 0]),
        }
        for r in range(5):
            cid = int(top5[i, r])
            rec[f"top{r+1}_class_id"] = cid
            rec[f"top{r+1}_label"] = names[cid]
            rec[f"top{r+1}_confidence"] = float(confs[i, r])
        pred_rows.append(rec)

    preds = pd.DataFrame(pred_rows)

    top1 = 100 * np.mean(y_true == y_pred)
    top5_acc = 100 * np.mean([t in ids for t, ids in zip(y_true, top5)])
    macro_f1 = 100 * np.mean(f1s)
    bal_acc = 100 * np.mean(recalls)

    preds_path = Path(f"{PREFIX}_val_predictions.csv")
    per_path = Path(f"{PREFIX}_val_per_class_metrics.csv")
    cm_path = Path(f"{PREFIX}_val_confusion_matrix.csv")
    conf_path = Path(f"{PREFIX}_val_top_confusions.csv")
    summary_path = Path(f"{PREFIX}_val_analysis_summary.txt")

    preds.to_csv(preds_path, index=False)
    per_class.sort_values(["recall", "f1"]).to_csv(per_path, index=False)
    pd.DataFrame(
        cm,
        index=[f"{c}:{names[c]}" for c in range(NUM_CLASSES)],
        columns=[f"{c}:{names[c]}" for c in range(NUM_CLASSES)]
    ).to_csv(cm_path)
    top_confusions.head(100).to_csv(conf_path, index=False)

    worst = per_class.sort_values(["recall", "f1"]).head(20)

    wrong = preds[preds["correct_top1"] == 0]
    wrong_conf = 100 * wrong["top1_confidence"].mean() if len(wrong) else 0.0

    print("\n" + "=" * 92)
    print("VALIDATION ANALYSIS SUMMARY")
    print("=" * 92)
    print(f"Best checkpoint epoch : {int(ckpt.get('epoch', -1))}")
    print(f"Validation Top1       : {top1:.2f}%")
    print(f"Validation Top5       : {top5_acc:.2f}%")
    print(f"Validation Macro-F1   : {macro_f1:.2f}%")
    print(f"Balanced accuracy     : {bal_acc:.2f}%")
    print(f"Correct / Total       : {(y_true == y_pred).sum()} / {len(y_true)}")
    print(f"Wrong predictions     : {(y_true != y_pred).sum()}")
    print(f"Wrong-pred mean conf  : {wrong_conf:.2f}%")

    print("\n20 LOWEST-RECALL CLASSES")
    show = worst[
        ["class_id", "label", "support", "correct", "recall", "f1",
         "most_confused_with_label", "confusion_count"]
    ].copy()
    show["recall"] = (100 * show["recall"]).round(1)
    show["f1"] = (100 * show["f1"]).round(1)
    print(show.to_string(index=False))

    print("\nTOP 20 CONFUSION PAIRS")
    if not top_confusions.empty:
        show2 = top_confusions.head(20)[
            ["true_label", "pred_label", "count",
             "true_class_support", "within_true_class_rate"]
        ].copy()
        show2["within_true_class_rate"] = (
            100 * show2["within_true_class_rate"]
        ).round(1)
        print(show2.to_string(index=False))

    summary_lines = [
        "INCLUDE262 BASELINE V1 VALIDATION ANALYSIS",
        "",
        f"Checkpoint epoch: {int(ckpt.get('epoch', -1))}",
        f"Validation Top1: {top1:.2f}%",
        f"Validation Top5: {top5_acc:.2f}%",
        f"Validation Macro-F1: {macro_f1:.2f}%",
        f"Balanced accuracy: {bal_acc:.2f}%",
        f"Wrong predictions: {(y_true != y_pred).sum()} / {len(y_true)}",
        f"Wrong prediction mean confidence: {wrong_conf:.2f}%",
        "",
        "20 lowest-recall classes:"
    ]
    for _, r in worst.iterrows():
        summary_lines.append(
            f"{int(r.class_id)} | {r.label} | support={int(r.support)} | "
            f"recall={100*r.recall:.1f}% | F1={100*r.f1:.1f}% | "
            f"confused_with={r.most_confused_with_label} ({int(r.confusion_count)})"
        )
    summary_lines += [
        "",
        "TEST SPLIT: LOCKED / NOT USED",
        f"Predictions: {preds_path}",
        f"Per-class metrics: {per_path}",
        f"Confusion matrix: {cm_path}",
        f"Top confusions: {conf_path}",
        "",
        "NEXT: use validation errors to design V2 improvements."
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nSaved analysis files.")
    print("TEST SPLIT REMAINS LOCKED ✅")
    print("=" * 92)

if __name__ == "__main__":
    main()