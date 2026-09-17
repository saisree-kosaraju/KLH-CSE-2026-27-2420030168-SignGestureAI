"""
End-to-end INCLUDE262 inference: raw video path -> prediction dict.

    preprocessing.preprocess_video()  ->  (64,534) (64,469) (64,4)
    model_loader.load_model()         ->  IncludePartAwareConvHTMAV2 + ema_state
    forward under model.eval() + torch.no_grad()  ->  softmax  ->  top-5
"""

import csv
import functools
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config          # noqa: E402
import model_loader    # noqa: E402
import preprocessing   # noqa: E402


@functools.lru_cache(maxsize=1)
def _load_class_map():
    mapping = {}
    with open(config.CLASS_MAP_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mapping[int(row["class_id"])] = row["label"]
    if len(mapping) != config.NUM_CLASSES:
        raise RuntimeError(
            f"Expected {config.NUM_CLASSES} classes in {config.CLASS_MAP_CSV}, "
            f"found {len(mapping)}."
        )
    return mapping


@functools.lru_cache(maxsize=3)
def _load_model(variant="original"):
    """Cached per variant. 'original' = the frozen 98.04%-test checkpoint (always used
    by /predict). 'webcam_adapted' = webcam/adapt_train.py output, falls back to
    original if absent — used by /predict/live when INCLUDE262_MODEL_VARIANT is set."""
    return model_loader.load_model(config.DEVICE, variant=variant)


def predict(video_path, topk=5) -> dict:
    device = torch.device(config.DEVICE)
    model, checkpoint = _load_model()
    class_map = _load_class_map()

    pre = preprocessing.preprocess_video(video_path)

    landmark = torch.from_numpy(pre["landmark"]).unsqueeze(0).to(device)
    geometry = torch.from_numpy(pre["geometry"]).unsqueeze(0).to(device)
    availability = torch.from_numpy(pre["availability"]).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        logits = model(landmark, geometry, availability)
        probs = torch.softmax(logits, dim=1)[0]
    inference_seconds = time.perf_counter() - t0

    k = min(topk, config.NUM_CLASSES)
    top_probs, top_ids = probs.topk(k, largest=True, sorted=True)
    top_ids = [int(i) for i in top_ids.cpu().numpy()]
    top_probs = [float(p) for p in top_probs.cpu().numpy()]

    top5 = [
        {"rank": r + 1, "class_id": cid, "sign": class_map[cid], "probability": p}
        for r, (cid, p) in enumerate(zip(top_ids, top_probs))
    ]

    pred_id = top_ids[0]

    return {
        "video_path": os.path.abspath(str(video_path)),
        "predicted_class_id": pred_id,
        "predicted_sign": class_map[pred_id],
        "confidence": top_probs[0],
        "top5": top5,
        "modality_coverage": pre["modality_coverage"],
        "extraction_status": pre["extraction_status"],
        "preprocess_seconds": pre["preprocess_seconds"],
        "inference_seconds": inference_seconds,
        "checkpoint": {
            "path": str(config.CHECKPOINT),
            "epoch": int(checkpoint.get("epoch", -1)),
            "weights": "ema_state",
        },
        "tensor_shapes": {
            "landmark": list(pre["landmark"].shape),
            "geometry": list(pre["geometry"].shape),
            "availability": list(pre["availability"].shape),
        },
    }
