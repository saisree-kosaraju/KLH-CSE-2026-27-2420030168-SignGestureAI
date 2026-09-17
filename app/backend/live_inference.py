"""
Live / webcam single-sign recognition pipeline.

    webcam_extract.extract_trimmed()   raw clip -> active-sign-trimmed 64-frame arrays
              |                        (+ luma/blur/motion stats)
    quality.assess()                   BLOCK obviously unusable input before the model runs
              |
    preprocessing.build_features_from_arrays()   the SAME script-79 features as training
              |
    model(landmark, geometry, availability)      frozen INCLUDE262 checkpoint (ema_state)
              |
    rejection.evaluate()               top-1 confidence + top1/top2 margin gate

The frozen `/predict` (upload) path is untouched; this is a parallel path used only by
the Live Camera tab via `POST /predict/live`.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

import config
import inference
import preprocessing
import quality
import rejection
import trim as trim_mod
import webcam_extract


def predict_live(
    video_path,
    topk: int = 5,
    trim_cfg: trim_mod.TrimConfig | None = None,
    quality_cfg: quality.QualityConfig | None = None,
    rejection_cfg: rejection.RejectionConfig | None = None,
    disable_trim: bool = False,
) -> dict:
    video_path = str(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    device = torch.device(config.DEVICE)
    class_map = inference._load_class_map()

    # -- 1. trimmed extraction -------------------------------------------
    t0 = time.perf_counter()
    try:
        arrays, xstats = webcam_extract.extract_trimmed(
            video_path, trim_cfg=trim_cfg, disable_trim=disable_trim
        )
    except RuntimeError as exc:
        return {
            "accepted": False,
            "rejection_reason": f"extraction:{exc}",
            "message": "Could not read the recording. Try recording again.",
            "stage": "extraction",
            "trim": None,
            "quality": None,
            "top5": [],
        }
    extract_seconds = time.perf_counter() - t0

    trim_report = {
        "original_duration_s": xstats.get("original_duration_s"),
        "trimmed_duration_s": xstats.get("trimmed_duration_s"),
        "trimmed_frame_range": xstats.get("trimmed_frame_range"),
        "total_frames": xstats.get("total_frames"),
        "motion_score": xstats.get("motion_score"),
        "trimmed": xstats.get("trimmed"),
        "trim_fallback": xstats.get("trim_fallback"),
        "scan_frames": xstats.get("scan_frames"),
        "fps": xstats.get("fps"),
    }

    # -- 2. quality gate ------------------------------------------------
    q = quality.assess(arrays, xstats, cfg=quality_cfg or quality.DEFAULT_QUALITY)
    if q["overall"] == "block":
        return {
            "accepted": False,
            "rejection_reason": "quality_block",
            "message": q["messages"][0] if q["messages"] else "Recording unusable — please try again.",
            "stage": "quality",
            "trim": trim_report,
            "quality": q,
            "top5": [],
            "timings_ms": {"extract": round(extract_seconds * 1000, 1)},
        }

    # -- 3. features (identical to training) ---------------------------
    feats = preprocessing.build_features_from_arrays(arrays)

    landmark = torch.from_numpy(feats["landmark"]).unsqueeze(0).to(device)
    geometry = torch.from_numpy(feats["geometry"]).unsqueeze(0).to(device)
    availability = torch.from_numpy(feats["availability"]).unsqueeze(0).to(device)

    # -- 4. model forward --------------------------------------------
    variant = config.active_live_variant()
    model, checkpoint = inference._load_model(variant)
    t1 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        logits = model(landmark, geometry, availability)
        probs = torch.softmax(logits, dim=1)[0]
    infer_seconds = time.perf_counter() - t1

    k = min(topk, config.NUM_CLASSES)
    top_probs, top_ids = probs.topk(k, largest=True, sorted=True)
    top_ids = [int(i) for i in top_ids.cpu().numpy()]
    top_probs = [float(p) for p in top_probs.cpu().numpy()]
    top5 = [
        {"rank": r + 1, "class_id": cid, "sign": class_map[cid], "probability": round(p, 4)}
        for r, (cid, p) in enumerate(zip(top_ids, top_probs))
    ]

    # -- 5. confidence / margin rejection --------------------------
    verdict = rejection.evaluate(top5, cfg=rejection_cfg or rejection.DEFAULT_REJECTION)

    modality_coverage = {
        "per_frame_availability": {
            "left_hand": float(feats["availability"][:, 0].mean()),
            "right_hand": float(feats["availability"][:, 1].mean()),
            "pose": float(feats["availability"][:, 2].mean()),
            "face": float(feats["availability"][:, 3].mean()),
        },
        "extractor_rates": {
            k2: float(xstats.get(k2, 0.0))
            for k2 in (
                "left_final_rate", "right_final_rate", "both_hands_final_rate",
                "left_rescue_rate", "right_rescue_rate", "pose_rate", "face_rate",
            )
        },
    }

    return {
        "accepted": verdict["accepted"],
        "rejection_reason": verdict["rejection_reason"],
        "message": verdict["message"] if not verdict["accepted"] else None,
        "stage": "complete",
        "predicted_class_id": top_ids[0],
        "predicted_sign": class_map[top_ids[0]],
        "confidence": round(top_probs[0], 4),
        "top1_sign": verdict["top1_sign"],
        "top1_confidence": verdict["top1_confidence"],
        "top2_sign": verdict["top2_sign"],
        "top2_confidence": verdict["top2_confidence"],
        "top2_margin": verdict["top2_margin"],
        "thresholds": verdict.get("thresholds"),
        "top5": top5,
        "modality_coverage": modality_coverage,
        "quality": q,
        "trim": trim_report,
        "extraction_status": xstats.get("status", "ok"),
        "tensor_shapes": {
            "landmark": list(feats["landmark"].shape),
            "geometry": list(feats["geometry"].shape),
            "availability": list(feats["availability"].shape),
        },
        "timings_ms": {
            "extract": round(extract_seconds * 1000, 1),
            "inference": round(infer_seconds * 1000, 1),
        },
        "checkpoint": {
            "path": checkpoint.get("_path", str(config.CHECKPOINT)),
            "epoch": int(checkpoint.get("epoch", -1)),
            "weights": "ema_state",
            "variant": variant,
        },
    }
