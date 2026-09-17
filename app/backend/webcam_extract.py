"""
Trimmed 64-frame landmark extraction for the LIVE webcam path.

Produces exactly the same `arrays` dict that scripts/76 `process_video` returns,
so `scripts/79.build_one_feature_file` (via `preprocessing.build_features_from_arrays`)
consumes it unchanged — the 534-D / 469-D / 4-D semantics stay byte-identical to
training. The ONLY difference from the frozen path is WHICH 64 original frames are
sampled: instead of linearly across the whole clip, they are sampled linearly
inside the detected active-sign span (see `trim.py`).

Two passes:
  1. SCAN  — lightweight `mp.solutions.pose.Pose` (complexity 0, no face) over up to
             `cfg.scan_frames` linearly-sampled frames -> wrist trajectories -> motion
             envelope -> active span. Also cheap luma / blur / backlight stats.
  2. FINAL — the frozen Holistic + pose-guided hand-rescue loop (identical to
             scripts/76 `process_video`, lines ~232-327) over 64 frames sampled inside
             the active span.

MediaPipe config is imported from `preprocessing.py` (HOLISTIC_KWARGS / HANDS_KWARGS,
copied verbatim from scripts/76 `main()`), never re-declared here.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

import preprocessing as _pp  # HOLISTIC_KWARGS, HANDS_KWARGS, _s76, extract helpers
import trim as _trim

_s76 = _pp._s76
mp = _pp._s76.mp  # mediapipe module, already imported by scripts/76

POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_VIS_MIN = 0.35
NUM_FRAMES = 64


# ---------------------------------------------------------------------------
# frame decoding
# ---------------------------------------------------------------------------
def _open(video_path: str) -> tuple[cv2.VideoCapture, int, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("cannot_open")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0:
        cap.release()
        raise RuntimeError("invalid_frame_count")
    return cap, total, fps, w, h


def _grab_by_index(cap: cv2.VideoCapture, targets: np.ndarray) -> list[np.ndarray] | None:
    """Sequential read; keep frames whose running index hits a (sorted) target. Pads with last good."""
    targets = np.sort(targets)
    frames: list[np.ndarray] = []
    pos = 0
    idx = 0
    last_good = None
    n = len(targets)
    while pos < n:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            break
        last_good = frame
        while pos < n and idx >= targets[pos]:
            frames.append(frame.copy())
            pos += 1
        idx += 1
    if last_good is None:
        return None
    while len(frames) < n:
        frames.append(last_good.copy())
    return frames[:n]


# ---------------------------------------------------------------------------
# pass 1 — motion scan
# ---------------------------------------------------------------------------
def _luma_stats(frame_bgr: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mean_luma = float(gray.mean()) / 255.0
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    y0, y1 = int(h * 0.25), int(h * 0.75)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    center = gray[y0:y1, x0:x1]
    center_mean = float(center.mean()) if center.size else mean_luma * 255.0
    border_sum = float(gray.sum()) - float(center.sum())
    border_n = gray.size - center.size
    border_mean = border_sum / border_n if border_n > 0 else center_mean
    backlight_ratio = border_mean / max(center_mean, 1.0)  # >1 => brighter surround than subject
    return mean_luma, lap_var, backlight_ratio


def _scan(video_path: str, total: int, cfg: _trim.TrimConfig):
    n = int(min(cfg.scan_frames, total))
    targets = np.linspace(0, total - 1, n).round().astype(np.int64)

    cap, *_ = _open(video_path)
    try:
        frames = _grab_by_index(cap, targets)
    finally:
        cap.release()
    if frames is None:
        raise RuntimeError("no_decodable_frames")

    wl = np.zeros((n, 2), dtype=np.float32)
    wr = np.zeros((n, 2), dtype=np.float32)
    pl = np.zeros(n, dtype=bool)
    pr = np.zeros(n, dtype=bool)
    body = np.full(n, 0.30, dtype=np.float32)
    luma = np.zeros((n, 3), dtype=np.float32)

    pose_kwargs = dict(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with mp.solutions.pose.Pose(**pose_kwargs) as pose_model:
        for i, frame in enumerate(frames):
            luma[i] = _luma_stats(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose_model.process(rgb)
            lms = getattr(res, "pose_landmarks", None)
            if lms is None:
                continue
            p = lms.landmark
            lw, rw = p[POSE_LEFT_WRIST], p[POSE_RIGHT_WRIST]
            ls, rs = p[POSE_LEFT_SHOULDER], p[POSE_RIGHT_SHOULDER]
            if lw.visibility >= POSE_VIS_MIN:
                wl[i] = (lw.x, lw.y)
                pl[i] = True
            if rw.visibility >= POSE_VIS_MIN:
                wr[i] = (rw.x, rw.y)
                pr[i] = True
            if ls.visibility >= POSE_VIS_MIN and rs.visibility >= POSE_VIS_MIN:
                sw = float(np.hypot(ls.x - rs.x, ls.y - rs.y))
                if sw > 0.03:
                    body[i] = sw

    env = _trim.motion_envelope(wl, wr, pl, pr, body, cfg)
    p_start, p_end, motion_score, info = _trim.find_active_span(env, cfg)

    luma_summary = {
        "mean_luma": float(np.median(luma[:, 0])),
        "laplacian_var": float(np.median(luma[:, 1])),
        "backlight_ratio": float(np.median(luma[:, 2])),
        "pose_scan_rate": float((pl | pr).mean()),
    }
    scan_info = {
        "scan_frames": n,
        "p_start": round(float(p_start), 4),
        "p_end": round(float(p_end), 4),
        "motion_score": round(float(motion_score), 5),
        "trim_fallback": info.get("fallback"),
        "motion_threshold": info.get("threshold"),
        "raw_span": info.get("raw_span"),
    }
    return p_start, p_end, motion_score, scan_info, luma_summary


# ---------------------------------------------------------------------------
# pass 2 — frozen Holistic extraction over the trimmed 64 frames
# (logic copied verbatim from scripts/76 process_video, lines ~232-327)
# ---------------------------------------------------------------------------
def _holistic_extract(frames: list[np.ndarray]) -> tuple[dict, dict]:
    left_seq = np.zeros((NUM_FRAMES, 21, 3), dtype=np.float32)
    right_seq = np.zeros((NUM_FRAMES, 21, 3), dtype=np.float32)
    pose_seq = np.zeros((NUM_FRAMES, 33, 4), dtype=np.float32)
    face_seq = np.zeros((NUM_FRAMES, len(_s76.FACE_IDS), 3), dtype=np.float32)

    left_detected = np.zeros(NUM_FRAMES, dtype=np.uint8)
    right_detected = np.zeros(NUM_FRAMES, dtype=np.uint8)
    left_rescued = np.zeros(NUM_FRAMES, dtype=np.uint8)
    right_rescued = np.zeros(NUM_FRAMES, dtype=np.uint8)
    pose_detected = np.zeros(NUM_FRAMES, dtype=np.uint8)
    face_detected = np.zeros(NUM_FRAMES, dtype=np.uint8)

    with _s76.mp_holistic.Holistic(**_pp.HOLISTIC_KWARGS) as holistic_model, \
         _s76.mp_hands.Hands(**_pp.HANDS_KWARGS) as hands_model:
        for t, frame_bgr in enumerate(frames):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = holistic_model.process(rgb)

            left, left_ok = _s76.landmarks_xyz(result.left_hand_landmarks, 21)
            right, right_ok = _s76.landmarks_xyz(result.right_hand_landmarks, 21)
            pose, pose_ok = _s76.pose_xyzw(result.pose_landmarks)
            face, face_ok = _s76.selected_face_xyz(result.face_landmarks)

            if pose_ok:
                pose_detected[t] = 1
            if face_ok:
                face_detected[t] = 1
            if left_ok:
                left_detected[t] = 1
            if right_ok:
                right_detected[t] = 1

            if (not left_ok) and pose_ok:
                rescued, ok = _s76.rescue_hand_from_wrist(frame_bgr, pose, "left", hands_model)
                if ok:
                    left = rescued
                    left_rescued[t] = 1
            if (not right_ok) and pose_ok:
                rescued, ok = _s76.rescue_hand_from_wrist(frame_bgr, pose, "right", hands_model)
                if ok:
                    right = rescued
                    right_rescued[t] = 1

            left_seq[t] = left
            right_seq[t] = right
            pose_seq[t] = pose
            face_seq[t] = face

    combined = np.stack(
        [
            _s76.flatten_frame(left_seq[t], right_seq[t], pose_seq[t], face_seq[t])
            for t in range(NUM_FRAMES)
        ],
        axis=0,
    )

    left_final = np.maximum(left_detected, left_rescued)
    right_final = np.maximum(right_detected, right_rescued)

    stats = {
        "status": "ok",
        "left_detect_rate_raw": float(left_detected.mean()),
        "right_detect_rate_raw": float(right_detected.mean()),
        "left_rescue_rate": float(left_rescued.mean()),
        "right_rescue_rate": float(right_rescued.mean()),
        "left_final_rate": float(left_final.mean()),
        "right_final_rate": float(right_final.mean()),
        "both_hands_final_rate": float(np.logical_and(left_final, right_final).mean()),
        "pose_rate": float(pose_detected.mean()),
        "face_rate": float(face_detected.mean()),
        "frames": NUM_FRAMES,
        "feature_dim": int(combined.shape[1]),
    }
    arrays = {
        "left_hand": left_seq,
        "right_hand": right_seq,
        "pose": pose_seq,
        "face_selected": face_seq,
        "face_indices": _s76.FACE_IDS,
        "combined_raw": combined,
        "left_detected": left_detected,
        "right_detected": right_detected,
        "left_rescued": left_rescued,
        "right_rescued": right_rescued,
        "pose_detected": pose_detected,
        "face_detected": face_detected,
    }
    return arrays, stats


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def extract_trimmed(
    video_path: str,
    trim_cfg: _trim.TrimConfig | None = None,
    disable_trim: bool = False,
) -> tuple[dict, dict]:
    """
    Returns (arrays, stats). `arrays` matches scripts/76 `process_video` exactly.
    `stats` adds the trim + quality-relevant fields:
        original_duration_s, trimmed_duration_s, trimmed_frame_range,
        motion_score, scan_frames, trim_fallback, fps, width, height,
        mean_luma, laplacian_var, backlight_ratio, pose_scan_rate, trimmed
    """
    cfg = trim_cfg or _trim.DEFAULT_TRIM
    t0 = time.perf_counter()

    cap, total, fps, w, h = _open(video_path)
    cap.release()
    original_duration_s = (total / fps) if fps > 1e-3 else float("nan")

    if disable_trim or total < 8:
        p_start, p_end, motion_score = 0.0, 1.0, float("nan")
        scan_info = {"scan_frames": 0, "trim_fallback": "disabled" if disable_trim else "clip_too_short"}
        luma_summary = {"mean_luma": float("nan"), "laplacian_var": float("nan"),
                        "backlight_ratio": float("nan"), "pose_scan_rate": float("nan")}
    else:
        p_start, p_end, motion_score, scan_info, luma_summary = _scan(video_path, total, cfg)

    lo, hi = _trim.span_to_frame_range(p_start, p_end, total)
    targets = np.linspace(lo, hi, NUM_FRAMES).round().astype(np.int64)

    cap, *_ = _open(video_path)
    try:
        frames = _grab_by_index(cap, targets)
    finally:
        cap.release()
    if frames is None:
        raise RuntimeError("no_decodable_frames")

    arrays, stats = _holistic_extract(frames)

    trimmed_frames = hi - lo + 1
    stats.update(
        {
            "original_duration_s": round(original_duration_s, 3) if original_duration_s == original_duration_s else None,
            "trimmed_duration_s": round((trimmed_frames / fps), 3) if fps > 1e-3 else None,
            "trimmed_frame_range": [int(lo), int(hi)],
            "total_frames": int(total),
            "motion_score": round(float(motion_score), 5) if motion_score == motion_score else None,
            "fps": round(fps, 2),
            "width": w,
            "height": h,
            "trimmed": bool(scan_info.get("trim_fallback") in (None,)),
            "trim_fallback": scan_info.get("trim_fallback"),
            "scan_frames": scan_info.get("scan_frames", 0),
            "p_start": scan_info.get("p_start"),
            "p_end": scan_info.get("p_end"),
            "mean_luma": luma_summary.get("mean_luma"),
            "laplacian_var": luma_summary.get("laplacian_var"),
            "backlight_ratio": luma_summary.get("backlight_ratio"),
            "pose_scan_rate": luma_summary.get("pose_scan_rate"),
            "extract_seconds": round(time.perf_counter() - t0, 2),
        }
    )
    return arrays, stats
