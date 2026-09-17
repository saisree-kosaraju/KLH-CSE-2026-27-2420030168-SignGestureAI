"""
Module 3 (segmentation primitives) / first-milestone item: ACTIVE-SIGN TRIMMING.

A raw webcam clip is `idle -> sign -> idle`. The frozen extractor
(scripts/76 `sample_64_frames`) samples 64 frames linearly across the WHOLE clip,
so idle padding dilutes the sign and shifts the temporal distribution away from
INCLUDE (whose videos are already tight sign clips). This module finds the active
span from hand/wrist motion so the 64-frame sampler can be pointed at the sign only.

Pure functions + a config dataclass. No file I/O, no MediaPipe here — the caller
(`webcam_extract.py`) supplies wrist trajectories. Everything is expressed in
normalized clip position [0, 1] so it is independent of fps and frame count.
The model input size (64 frames) is never changed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrimConfig:
    """
    Deliberately CONSERVATIVE. INCLUDE videos are already tight sign clips, so on
    them trimming must be close to a no-op; the goal is only to strip the clearly
    idle head/tail of a raw webcam clip (hands resting), never to cut into the sign.
    The threshold is a fraction of the clip's PEAK motion (not a percentile), so a
    clip that is all-signing keeps all of its frames.
    """
    # --- motion scan -------------------------------------------------------
    scan_frames: int = 120          # max frames sampled to build the motion envelope
    smooth_win: int = 5             # moving-average window on the motion signal (odd)

    # --- activity thresholding ------------------------------------------
    peak_fraction: float = 0.12     # active <=> motion >= peak_fraction * (95th-pctl motion)
    motion_floor: float = 0.006     # ...but always at least this (shoulder-width units / frame)
    sustain: int = 2                # need this many consecutive active scan-frames to mark an edge

    # --- span shaping (all in clip-position units, [0,1]) -----------------
    pre_roll: float = 0.08          # added before the detected start
    post_roll: float = 0.12         # added after the detected end
    min_active_frames: int = 6      # scan-frames of activity; fewer => fall back to full clip
    min_span_frac: float = 0.50     # NEVER keep less than half the clip — hard safety floor


DEFAULT_TRIM = TrimConfig()


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or x.size < win:
        return x.astype(np.float32)
    win = int(win) | 1  # force odd
    pad = win // 2
    padded = np.pad(x.astype(np.float32), pad, mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(padded, kernel, mode="valid")[: x.size]


def motion_envelope(
    wrist_l: np.ndarray,
    wrist_r: np.ndarray,
    present_l: np.ndarray,
    present_r: np.ndarray,
    body_scale: np.ndarray,
    cfg: TrimConfig = DEFAULT_TRIM,
) -> np.ndarray:
    """
    Per-scan-frame activity signal, shoulder-width-normalized and smoothed.

    wrist_* : (N, 2) normalized image coords (x, y in [0, 1]).
    present_*: (N,)  bool — was that wrist detected on that scan frame.
    body_scale: (N,) shoulder width in normalized units (>0; caller supplies a floor).
    Returns (N,) float32, >= 0.
    """
    n = len(wrist_l)
    env = np.zeros(n, dtype=np.float32)
    scale = np.clip(np.asarray(body_scale, dtype=np.float32), 0.05, None)

    for i in range(1, n):
        speeds = []
        for wrist, present in ((wrist_l, present_l), (wrist_r, present_r)):
            if present[i] and present[i - 1]:
                d = np.linalg.norm(wrist[i] - wrist[i - 1]) / scale[i]
                speeds.append(float(d))
        if speeds:
            env[i] = max(speeds)
        else:
            env[i] = env[i - 1]  # hold through a detection dropout

    env[0] = env[1] if n > 1 else 0.0
    return _moving_average(env, cfg.smooth_win)


def _first_last_sustained(active: np.ndarray, sustain: int) -> tuple[int, int] | None:
    """
    Outer bounds of activity: first index that begins `sustain` consecutive active
    frames, and last index that ends `sustain` consecutive active frames. Using the
    outer envelope (not the longest run) keeps multi-part signs with an internal
    lull intact.
    """
    n = len(active)
    sustain = max(1, sustain)
    first = None
    for i in range(n - sustain + 1):
        if active[i : i + sustain].all():
            first = i
            break
    if first is None:
        return None
    last = None
    for i in range(n - sustain, -1, -1):
        if active[i : i + sustain].all():
            last = i + sustain - 1
            break
    if last is None or last <= first:
        return None
    return first, last


def find_active_span(
    env: np.ndarray, cfg: TrimConfig = DEFAULT_TRIM
) -> tuple[float, float, float, dict]:
    """
    Locate the active sign inside the motion envelope.

    Returns (p_start, p_end, motion_score, info). p_start/p_end are clip positions
    in [0, 1], already pre/post-rolled and clamped. If no confident span is found
    (or trimming would remove too much) it returns the FULL clip (0.0, 1.0) with
    info['fallback'] set — so this can never make things worse than the frozen path.
    """
    n = len(env)
    info: dict = {"fallback": None, "threshold": 0.0, "raw_span": None, "n_scan": n}
    if n < 6:
        info["fallback"] = "too_few_scan_frames"
        return 0.0, 1.0, float(env.mean() if n else 0.0), info

    peak = float(np.percentile(env, 95))
    thr = max(cfg.motion_floor, peak * cfg.peak_fraction)
    info["threshold"] = round(thr, 5)
    info["peak"] = round(peak, 5)

    active = env >= thr
    if int(active.sum()) < cfg.min_active_frames:
        info["fallback"] = "insufficient_motion"
        return 0.0, 1.0, float(env.mean()), info

    bounds = _first_last_sustained(active, cfg.sustain)
    if bounds is None:
        info["fallback"] = "no_sustained_activity"
        return 0.0, 1.0, float(env.mean()), info

    s, e = bounds
    p_start = s / (n - 1)
    p_end = e / (n - 1)
    info["raw_span"] = (round(p_start, 4), round(p_end, 4))
    motion_score = float(env[s : e + 1].mean())

    p_start = max(0.0, p_start - cfg.pre_roll)
    p_end = min(1.0, p_end + cfg.post_roll)

    if (p_end - p_start) < cfg.min_span_frac:
        # Activity fills most of the clip (or detection was jumpy) — keep everything.
        info["fallback"] = "span_covers_most"
        return 0.0, 1.0, motion_score, info

    return p_start, p_end, motion_score, info


def span_to_frame_range(p_start: float, p_end: float, total_frames: int) -> tuple[int, int]:
    """Map a clip-position span to inclusive original-frame indices."""
    if total_frames <= 1:
        return 0, max(0, total_frames - 1)
    lo = int(round(p_start * (total_frames - 1)))
    hi = int(round(p_end * (total_frames - 1)))
    lo = max(0, min(lo, total_frames - 1))
    hi = max(lo, min(hi, total_frames - 1))
    return lo, hi
