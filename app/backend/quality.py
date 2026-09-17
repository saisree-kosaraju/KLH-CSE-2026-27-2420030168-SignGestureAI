"""
Module 2 / Module 16 — LIVE CAMERA QUALITY GATE.

Given the extracted arrays + stats from `webcam_extract.extract_trimmed`, decide
whether the clip is good enough to run the classifier on, and produce actionable
feedback. Obviously-unusable input (no hand at all, far too short, no pose) is
BLOCKED before the model runs — returning a confident random label on garbage
input is worse than saying "please repeat".

Pure function + a config dataclass. User-facing strings are exactly the ones the
milestone spec lists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    min_hands_any_rate: float = 0.45   # >= 1 hand present in this fraction of the 64 frames
    hard_no_hand_rate: float = 0.10    # below this -> BLOCK ("no hand detected")
    min_pose_rate: float = 0.55        # upper body must be tracked
    hard_no_pose_rate: float = 0.20
    min_usable_frames: int = 20        # frames with (>=1 hand AND pose)
    min_duration_s: float = 0.8
    max_duration_s: float = 9.0
    edge_margin: float = 0.045         # landmark within this of the frame edge = "at the edge"
    max_hand_edge_rate: float = 0.35   # fraction of present-hand frames with a hand at the edge
    min_motion_score: float = 0.006    # trim motion score below this = "no real signing"
    max_palm_scale: float = 0.42       # median hand span (frac of frame) above this = "too close"
    min_palm_scale: float = 0.045      # ... below this = "too far"
    blur_var_min: float = 22.0         # median Laplacian variance below this = motion blur / soft focus
    backlight_ratio_max: float = 1.55  # surround much brighter than subject = backlight


DEFAULT_QUALITY = QualityConfig()

# stable ids -> user-facing message (plain ASCII — renders cleanly everywhere)
_MESSAGES = {
    "no_hand": "No hand detected. Bring at least one hand into the frame.",
    "low_hand": "Keep your hands inside the frame throughout the sign.",
    "hand_at_edge": "Keep your hands inside the frame - they are drifting off the edge.",
    "no_pose": "Center your upper body so your shoulders are visible.",
    "low_pose": "Center your upper body - your shoulders keep leaving the frame.",
    "too_close": "Move slightly farther from the camera.",
    "too_far": "Move a little closer to the camera.",
    "too_short": "Recording too short - hold the sign for about 2 seconds.",
    "too_long": "Recording quite long - one sign per clip works best.",
    "no_motion": "Perform the sign clearly - very little hand motion was detected.",
    "low_light": "Improve lighting - the scene is quite dark.",
    "backlight": "Move away from strong backlight (a bright window behind you).",
    "blurry": "Hold steadier or improve focus - the video looks blurry.",
    "few_frames": "Too few usable frames - make sure your hands and upper body stay visible.",
}


def _median_palm_scale(arrays: dict) -> float:
    """Median wrist->middle-MCP distance (fraction of frame) over frames where a hand is present."""
    vals = []
    for hand_key, det_key, res_key in (
        ("left_hand", "left_detected", "left_rescued"),
        ("right_hand", "right_detected", "right_rescued"),
    ):
        hand = arrays[hand_key]
        present = np.maximum(arrays[det_key], arrays[res_key]).astype(bool)
        for t in np.where(present)[0]:
            w = hand[t, 0, :2]
            m = hand[t, 9, :2]
            d = float(np.linalg.norm(m - w))
            if d > 1e-4:
                vals.append(d)
    return float(np.median(vals)) if vals else float("nan")


def _hand_edge_rate(arrays: dict, margin: float) -> float:
    total = 0
    at_edge = 0
    for hand_key, det_key, res_key in (
        ("left_hand", "left_detected", "left_rescued"),
        ("right_hand", "right_detected", "right_rescued"),
    ):
        hand = arrays[hand_key]
        present = np.maximum(arrays[det_key], arrays[res_key]).astype(bool)
        for t in np.where(present)[0]:
            xy = hand[t, :, :2]
            total += 1
            if (xy < margin).any() or (xy > 1.0 - margin).any():
                at_edge += 1
    return (at_edge / total) if total else 0.0


def assess(arrays: dict, stats: dict, cfg: QualityConfig = DEFAULT_QUALITY) -> dict:
    """
    Returns:
        {
          overall: "ok" | "warn" | "block",
          checks:  [ {id, passed, severity: "info"|"warn"|"block", message} ],
          messages: [str, ...]          # ordered, most severe first
          metrics: {...}                # the raw numbers used
        }
    """
    left_present = np.maximum(arrays["left_detected"], arrays["left_rescued"]).astype(bool)
    right_present = np.maximum(arrays["right_detected"], arrays["right_rescued"]).astype(bool)
    pose_present = arrays["pose_detected"].astype(bool)

    hands_any_rate = float((left_present | right_present).mean())
    pose_rate = float(pose_present.mean())
    usable_frames = int(((left_present | right_present) & pose_present).sum())
    duration_s = stats.get("trimmed_duration_s") or stats.get("original_duration_s") or 0.0
    motion_score = stats.get("motion_score")
    palm_scale = _median_palm_scale(arrays)
    edge_rate = _hand_edge_rate(arrays, cfg.edge_margin)
    lap_var = stats.get("laplacian_var")
    backlight = stats.get("backlight_ratio")
    mean_luma = stats.get("mean_luma")

    checks: list[dict] = []

    def add(cid: str, passed: bool, severity: str):
        checks.append({"id": cid, "passed": passed, "severity": severity, "message": _MESSAGES[cid]})

    # --- hard blocks ----------------------------------------------------
    add("no_hand", hands_any_rate >= cfg.hard_no_hand_rate, "block")
    add("no_pose", pose_rate >= cfg.hard_no_pose_rate, "block")
    add("too_short", (duration_s or 0.0) >= cfg.min_duration_s, "block")
    add("few_frames", usable_frames >= max(8, cfg.min_usable_frames // 2), "block")

    # --- warnings -----------------------------------------------------
    add("low_hand", hands_any_rate >= cfg.min_hands_any_rate, "warn")
    add("low_pose", pose_rate >= cfg.min_pose_rate, "warn")
    add("hand_at_edge", edge_rate <= cfg.max_hand_edge_rate, "warn")
    add("too_long", (duration_s or 0.0) <= cfg.max_duration_s, "warn")
    if motion_score is not None and motion_score == motion_score:
        add("no_motion", motion_score >= cfg.min_motion_score, "warn")
    if palm_scale == palm_scale:  # not NaN
        add("too_close", palm_scale <= cfg.max_palm_scale, "warn")
        add("too_far", palm_scale >= cfg.min_palm_scale, "warn")
    if mean_luma is not None and mean_luma == mean_luma:
        add("low_light", mean_luma >= 0.22, "warn")
    if backlight is not None and backlight == backlight:
        add("backlight", backlight <= cfg.backlight_ratio_max, "warn")
    if lap_var is not None and lap_var == lap_var:
        add("blurry", lap_var >= cfg.blur_var_min, "warn")

    failed = [c for c in checks if not c["passed"]]
    if any(c["severity"] == "block" for c in failed):
        overall = "block"
    elif failed:
        overall = "warn"
    else:
        overall = "ok"

    order = {"block": 0, "warn": 1, "info": 2}
    messages = [c["message"] for c in sorted(failed, key=lambda c: order[c["severity"]])]

    return {
        "overall": overall,
        "checks": checks,
        "messages": messages,
        "metrics": {
            "hands_any_rate": round(hands_any_rate, 3),
            "pose_rate": round(pose_rate, 3),
            "usable_frames": usable_frames,
            "duration_s": round(float(duration_s or 0.0), 2),
            "motion_score": motion_score,
            "median_palm_scale": round(palm_scale, 4) if palm_scale == palm_scale else None,
            "hand_edge_rate": round(edge_rate, 3),
            "laplacian_var": round(lap_var, 1) if isinstance(lap_var, (int, float)) and lap_var == lap_var else None,
            "backlight_ratio": round(backlight, 3) if isinstance(backlight, (int, float)) and backlight == backlight else None,
            "mean_luma": round(mean_luma, 3) if isinstance(mean_luma, (int, float)) and mean_luma == mean_luma else None,
        },
    }
