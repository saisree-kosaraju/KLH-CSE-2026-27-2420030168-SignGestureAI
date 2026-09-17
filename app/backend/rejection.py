"""
Module 5 — CONFIDENCE + AMBIGUITY (UNKNOWN) REJECTION.

The 262-class classifier always produces a softmax; it never says "I don't know".
For live/webcam use we gate on top-1 confidence and the top-1 / top-2 margin so an
ambiguous input surfaces "Sign uncertain — please repeat" instead of a confident
random word.

Starting DEVELOPMENT thresholds only (milestone spec). They must be tuned later on
WEBCAM development/validation data, never on the locked INCLUDE test split.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RejectionConfig:
    min_confidence: float = 0.70   # top-1 probability floor
    min_margin: float = 0.15       # top1 - top2 probability floor


DEFAULT_REJECTION = RejectionConfig()

UNCERTAIN_MESSAGE = "Sign uncertain - please repeat."
OUT_OF_VOCAB_NOTE = (
    "The model recognises a fixed set of 262 signs. If you signed something else, "
    "it cannot be recognised."
)


def evaluate(top5: list[dict], cfg: RejectionConfig = DEFAULT_REJECTION) -> dict:
    """
    top5: [{class_id, sign, probability, rank?}, ...] sorted by probability desc.
    Returns the rejection verdict + the top-1/top-2 breakdown.
    """
    if not top5:
        return {
            "accepted": False,
            "rejection_reason": "no_prediction",
            "message": UNCERTAIN_MESSAGE,
            "top1_sign": None,
            "top1_confidence": 0.0,
            "top2_sign": None,
            "top2_confidence": 0.0,
            "top2_margin": 0.0,
        }

    t1 = top5[0]
    t2 = top5[1] if len(top5) > 1 else {"sign": None, "probability": 0.0}
    p1 = float(t1["probability"])
    p2 = float(t2["probability"])
    margin = p1 - p2

    reason = None
    if p1 < cfg.min_confidence:
        reason = "low_confidence"
    elif margin < cfg.min_margin:
        reason = "ambiguous_margin"

    return {
        "accepted": reason is None,
        "rejection_reason": reason,
        "message": None if reason is None else UNCERTAIN_MESSAGE,
        "top1_sign": t1["sign"],
        "top1_confidence": round(p1, 4),
        "top2_sign": t2["sign"],
        "top2_confidence": round(p2, 4),
        "top2_margin": round(margin, 4),
        "thresholds": {"min_confidence": cfg.min_confidence, "min_margin": cfg.min_margin},
    }
