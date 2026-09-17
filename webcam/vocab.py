"""
Phase 1 — supported-vocabulary verification.

The INCLUDE262 classifier is CLOSED-SET: 262 signs, class_id 0..261, listed in
metadata/include262_class_mapping_final.csv. If an intended sign is not one of those
262 classes the system must say so plainly rather than silently mapping it elsewhere.

Usage:
    python -m webcam.vocab "orange"          # search
    python -m webcam.vocab --list             # print all 262
    python -m webcam.vocab --check goodnight  # exact-membership test (exit 0/1)

Importable:
    from webcam.vocab import search, contains, all_classes, OUT_OF_VOCAB_MESSAGE
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
CLASS_MAP_CSV = ROOT / "metadata" / "include262_class_mapping_final.csv"
NUM_CLASSES = 262

OUT_OF_VOCAB_MESSAGE = "This sign is outside the current 262-sign vocabulary."


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse non-alphanumerics — 'thank you' ~ 'thankyou'."""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


@lru_cache(maxsize=1)
def _load() -> list[dict]:
    if not CLASS_MAP_CSV.is_file():
        raise FileNotFoundError(f"Class mapping not found: {CLASS_MAP_CSV}")
    rows: list[dict] = []
    with open(CLASS_MAP_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({"class_id": int(row["class_id"]), "label": row["label"].strip()})
    if len(rows) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} classes, found {len(rows)} in {CLASS_MAP_CSV}")
    return rows


def all_classes() -> list[dict]:
    """[{'class_id': int, 'label': str}, ...] ordered by class_id."""
    return [dict(r) for r in _load()]


def contains(label: str) -> Optional[dict]:
    """Exact (normalized) membership test → the matching row, or None."""
    key = _normalize(label)
    for r in _load():
        if _normalize(r["label"]) == key:
            return dict(r)
    return None


def search(query: str, limit: int = 12) -> list[dict]:
    """
    Ranked matches for a free-text query. Each result:
        {class_id, label, score (0..1), exact (bool)}
    Empty list ⇒ nothing close ⇒ caller shows OUT_OF_VOCAB_MESSAGE.
    """
    q = _normalize(query)
    if not q:
        return []

    scored: list[dict] = []
    for r in _load():
        norm = _normalize(r["label"])
        if norm == q:
            score, exact = 1.0, True
        elif len(q) >= 2 and q in norm:
            score, exact = 0.80 + 0.15 * (len(q) / max(len(norm), 1)), False
        elif len(norm) >= 4 and norm in q:
            score, exact = 0.62, False
        else:
            score = difflib.SequenceMatcher(None, q, norm).ratio() * 0.65
            exact = False
        if score >= 0.34 or exact:
            scored.append({"class_id": r["class_id"], "label": r["label"],
                           "score": round(float(score), 3), "exact": exact})

    scored.sort(key=lambda d: (-d["score"], d["label"]))
    return scored[:limit]


def _cli(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="INCLUDE262 vocabulary lookup (closed-set, 262 signs).")
    ap.add_argument("query", nargs="?", help="sign to search for")
    ap.add_argument("--list", action="store_true", help="print all 262 supported signs")
    ap.add_argument("--check", metavar="LABEL", help="exact membership test; exit 0 if supported else 1")
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args(argv)

    if args.list:
        for r in all_classes():
            print(f"{r['class_id']:3d}  {r['label']}")
        return 0

    if args.check is not None:
        hit = contains(args.check)
        if hit:
            print(f"supported: class_id={hit['class_id']} label={hit['label']!r}")
            return 0
        print(OUT_OF_VOCAB_MESSAGE)
        return 1

    if not args.query:
        ap.print_help()
        return 2

    results = search(args.query, limit=args.limit)
    if not results:
        print(OUT_OF_VOCAB_MESSAGE)
        return 1

    exact = [r for r in results if r["exact"]]
    if exact:
        r = exact[0]
        print(f"EXACT MATCH: class_id={r['class_id']} label={r['label']!r}")
    else:
        print(f"No exact match for {args.query!r}. Closest supported signs:")
    for r in results:
        tag = "  <-- exact" if r["exact"] else ""
        print(f"  {r['class_id']:3d}  {r['label']:<20} (score {r['score']}){tag}")
    if not exact:
        print("\n" + OUT_OF_VOCAB_MESSAGE)
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(_cli())
