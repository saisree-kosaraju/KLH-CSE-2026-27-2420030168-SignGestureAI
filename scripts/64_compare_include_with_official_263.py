from pathlib import Path
import re
import difflib
import pandas as pd

ROOT = Path.cwd()
META = ROOT / "metadata"
DATASET = ROOT / "datasets" / "INCLUDE"
RESULTS = ROOT / "results"

SOURCE = META / "include_metadata.csv"
REPORT = META / "include_official_263_comparison_v1.csv"
SUMMARY = RESULTS / "include_official_263_comparison_v1.txt"

OFFICIAL_LABELS = ['actor', 'adult', 'afternoon', 'alive', 'alright', 'animal', 'artist', 'attack', 'author', 'baby', 'bad', 'bag', 'ball', 'bank', 'bathroom', 'beautiful', 'bed', 'bedroom', 'bicycle', 'biglarge', 'bill', 'bird', 'black', 'blind', 'blue', 'boat', 'book', 'box', 'boy', 'brother', 'brown', 'bus', 'camera', 'car', 'card', 'cat', 'cellphone', 'chair', 'cheap', 'child', 'city', 'clean', 'clock', 'clothing', 'cold', 'colour', 'computer', 'cool', 'court', 'cow', 'crowd', 'curved', 'daughter', 'dead', 'deaf', 'death', 'deep', 'dirty', 'doctor', 'dog', 'door', 'dream', 'dress', 'dry', 'election', 'energy', 'evening', 'exercise', 'exmonsoon', 'expensive', 'fall', 'family', 'famous', 'fan', 'fast', 'father', 'female', 'fish', 'flat', 'friday', 'friend', 'gift', 'girl', 'god', 'good', 'goodafternoon', 'goodevening', 'goodmorning', 'goodnight', 'grandfather', 'grandmother', 'green', 'grey', 'ground', 'gun', 'happy', 'hard', 'hat', 'he', 'healthy', 'heavy', 'hello', 'high', 'horse', 'hospital', 'hot', 'hour', 'house', 'howareyou', 'husband', 'i', 'india', 'it', 'job', 'key', 'king', 'kitchen', 'lamp', 'laptop', 'lawyer', 'letter', 'library', 'light', 'location', 'lock', 'long', 'loose', 'loud', 'low', 'male', 'man', 'manager', 'market', 'marriage', 'mean', 'medicine', 'minute', 'monday', 'money', 'month', 'morning', 'mother', 'mouse', 'narrow', 'neighbour', 'new', 'newspaper', 'nice', 'night', 'office', 'old', 'orange', 'page', 'paint', 'pant', 'paper', 'parent', 'park', 'patient', 'peace', 'pen', 'pencil', 'photograph', 'pink', 'plane', 'player', 'pleased', 'pocket', 'police', 'poor', 'president', 'price', 'priest', 'queen', 'quiet', 'raceethnicity', 'radio', 'red', 'religion', 'reporter', 'restaurant', 'rich', 'ring', 'sad', 'saturday', 'school', 'science', 'screen', 'season', 'second', 'secondnumber', 'secretary', 'shallow', 'she', 'shirt', 'shoes', 'short', 'sick', 'sign', 'sister', 'skirt', 'slow', 'smalllittle', 'soap', 'soft', 'soldier', 'son', 'sport', 'spring', 'storeorshop', 'streetorroad', 'strong', 'student', 'suit', 'summer', 'sunday', 'table', 'tall', 'teacher', 'team', 'technology', 'telephone', 'television', 'temple', 'thankyou', 'they', 'thick', 'thin', 'thursday', 'tight', 'time', 'today', 'tomorrow', 'tool', 'train', 'trainstation', 'trainticket', 'transportation', 'truck', 'tshirt', 'tuesday', 'ugly', 'university', 'waiter', 'war', 'warm', 'we', 'weak', 'wednesday', 'week', 'wet', 'white', 'wide', 'wife', 'window', 'winter', 'woman', 'year', 'yellow', 'yesterday', 'you', 'young', 'youplural']


def normalize_label(value):
    s = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("INCLUDE STEP 3 - COMPARE LOCAL 257 LABELS WITH OFFICIAL 263")
    print("=" * 78)

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    df = pd.read_csv(SOURCE)

    if "label" not in df.columns:
        raise RuntimeError(
            f"'label' column missing. Columns: {df.columns.tolist()}"
        )

    df["label_raw"] = df["label"].astype(str).str.strip()
    df["label_norm_audit"] = df["label_raw"].map(normalize_label)

    local_norm_to_raw = (
        df.groupby("label_norm_audit")["label_raw"]
        .agg(lambda x: sorted(set(x)))
        .to_dict()
    )

    local_norm = set(local_norm_to_raw)
    official_norm_to_raw = {normalize_label(x): x for x in OFFICIAL_LABELS}
    official_norm = set(official_norm_to_raw)

    missing_norm = sorted(official_norm - local_norm)
    extra_norm = sorted(local_norm - official_norm)

    missing = [official_norm_to_raw[x] for x in missing_norm]

    print(f"\nOfficial labels : {len(OFFICIAL_LABELS)}")
    print(f"Local labels    : {df['label_raw'].nunique()}")
    print(f"Local normalized: {len(local_norm)}")
    print(f"Missing official: {len(missing)}")
    print(f"Extra local     : {len(extra_norm)}")

    print("\n" + "-" * 78)
    print("MISSING OFFICIAL LABELS")
    print("-" * 78)

    if missing:
        for x in missing:
            print(f"  - {x}")
    else:
        print("None")

    print("\n" + "-" * 78)
    print("EXTRA / UNMATCHED LOCAL LABELS")
    print("-" * 78)

    rows = []

    for norm in extra_norm:
        raw_values = local_norm_to_raw[norm]
        suggestions = difflib.get_close_matches(
            norm,
            list(official_norm),
            n=3,
            cutoff=0.55,
        )
        suggestion_labels = [official_norm_to_raw[s] for s in suggestions]

        print(
            f"  - local={raw_values} | possible official={suggestion_labels}"
        )

        rows.append(
            {
                "type": "extra_local",
                "normalized": norm,
                "local_raw": " | ".join(raw_values),
                "official_label": "",
                "suggestions": " | ".join(suggestion_labels),
            }
        )

    for official_label in missing:
        norm = normalize_label(official_label)
        suggestions = difflib.get_close_matches(
            norm,
            list(local_norm),
            n=3,
            cutoff=0.55,
        )
        local_suggestions = []
        for s in suggestions:
            local_suggestions.extend(local_norm_to_raw[s])

        rows.append(
            {
                "type": "missing_official",
                "normalized": norm,
                "local_raw": "",
                "official_label": official_label,
                "suggestions": " | ".join(sorted(set(local_suggestions))),
            }
        )

    pd.DataFrame(rows).to_csv(REPORT, index=False)

    print("\n" + "-" * 78)
    print("SEARCHING DATASET FILENAMES FOR MISSING LABEL WORDS")
    print("-" * 78)

    dataset_hits = {}

    if DATASET.exists():
        files = [p for p in DATASET.rglob("*") if p.is_file()]

        for label in missing:
            key = normalize_label(label)
            hits = [
                str(p.relative_to(DATASET))
                for p in files
                if key and key in normalize_label(p.name)
            ]
            dataset_hits[label] = hits[:20]

            print(
                f"{label:20s} -> {len(hits)} filename matches"
            )
    else:
        print(f"Dataset folder not found: {DATASET}")

    summary_lines = [
        "=" * 78,
        "INCLUDE OFFICIAL-263 COMPARISON SUMMARY",
        "=" * 78,
        "",
        f"Official labels: {len(OFFICIAL_LABELS)}",
        f"Local raw labels: {df['label_raw'].nunique()}",
        f"Local normalized labels: {len(local_norm)}",
        f"Missing official labels: {len(missing)}",
        f"Extra/unmatched local normalized labels: {len(extra_norm)}",
        "",
        "Missing official labels:",
        *([f"- {x}" for x in missing] if missing else ["None"]),
        "",
        "Extra local labels:",
    ]

    if extra_norm:
        for norm in extra_norm:
            summary_lines.append(
                f"- {' | '.join(local_norm_to_raw[norm])}"
            )
    else:
        summary_lines.append("None")

    summary_lines += [
        "",
        "Dataset filename hits for missing labels:",
    ]

    if dataset_hits:
        for label, hits in dataset_hits.items():
            summary_lines.append(
                f"- {label}: {len(hits)} shown matches"
            )
            for h in hits[:5]:
                summary_lines.append(f"    {h}")
    else:
        summary_lines.append("No filename search performed.")

    summary_lines += [
        "",
        f"CSV report: {REPORT}",
        "",
        "NEXT:",
        "If exactly 6 official labels are missing, inspect whether their videos are absent",
        "from the downloaded dataset or only absent/mislabeled in local metadata.",
    ]

    SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "\n".join(summary_lines))
    print("=" * 78)


if __name__ == "__main__":
    main()