from pathlib import Path
import re
import urllib.request
from collections import defaultdict

import pandas as pd


ROOT = Path.cwd()
DATASET = ROOT / "datasets" / "INCLUDE"
META = ROOT / "metadata"
RESULTS = ROOT / "results"

OUTPUT_CSV = META / "include262_official_split_v2.csv"
CLASS_MAP_CSV = META / "include262_class_mapping_v2.csv"
DUPLICATES_CSV = META / "include262_duplicate_local_copies_v2.csv"
UNMATCHED_LOCAL_CSV = META / "include262_unmatched_local_videos_v2.csv"
MISSING_OFFICIAL_CSV = META / "include262_missing_official_videos_v2.csv"
SUMMARY_TXT = RESULTS / "include262_official_split_v2_summary.txt"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}

OFFICIAL_LABELS_263 = ['actor', 'adult', 'afternoon', 'alive', 'alright', 'animal', 'artist', 'attack', 'author', 'baby', 'bad', 'bag', 'ball', 'bank', 'bathroom', 'beautiful', 'bed', 'bedroom', 'bicycle', 'biglarge', 'bill', 'bird', 'black', 'blind', 'blue', 'boat', 'book', 'box', 'boy', 'brother', 'brown', 'bus', 'camera', 'car', 'card', 'cat', 'cellphone', 'chair', 'cheap', 'child', 'city', 'clean', 'clock', 'clothing', 'cold', 'colour', 'computer', 'cool', 'court', 'cow', 'crowd', 'curved', 'daughter', 'dead', 'deaf', 'death', 'deep', 'dirty', 'doctor', 'dog', 'door', 'dream', 'dress', 'dry', 'election', 'energy', 'evening', 'exercise', 'exmonsoon', 'expensive', 'fall', 'family', 'famous', 'fan', 'fast', 'father', 'female', 'fish', 'flat', 'friday', 'friend', 'gift', 'girl', 'god', 'good', 'goodafternoon', 'goodevening', 'goodmorning', 'goodnight', 'grandfather', 'grandmother', 'green', 'grey', 'ground', 'gun', 'happy', 'hard', 'hat', 'he', 'healthy', 'heavy', 'hello', 'high', 'horse', 'hospital', 'hot', 'hour', 'house', 'howareyou', 'husband', 'i', 'india', 'it', 'job', 'key', 'king', 'kitchen', 'lamp', 'laptop', 'lawyer', 'letter', 'library', 'light', 'location', 'lock', 'long', 'loose', 'loud', 'low', 'male', 'man', 'manager', 'market', 'marriage', 'mean', 'medicine', 'minute', 'monday', 'money', 'month', 'morning', 'mother', 'mouse', 'narrow', 'neighbour', 'new', 'newspaper', 'nice', 'night', 'office', 'old', 'orange', 'page', 'paint', 'pant', 'paper', 'parent', 'park', 'patient', 'peace', 'pen', 'pencil', 'photograph', 'pink', 'plane', 'player', 'pleased', 'pocket', 'police', 'poor', 'president', 'price', 'priest', 'queen', 'quiet', 'raceethnicity', 'radio', 'red', 'religion', 'reporter', 'restaurant', 'rich', 'ring', 'sad', 'saturday', 'school', 'science', 'screen', 'season', 'second', 'secondnumber', 'secretary', 'shallow', 'she', 'shirt', 'shoes', 'short', 'sick', 'sign', 'sister', 'skirt', 'slow', 'smalllittle', 'soap', 'soft', 'soldier', 'son', 'sport', 'spring', 'storeorshop', 'streetorroad', 'strong', 'student', 'suit', 'summer', 'sunday', 'table', 'tall', 'teacher', 'team', 'technology', 'telephone', 'television', 'temple', 'thankyou', 'they', 'thick', 'thin', 'thursday', 'tight', 'time', 'today', 'tomorrow', 'tool', 'train', 'trainstation', 'trainticket', 'transportation', 'truck', 'tshirt', 'tuesday', 'ugly', 'university', 'waiter', 'war', 'warm', 'we', 'weak', 'wednesday', 'week', 'wet', 'white', 'wide', 'wife', 'window', 'winter', 'woman', 'year', 'yellow', 'yesterday', 'you', 'young', 'youplural']
AVAILABLE_LABELS_262 = [x for x in OFFICIAL_LABELS_263 if x != "secondnumber"]

SPLIT_URLS = {
    "train": "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/include_train.txt",
    "val": "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/include_val.txt",
    "test": "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/include_test.txt",
}


def norm_text(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def clean_class_name(folder_name):
    # Examples:
    # "43. Pant" -> "pant"
    # "42. T-Shirt" -> "tshirt"
    # "Ex. Monsoon" -> "exmonsoon"
    s = str(folder_name).strip()
    s = re.sub(r"^\s*\d+\s*[\.\-_:)]*\s*", "", s)
    return norm_text(s)


def parse_official_text(text):
    # GitHub raw split files are rendered as whitespace-separated paths,
    # while some class folders themselves contain spaces.
    # Every entry ends with a video extension, so split AFTER extensions.
    text = text.replace("\r", " ").replace("\n", " ")
    pattern = re.compile(r"(?i)(\.mov|\.mp4|\.avi|\.mkv|\.webm)\s+")
    text = pattern.sub(lambda m: m.group(1) + "\n", text)
    entries = [x.strip() for x in text.splitlines() if x.strip()]
    return entries


def download_split(split, url):
    print(f"Downloading official {split} split...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read().decode("utf-8")
    entries = parse_official_text(data)
    print(f"  Parsed {len(entries)} official {split} paths")
    return entries


def official_key_from_relpath(relpath):
    parts = [p for p in str(relpath).replace("\\", "/").split("/") if p]
    if len(parts) < 3:
        return None, None, None

    category = norm_text(parts[0])
    class_label = clean_class_name(parts[1])
    subdirs = [norm_text(x) for x in parts[2:-1]]
    filename = parts[-1].strip().lower()

    key = "|".join([category, class_label] + subdirs + [filename])
    return key, class_label, category


def find_class_ancestor(video_path):
    rel = video_path.relative_to(DATASET)
    parts = rel.parts

    # Search from the video's parent upward, but remain inside DATASET.
    # A class folder is a folder whose normalized class name is one of
    # the official labels. "Extra" therefore cannot become a class.
    candidates = []

    for idx in range(len(parts) - 1):
        label = clean_class_name(parts[idx])
        if label in OFFICIAL_LABELS_263:
            candidates.append((idx, label))

    if not candidates:
        return None

    # Nearest matching ancestor to the video.
    idx, label = candidates[-1]
    return idx, label, parts


def local_record(video_path):
    info = find_class_ancestor(video_path)
    if info is None:
        return None

    class_idx, label, parts = info

    # The first folder beneath datasets/INCLUDE is the broad category.
    category_raw = parts[0]
    category = norm_text(category_raw)

    # Preserve optional subfolders such as Extra.
    subdirs_raw = list(parts[class_idx + 1 : -1])
    subdirs = [norm_text(x) for x in subdirs_raw]

    filename = parts[-1].lower()

    key = "|".join([category, label] + subdirs + [filename])

    return {
        "key": key,
        "label": label,
        "category": category_raw,
        "class_folder": parts[class_idx],
        "video_name": parts[-1],
        "video_path": str(video_path.resolve()),
        "relative_local_path": str(video_path.relative_to(DATASET)),
        "subdirs": "/".join(subdirs_raw),
    }


def main():
    print("=" * 78)
    print("INCLUDE STEP 6 - BUILD CLEAN 262-CLASS OFFICIAL-SPLIT METADATA")
    print("=" * 78)

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET}")

    META.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Load official split paths
    # ------------------------------------------------------------
    official_rows = []
    official_by_key = {}

    for split, url in SPLIT_URLS.items():
        entries = download_split(split, url)

        for relpath in entries:
            key, label, category = official_key_from_relpath(relpath)

            if key is None:
                continue

            row = {
                "official_relpath": relpath,
                "key": key,
                "label": label,
                "official_category_norm": category,
                "split": split,
            }

            if key in official_by_key:
                raise RuntimeError(
                    f"Duplicate official split entry detected:\n{relpath}"
                )

            official_by_key[key] = row
            official_rows.append(row)

    official_df = pd.DataFrame(official_rows)

    print("\nOfficial split summary BEFORE removing unavailable class:")
    print(official_df["split"].value_counts().to_dict())
    print(f"Official total paths: {len(official_df)}")
    print(f"Official labels represented: {official_df['label'].nunique()}")

    second_number_df = official_df[
        official_df["label"] == "secondnumber"
    ].copy()

    print(
        f"Official 'secondnumber' videos unavailable locally: "
        f"{len(second_number_df)}"
    )

    target_official_df = official_df[
        official_df["label"] != "secondnumber"
    ].copy().reset_index(drop=True)

    target_official_keys = set(target_official_df["key"])

    # ------------------------------------------------------------
    # 2. Scan local video files
    # ------------------------------------------------------------
    print("\nScanning local INCLUDE videos...")

    videos = [
        p for p in DATASET.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]

    print(f"Physical video files found: {len(videos)}")

    local_rows = []
    no_class_rows = []

    for p in videos:
        rec = local_record(p)

        if rec is None:
            no_class_rows.append(str(p))
        else:
            local_rows.append(rec)

    local_df = pd.DataFrame(local_rows)

    print(f"Videos assigned to an official class folder: {len(local_df)}")
    print(f"Videos with no recognized class ancestor: {len(no_class_rows)}")
    print(f"Unique local labels detected: {local_df['label'].nunique()}")

    present_labels = set(local_df["label"].unique())
    expected_labels = set(AVAILABLE_LABELS_262)

    missing_labels = sorted(expected_labels - present_labels)
    unexpected_labels = sorted(present_labels - expected_labels)

    print(f"Missing available labels: {len(missing_labels)}")
    print(f"Unexpected labels: {len(unexpected_labels)}")

    if missing_labels:
        print("Missing labels:", missing_labels)

    if unexpected_labels:
        print("Unexpected labels:", unexpected_labels)

    # ------------------------------------------------------------
    # 3. Collapse duplicate extracted copies by official identity
    # ------------------------------------------------------------
    grouped = defaultdict(list)

    for row in local_rows:
        grouped[row["key"]].append(row)

    duplicate_rows = []
    selected_local = {}

    for key, rows in grouped.items():
        # Prefer shortest local relative path as the canonical copy.
        rows_sorted = sorted(
            rows,
            key=lambda r: (len(r["relative_local_path"]), r["relative_local_path"].lower())
        )

        selected_local[key] = rows_sorted[0]

        if len(rows_sorted) > 1:
            for r in rows_sorted:
                duplicate_rows.append({
                    "key": key,
                    "copies": len(rows_sorted),
                    "selected": r["video_path"] == rows_sorted[0]["video_path"],
                    **r,
                })

    duplicate_df = pd.DataFrame(duplicate_rows)
    duplicate_df.to_csv(DUPLICATES_CSV, index=False)

    duplicate_keys = sum(1 for rows in grouped.values() if len(rows) > 1)

    print(f"Duplicate video identities caused by extraction copies: {duplicate_keys}")

    # ------------------------------------------------------------
    # 4. Match physical files to official split
    # ------------------------------------------------------------
    matched_rows = []
    missing_official_rows = []

    for _, off in target_official_df.iterrows():
        key = off["key"]

        if key not in selected_local:
            missing_official_rows.append(off.to_dict())
            continue

        local = selected_local[key]

        matched_rows.append({
            "video_name": local["video_name"],
            "video_path": local["video_path"],
            "label": off["label"],
            "category": local["category"],
            "class_folder": local["class_folder"],
            "official_relpath": off["official_relpath"],
            "split": off["split"],
            "duplicate_local_copies": len(grouped[key]),
        })

    matched_df = pd.DataFrame(matched_rows)
    missing_official_df = pd.DataFrame(missing_official_rows)

    missing_official_df.to_csv(MISSING_OFFICIAL_CSV, index=False)

    # Local videos that are not part of the official 262-class split.
    unmatched_local = []

    for key, row in selected_local.items():
        if key not in target_official_keys:
            unmatched_local.append(row)

    pd.DataFrame(unmatched_local).to_csv(
        UNMATCHED_LOCAL_CSV,
        index=False
    )

    print("\nMATCHING RESULTS")
    print("-" * 78)
    print(f"Official target paths (262 classes): {len(target_official_df)}")
    print(f"Matched physical videos           : {len(matched_df)}")
    print(f"Missing official physical videos  : {len(missing_official_df)}")
    print(f"Unique local videos outside split : {len(unmatched_local)}")

    # ------------------------------------------------------------
    # 5. Assign stable 262-class IDs in official label-map order
    # ------------------------------------------------------------
    class_to_id = {
        label: idx
        for idx, label in enumerate(AVAILABLE_LABELS_262)
    }

    if not matched_df.empty:
        matched_df["class_id"] = matched_df["label"].map(class_to_id)

        if matched_df["class_id"].isna().any():
            bad = matched_df[matched_df["class_id"].isna()]["label"].unique().tolist()
            raise RuntimeError(
                f"Could not map class IDs for labels: {bad}"
            )

        matched_df["class_id"] = matched_df["class_id"].astype(int)

        matched_df = matched_df[
            [
                "video_name",
                "video_path",
                "label",
                "class_id",
                "category",
                "class_folder",
                "split",
                "official_relpath",
                "duplicate_local_copies",
            ]
        ].sort_values(
            ["split", "class_id", "video_name"]
        ).reset_index(drop=True)

    class_map_df = pd.DataFrame({
        "class_id": list(range(len(AVAILABLE_LABELS_262))),
        "label": AVAILABLE_LABELS_262,
    })

    class_map_df.to_csv(CLASS_MAP_CSV, index=False)
    matched_df.to_csv(OUTPUT_CSV, index=False)

    # ------------------------------------------------------------
    # 6. Final safety checks
    # ------------------------------------------------------------
    split_counts = (
        matched_df["split"].value_counts().to_dict()
        if not matched_df.empty
        else {}
    )

    represented_classes = (
        matched_df["class_id"].nunique()
        if not matched_df.empty
        else 0
    )

    # Check one physical path cannot appear in two official splits.
    leakage = 0
    if not matched_df.empty:
        leakage = int(
            (
                matched_df.groupby("video_path")["split"].nunique() > 1
            ).sum()
        )

    class_split_counts = (
        matched_df.groupby(["class_id", "label", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        if not matched_df.empty
        else pd.DataFrame()
    )

    zero_train = []
    if not class_split_counts.empty:
        if "train" not in class_split_counts.columns:
            zero_train = AVAILABLE_LABELS_262
        else:
            zero_train = class_split_counts[
                class_split_counts["train"] == 0
            ]["label"].tolist()

    print("\nFINAL 262-CLASS METADATA")
    print("-" * 78)
    print(f"Rows                 : {len(matched_df)}")
    print(f"Represented classes  : {represented_classes} / 262")
    print(f"Split counts         : {split_counts}")
    print(f"Cross-split leakage  : {leakage}")
    print(f"Classes with 0 train : {len(zero_train)}")

    if zero_train:
        print("Zero-train classes:", zero_train)

    summary_lines = [
        "=" * 78,
        "INCLUDE 262-CLASS OFFICIAL-SPLIT V2 SUMMARY",
        "=" * 78,
        "",
        f"Physical videos scanned: {len(videos)}",
        f"Official paths total (263-class metadata): {len(official_df)}",
        f"Official secondnumber paths excluded: {len(second_number_df)}",
        f"Official target paths after exclusion: {len(target_official_df)}",
        "",
        f"Matched videos: {len(matched_df)}",
        f"Missing official physical videos: {len(missing_official_df)}",
        f"Unique local videos outside official split: {len(unmatched_local)}",
        f"Duplicate extracted video identities: {duplicate_keys}",
        "",
        f"Represented classes: {represented_classes} / 262",
        f"Split counts: {split_counts}",
        f"Cross-split leakage: {leakage}",
        f"Classes with zero training examples: {len(zero_train)}",
        "",
        f"Metadata: {OUTPUT_CSV}",
        f"Class mapping: {CLASS_MAP_CSV}",
        f"Duplicate report: {DUPLICATES_CSV}",
        f"Missing official videos: {MISSING_OFFICIAL_CSV}",
        f"Unmatched local videos: {UNMATCHED_LOCAL_CSV}",
        "",
        "STATUS:",
    ]

    success = (
        represented_classes == 262
        and leakage == 0
        and len(zero_train) == 0
        and len(missing_official_df) == 0
    )

    if success:
        summary_lines.append(
            "CLEAN 262-CLASS OFFICIAL-SPLIT METADATA READY ✅"
        )
        summary_lines.append(
            "NEXT: extract new 64-frame MediaPipe landmarks using TRAIN/VAL/TEST lock."
        )
    else:
        summary_lines.append(
            "METADATA BUILD COMPLETED, BUT GAPS REMAIN - REVIEW BEFORE FEATURE EXTRACTION ⚠️"
        )

    SUMMARY_TXT.write_text(
        "\n".join(summary_lines),
        encoding="utf-8"
    )

    print("\n" + "\n".join(summary_lines))
    print("=" * 78)


if __name__ == "__main__":
    main()