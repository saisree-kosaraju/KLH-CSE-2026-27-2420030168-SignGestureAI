from pathlib import Path
import re
import urllib.request
from collections import defaultdict

import pandas as pd

ROOT = Path.cwd()
DATASET = ROOT / "datasets" / "INCLUDE"
META = ROOT / "metadata"
RESULTS = ROOT / "results"

OUTPUT = META / "include262_official_split_v3.csv"
CLASS_MAP = META / "include262_class_mapping_v3.csv"
CONFLICTS = META / "include262_filename_class_conflicts_v3.csv"
MISSING = META / "include262_missing_official_v3.csv"
SUMMARY = RESULTS / "include262_official_split_v3_summary.txt"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}

OFFICIAL_LABELS = [
"actor","adult","afternoon","alive","alright","animal","artist","attack","author","baby",
"bad","bag","ball","bank","bathroom","beautiful","bed","bedroom","bicycle","biglarge",
"bill","bird","black","blind","blue","boat","book","box","boy","brother","brown","bus",
"camera","car","card","cat","cellphone","chair","cheap","child","city","clean","clock",
"clothing","cold","colour","computer","cool","court","cow","crowd","curved","daughter",
"dead","deaf","death","deep","dirty","doctor","dog","door","dream","dress","dry",
"election","energy","evening","exercise","exmonsoon","expensive","fall","family","famous",
"fan","fast","father","female","fish","flat","friday","friend","gift","girl","god","good",
"goodafternoon","goodevening","goodmorning","goodnight","grandfather","grandmother",
"green","grey","ground","gun","happy","hard","hat","he","healthy","heavy","hello","high",
"horse","hospital","hot","hour","house","howareyou","husband","i","india","it","job",
"key","king","kitchen","lamp","laptop","lawyer","letter","library","light","location",
"lock","long","loose","loud","low","male","man","manager","market","marriage","mean",
"medicine","minute","monday","money","month","morning","mother","mouse","narrow",
"neighbour","new","newspaper","nice","night","office","old","orange","page","paint",
"pant","paper","parent","park","patient","peace","pen","pencil","photograph","pink",
"plane","player","pleased","pocket","police","poor","president","price","priest","queen",
"quiet","raceethnicity","radio","red","religion","reporter","restaurant","rich","ring",
"sad","saturday","school","science","screen","season","second","secondnumber","secretary",
"shallow","she","shirt","shoes","short","sick","sign","sister","skirt","slow",
"smalllittle","soap","soft","soldier","son","sport","spring","storeorshop","streetorroad",
"strong","student","suit","summer","sunday","table","tall","teacher","team","technology",
"telephone","television","temple","thankyou","they","thick","thin","thursday","tight",
"time","today","tomorrow","tool","train","trainstation","trainticket","transportation",
"truck","tshirt","tuesday","ugly","university","waiter","war","warm","we","weak",
"wednesday","week","wet","white","wide","wife","window","winter","woman","year","yellow",
"yesterday","you","young","youplural"
]
AVAILABLE = [x for x in OFFICIAL_LABELS if x != "secondnumber"]

SPLIT_URLS = {
    "train": "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/include_train.txt",
    "val": "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/include_val.txt",
    "test": "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/include_test.txt",
}


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def clean_class(folder):
    s = re.sub(r"^\s*\d+\s*[\.\-_:)]*\s*", "", str(folder).strip())
    return norm(s)


def parse_split_text(text):
    text = text.replace("\r", " ").replace("\n", " ")
    pattern = re.compile(r"(?i)(\.mov|\.mp4|\.avi|\.mkv|\.webm)\s+")
    text = pattern.sub(lambda m: m.group(1) + "\n", text)
    return [x.strip() for x in text.splitlines() if x.strip()]


def download_entries(split, url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        text = r.read().decode("utf-8")
    return parse_split_text(text)


def official_record(relpath, split):
    parts = [p for p in relpath.replace("\\", "/").split("/") if p]
    if len(parts) < 3:
        return None
    label = clean_class(parts[1])
    filename = parts[-1].strip().lower()
    return {
        "split": split,
        "official_relpath": relpath,
        "label": label,
        "filename": filename,
        "pair_key": f"{label}|{filename}",
    }


def find_class_ancestor(video):
    rel = video.relative_to(DATASET)
    parts = rel.parts

    matches = []
    for idx in range(len(parts) - 1):
        label = clean_class(parts[idx])
        if label in OFFICIAL_LABELS:
            matches.append((idx, label))

    if not matches:
        return None

    idx, label = matches[-1]
    return idx, label, parts


def local_record(video):
    info = find_class_ancestor(video)
    if info is None:
        return None

    idx, label, parts = info
    filename = parts[-1].strip().lower()

    return {
        "label": label,
        "filename": filename,
        "pair_key": f"{label}|{filename}",
        "video_path": str(video.resolve()),
        "relative_local_path": str(video.relative_to(DATASET)),
        "class_folder": parts[idx],
        "category": parts[0],
    }


def main():
    print("=" * 78)
    print("INCLUDE STEP 7 - FIX OFFICIAL SPLIT MATCHING BY CLASS + FILENAME")
    print("=" * 78)

    META.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Official metadata
    official = []
    for split, url in SPLIT_URLS.items():
        print(f"Downloading official {split} split...")
        for relpath in download_entries(split, url):
            rec = official_record(relpath, split)
            if rec is not None:
                official.append(rec)

    official_df = pd.DataFrame(official)
    official_df = official_df[official_df["label"] != "secondnumber"].copy()

    print(f"\nOfficial target videos after excluding secondnumber: {len(official_df)}")
    print(f"Official target classes: {official_df['label'].nunique()}")

    # Local physical videos
    videos = [
        p for p in DATASET.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]

    local = []
    for p in videos:
        rec = local_record(p)
        if rec is not None:
            local.append(rec)

    local_df = pd.DataFrame(local)

    print(f"Physical local videos scanned: {len(videos)}")
    print(f"Local videos with recognized class: {len(local_df)}")
    print(f"Local classes: {local_df['label'].nunique()}")

    # Group local copies by class+filename.
    grouped = defaultdict(list)
    for row in local:
        grouped[row["pair_key"]].append(row)

    selected = {}
    duplicate_copy_count = 0

    for key, rows in grouped.items():
        rows = sorted(
            rows,
            key=lambda r: (len(r["relative_local_path"]), r["relative_local_path"].lower())
        )
        selected[key] = rows[0]
        duplicate_copy_count += max(0, len(rows) - 1)

    # Check official pair-key uniqueness. If the same class+filename occurs in two
    # official entries, we must not silently assign it.
    official_key_counts = official_df["pair_key"].value_counts()
    conflict_keys = set(official_key_counts[official_key_counts > 1].index)

    conflicts = official_df[official_df["pair_key"].isin(conflict_keys)].copy()
    conflicts.to_csv(CONFLICTS, index=False)

    print(f"Official class+filename conflict keys: {len(conflict_keys)}")
    print(f"Duplicate extracted local copies discarded: {duplicate_copy_count}")

    rows = []
    missing_rows = []

    for _, off in official_df.iterrows():
        key = off["pair_key"]

        if key in conflict_keys:
            missing_rows.append({
                **off.to_dict(),
                "reason": "official class+filename key is not unique",
            })
            continue

        local_rec = selected.get(key)
        if local_rec is None:
            missing_rows.append({
                **off.to_dict(),
                "reason": "no local physical video with same class+filename",
            })
            continue

        rows.append({
            "video_name": local_rec["filename"],
            "video_path": local_rec["video_path"],
            "label": off["label"],
            "category": local_rec["category"],
            "class_folder": local_rec["class_folder"],
            "split": off["split"],
            "official_relpath": off["official_relpath"],
            "duplicate_local_copies": len(grouped[key]),
        })

    out = pd.DataFrame(rows)
    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(MISSING, index=False)

    class_to_id = {label: i for i, label in enumerate(AVAILABLE)}
    out["class_id"] = out["label"].map(class_to_id).astype(int)

    out = out[
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
    ].sort_values(["split", "class_id", "video_name"]).reset_index(drop=True)

    out.to_csv(OUTPUT, index=False)

    pd.DataFrame({
        "class_id": list(range(len(AVAILABLE))),
        "label": AVAILABLE,
    }).to_csv(CLASS_MAP, index=False)

    represented = out["class_id"].nunique()
    split_counts = out["split"].value_counts().to_dict()
    leakage = int((out.groupby("video_path")["split"].nunique() > 1).sum())

    # Check each class has train examples.
    train_counts = out[out["split"] == "train"]["class_id"].value_counts()
    zero_train = [i for i in range(262) if train_counts.get(i, 0) == 0]

    print("\n" + "-" * 78)
    print("FINAL RESULTS")
    print("-" * 78)
    print(f"Matched videos          : {len(out)} / {len(official_df)}")
    print(f"Missing official videos : {len(missing_df)}")
    print(f"Represented classes     : {represented} / 262")
    print(f"Split counts            : {split_counts}")
    print(f"Cross-split leakage     : {leakage}")
    print(f"Classes with 0 train    : {len(zero_train)}")

    summary = [
        "=" * 78,
        "INCLUDE 262 OFFICIAL-SPLIT V3 SUMMARY",
        "=" * 78,
        "",
        f"Official target videos: {len(official_df)}",
        f"Matched physical videos: {len(out)}",
        f"Missing official videos: {len(missing_df)}",
        f"Official class+filename conflict keys: {len(conflict_keys)}",
        f"Duplicate local copies discarded: {duplicate_copy_count}",
        "",
        f"Represented classes: {represented} / 262",
        f"Split counts: {split_counts}",
        f"Cross-split leakage: {leakage}",
        f"Classes with zero training examples: {len(zero_train)}",
        "",
        f"Metadata: {OUTPUT}",
        f"Class mapping: {CLASS_MAP}",
        f"Conflicts: {CONFLICTS}",
        f"Missing official videos: {MISSING}",
        "",
        "STATUS:",
    ]

    success = (
        len(out) == len(official_df)
        and len(missing_df) == 0
        and len(conflict_keys) == 0
        and represented == 262
        and leakage == 0
        and len(zero_train) == 0
    )

    if success:
        summary.append("CLEAN 262-CLASS OFFICIAL SPLIT READY ✅")
        summary.append("NEXT: 64-frame MediaPipe hand + pose + face extraction.")
    else:
        summary.append("MATCHING STILL HAS GAPS - REVIEW BEFORE FEATURE EXTRACTION ⚠️")

    SUMMARY.write_text("\n".join(summary), encoding="utf-8")

    print("\n" + "\n".join(summary))
    print("=" * 78)


if __name__ == "__main__":
    main()