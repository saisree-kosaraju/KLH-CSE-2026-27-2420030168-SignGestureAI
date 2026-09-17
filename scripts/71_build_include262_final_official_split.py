from pathlib import Path
import re
import urllib.request
from collections import defaultdict

import pandas as pd

ROOT = Path.cwd()
DATASET = ROOT / "datasets" / "INCLUDE"
META = ROOT / "metadata"
RESULTS = ROOT / "results"

OUTPUT = META / "include262_official_split_final.csv"
CLASS_MAP = META / "include262_class_mapping_final.csv"
CONFLICTS = META / "include262_stem_conflicts_final.csv"
SUMMARY = RESULTS / "include262_official_split_final_summary.txt"

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


def stem_norm(filename):
    return norm(Path(str(filename)).stem)


def parse_split_text(text):
    text = text.replace("\r", " ").replace("\n", " ")
    pat = re.compile(r"(?i)(\.mov|\.mp4|\.avi|\.mkv|\.webm)\s+")
    text = pat.sub(lambda m: m.group(1) + "\n", text)
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
    filename = parts[-1].strip()

    return {
        "split": split,
        "official_relpath": relpath,
        "label": label,
        "official_filename": filename,
        "stem_norm": stem_norm(filename),
        "match_key": f"{label}|{stem_norm(filename)}",
    }


def local_record(video):
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

    return {
        "label": label,
        "local_filename": parts[-1],
        "stem_norm": stem_norm(parts[-1]),
        "match_key": f"{label}|{stem_norm(parts[-1])}",
        "video_path": str(video.resolve()),
        "relative_local_path": str(rel),
        "category": parts[0],
        "class_folder": parts[idx],
    }


def main():
    print("=" * 80)
    print("INCLUDE STEP 9 - BUILD FINAL 262-CLASS OFFICIAL SPLIT")
    print("MATCHING BY CLASS + NORMALIZED VIDEO STEM")
    print("=" * 80)

    META.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Official split
    # ------------------------------------------------------------
    official = []

    for split, url in SPLIT_URLS.items():
        print(f"Downloading official {split} split...")
        for relpath in download_entries(split, url):
            rec = official_record(relpath, split)
            if rec is not None and rec["label"] != "secondnumber":
                official.append(rec)

    official_df = pd.DataFrame(official)

    print(f"\nOfficial target videos : {len(official_df)}")
    print(f"Official target classes: {official_df['label'].nunique()}")

    # ------------------------------------------------------------
    # 2. Local physical dataset
    # ------------------------------------------------------------
    videos = [
        p for p in DATASET.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]

    local = []
    for p in videos:
        rec = local_record(p)
        if rec is not None:
            local.append(rec)

    local_df_all = pd.DataFrame(local)

    print(f"Local physical videos  : {len(videos)}")
    print(f"Local recognized videos: {len(local_df_all)}")
    print(f"Local classes          : {local_df_all['label'].nunique()}")

    # ------------------------------------------------------------
    # 3. Collapse duplicate extracted copies by class + stem
    # ------------------------------------------------------------
    grouped = defaultdict(list)
    for row in local:
        grouped[row["match_key"]].append(row)

    local_unique = {}
    duplicate_copies_discarded = 0

    for key, rows in grouped.items():
        rows = sorted(
            rows,
            key=lambda r: (
                len(r["relative_local_path"]),
                r["relative_local_path"].lower(),
            ),
        )
        local_unique[key] = rows[0]
        duplicate_copies_discarded += max(0, len(rows) - 1)

    print(f"Unique local identities: {len(local_unique)}")
    print(f"Duplicate copies removed: {duplicate_copies_discarded}")

    # ------------------------------------------------------------
    # 4. Collision safety checks
    # ------------------------------------------------------------
    official_counts = official_df["match_key"].value_counts()
    official_conflict_keys = set(
        official_counts[official_counts > 1].index
    )

    local_conflict_keys = {
        key for key, rows in grouped.items()
        if len({r["video_path"] for r in rows}) > 1
        and len({r["relative_local_path"] for r in rows}) > 1
        and len({r["local_filename"].lower() for r in rows}) > 1
    }

    conflict_rows = []

    for key in sorted(official_conflict_keys):
        conflict_rows.append({
            "source": "official",
            "match_key": key,
            "details": "same class+stem occurs more than once in official split",
        })

    # Local duplicate extracted copies are expected. A local conflict is only
    # suspicious if the same class+stem has genuinely different filenames.
    for key in sorted(local_conflict_keys):
        filenames = sorted({r["local_filename"] for r in grouped[key]})
        conflict_rows.append({
            "source": "local",
            "match_key": key,
            "details": " | ".join(filenames),
        })

    pd.DataFrame(conflict_rows).to_csv(CONFLICTS, index=False)

    print(f"Official stem conflicts: {len(official_conflict_keys)}")
    print(f"Suspicious local stem conflicts: {len(local_conflict_keys)}")

    if official_conflict_keys:
        raise RuntimeError(
            "Official class+stem collisions found. Review conflict CSV before continuing."
        )

    # ------------------------------------------------------------
    # 5. Match all official videos to local physical videos
    # ------------------------------------------------------------
    matched = []
    missing = []

    for _, off in official_df.iterrows():
        key = off["match_key"]
        loc = local_unique.get(key)

        if loc is None:
            missing.append(off.to_dict())
            continue

        matched.append({
            "video_name": loc["local_filename"],
            "video_path": loc["video_path"],
            "label": off["label"],
            "category": loc["category"],
            "class_folder": loc["class_folder"],
            "split": off["split"],
            "official_relpath": off["official_relpath"],
            "official_filename": off["official_filename"],
            "local_filename": loc["local_filename"],
            "stem_norm": off["stem_norm"],
            "duplicate_local_copies": len(grouped[key]),
        })

    out = pd.DataFrame(matched)

    # ------------------------------------------------------------
    # 6. Stable 262-class mapping
    # ------------------------------------------------------------
    class_to_id = {label: i for i, label in enumerate(AVAILABLE)}

    if not out.empty:
        out["class_id"] = out["label"].map(class_to_id)

        if out["class_id"].isna().any():
            bad = sorted(out.loc[out["class_id"].isna(), "label"].unique())
            raise RuntimeError(f"Unmapped labels: {bad}")

        out["class_id"] = out["class_id"].astype(int)

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
                "official_filename",
                "local_filename",
                "stem_norm",
                "duplicate_local_copies",
            ]
        ].sort_values(
            ["split", "class_id", "stem_norm"]
        ).reset_index(drop=True)

    class_map_df = pd.DataFrame({
        "class_id": list(range(262)),
        "label": AVAILABLE,
    })

    out.to_csv(OUTPUT, index=False)
    class_map_df.to_csv(CLASS_MAP, index=False)

    # ------------------------------------------------------------
    # 7. Final checks
    # ------------------------------------------------------------
    represented_classes = out["class_id"].nunique() if not out.empty else 0
    split_counts = out["split"].value_counts().to_dict() if not out.empty else {}

    leakage = (
        int((out.groupby("video_path")["split"].nunique() > 1).sum())
        if not out.empty else 0
    )

    duplicate_final_paths = (
        int(out["video_path"].duplicated().sum())
        if not out.empty else 0
    )

    train_ids = set(
        out.loc[out["split"] == "train", "class_id"].unique()
    ) if not out.empty else set()

    zero_train = [
        i for i in range(262)
        if i not in train_ids
    ]

    expected_split_counts = (
        official_df["split"].value_counts().to_dict()
    )

    # Ensure every official identity was matched.
    missing_count = len(missing)

    print("\n" + "=" * 80)
    print("FINAL VALIDATION")
    print("=" * 80)
    print(f"Matched videos          : {len(out)} / {len(official_df)}")
    print(f"Missing videos          : {missing_count}")
    print(f"Represented classes     : {represented_classes} / 262")
    print(f"Split counts            : {split_counts}")
    print(f"Expected split counts   : {expected_split_counts}")
    print(f"Cross-split leakage     : {leakage}")
    print(f"Duplicate final paths   : {duplicate_final_paths}")
    print(f"Classes with 0 train    : {len(zero_train)}")

    success = (
        len(out) == len(official_df) == 4284
        and missing_count == 0
        and represented_classes == 262
        and split_counts == expected_split_counts
        and leakage == 0
        and duplicate_final_paths == 0
        and len(zero_train) == 0
        and len(official_conflict_keys) == 0
    )

    summary_lines = [
        "=" * 80,
        "INCLUDE 262-CLASS OFFICIAL SPLIT - FINAL SUMMARY",
        "=" * 80,
        "",
        "Matching rule: official class label + normalized filename stem",
        "Reason: 605 Zenodo/GitHub entries differ only by extension/filename formatting.",
        "",
        f"Official target videos: {len(official_df)}",
        f"Matched videos: {len(out)}",
        f"Missing videos: {missing_count}",
        f"Physical videos scanned: {len(videos)}",
        f"Unique local identities: {len(local_unique)}",
        f"Duplicate extracted copies removed: {duplicate_copies_discarded}",
        "",
        f"Represented classes: {represented_classes} / 262",
        f"Split counts: {split_counts}",
        f"Expected split counts: {expected_split_counts}",
        f"Cross-split leakage: {leakage}",
        f"Duplicate final paths: {duplicate_final_paths}",
        f"Classes with zero train examples: {len(zero_train)}",
        "",
        f"Metadata: {OUTPUT}",
        f"Class mapping: {CLASS_MAP}",
        f"Conflict report: {CONFLICTS}",
        "",
        "STATUS:",
    ]

    if success:
        summary_lines.append("CLEAN 262-CLASS OFFICIAL-SPLIT DATASET READY ✅")
        summary_lines.append(
            "The 605 previous mismatches were naming/extension differences, not missing videos."
        )
        summary_lines.append(
            "NEXT: run video integrity verification, then extract 64-frame MediaPipe features."
        )
    else:
        summary_lines.append("FINAL VALIDATION FAILED ⚠️")
        summary_lines.append("Do not start feature extraction yet.")

    SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "\n".join(summary_lines))
    print("=" * 80)

    if not success:
        raise RuntimeError(
            "Final INCLUDE metadata validation failed. Review the summary above."
        )


if __name__ == "__main__":
    main()