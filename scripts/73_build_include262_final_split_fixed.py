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
MISSING = META / "include262_missing_final.csv"
CONFLICTS = META / "include262_match_conflicts_final.csv"
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


def norm_label(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def clean_class(folder):
    s = re.sub(r"^\s*\d+\s*[\.\-_:)]*\s*", "", str(folder).strip())
    return norm_label(s)


def raw_stem(filename):
    # IMPORTANT:
    # Remove extension only. Preserve underscores and internal punctuation.
    # This keeps distinct files such as:
    # MVI_005_6 vs MVI_0056
    # MVI_0057_ vs MVI_0057
    # MVI_0060_ vs MVI_0060
    return Path(str(filename).strip()).stem.casefold()


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
        "stem": raw_stem(filename),
        "match_key": f"{label}|{raw_stem(filename)}",
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
    filename = parts[-1]

    return {
        "label": label,
        "local_filename": filename,
        "stem": raw_stem(filename),
        "match_key": f"{label}|{raw_stem(filename)}",
        "video_path": str(video.resolve()),
        "relative_local_path": str(rel),
        "category": parts[0],
        "class_folder": parts[idx],
    }


def main():
    print("=" * 84)
    print("INCLUDE STEP 11 - FINAL 262-CLASS OFFICIAL SPLIT")
    print("MATCH RULE: CLASS + RAW FILENAME STEM (EXTENSION REMOVED ONLY)")
    print("=" * 84)

    META.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Official split metadata
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
    # 2. Local videos
    # ------------------------------------------------------------
    physical_videos = [
        p for p in DATASET.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]

    local = []
    unrecognized = []

    for p in physical_videos:
        rec = local_record(p)
        if rec is None:
            unrecognized.append(str(p))
        else:
            local.append(rec)

    print(f"Local physical videos  : {len(physical_videos)}")
    print(f"Local recognized videos: {len(local)}")
    print(f"Unrecognized videos    : {len(unrecognized)}")

    # ------------------------------------------------------------
    # 3. Group by class + raw stem
    # ------------------------------------------------------------
    official_groups = defaultdict(list)
    for row in official:
        official_groups[row["match_key"]].append(row)

    local_groups = defaultdict(list)
    for row in local:
        local_groups[row["match_key"]].append(row)

    # Official keys must be unique now.
    official_conflicts = {
        key: rows
        for key, rows in official_groups.items()
        if len(rows) > 1
    }

    # Local repeats can be duplicate extraction copies. They are safe only if
    # they all have the same filename stem identity; select one physical copy.
    selected_local = {}
    duplicate_copies_removed = 0
    local_suspicious = {}

    for key, rows in local_groups.items():
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                len(r["relative_local_path"]),
                r["relative_local_path"].casefold(),
            ),
        )

        # If multiple files share the key, they should be duplicate copies
        # of the same logical video after extraction.
        selected_local[key] = rows_sorted[0]
        duplicate_copies_removed += max(0, len(rows_sorted) - 1)

        # Record details only; duplicates themselves are expected.
        if len(rows_sorted) > 1:
            labels = {r["label"] for r in rows_sorted}
            stems = {r["stem"] for r in rows_sorted}
            if len(labels) != 1 or len(stems) != 1:
                local_suspicious[key] = rows_sorted

    conflict_rows = []

    for key, rows in official_conflicts.items():
        conflict_rows.append({
            "source": "official",
            "match_key": key,
            "count": len(rows),
            "details": " || ".join(
                f"[{r['split']}] {r['official_relpath']}"
                for r in rows
            ),
        })

    for key, rows in local_suspicious.items():
        conflict_rows.append({
            "source": "local",
            "match_key": key,
            "count": len(rows),
            "details": " || ".join(r["relative_local_path"] for r in rows),
        })

    pd.DataFrame(
        conflict_rows,
        columns=["source", "match_key", "count", "details"]
    ).to_csv(CONFLICTS, index=False)

    print(f"Official match conflicts: {len(official_conflicts)}")
    print(f"Suspicious local conflicts: {len(local_suspicious)}")
    print(f"Unique local identities: {len(selected_local)}")
    print(f"Duplicate copies removed: {duplicate_copies_removed}")

    if official_conflicts:
        raise RuntimeError(
            "Official conflicts still exist. Do not continue."
        )

    # ------------------------------------------------------------
    # 4. Match every official entry
    # ------------------------------------------------------------
    matched = []
    missing = []

    for row in official:
        loc = selected_local.get(row["match_key"])

        if loc is None:
            missing.append(row)
            continue

        matched.append({
            "video_name": loc["local_filename"],
            "video_path": loc["video_path"],
            "label": row["label"],
            "category": loc["category"],
            "class_folder": loc["class_folder"],
            "split": row["split"],
            "official_relpath": row["official_relpath"],
            "official_filename": row["official_filename"],
            "local_filename": loc["local_filename"],
            "raw_stem": row["stem"],
            "duplicate_local_copies": len(local_groups[row["match_key"]]),
        })

    missing_df = pd.DataFrame(missing)
    missing_df.to_csv(MISSING, index=False)

    out = pd.DataFrame(matched)

    # ------------------------------------------------------------
    # 5. Class IDs
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
                "raw_stem",
                "duplicate_local_copies",
            ]
        ].sort_values(
            ["split", "class_id", "raw_stem"]
        ).reset_index(drop=True)

    pd.DataFrame({
        "class_id": list(range(262)),
        "label": AVAILABLE,
    }).to_csv(CLASS_MAP, index=False)

    out.to_csv(OUTPUT, index=False)

    # ------------------------------------------------------------
    # 6. Final validation
    # ------------------------------------------------------------
    represented = out["class_id"].nunique() if not out.empty else 0
    split_counts = out["split"].value_counts().to_dict() if not out.empty else {}
    expected_counts = official_df["split"].value_counts().to_dict()

    leakage = (
        int((out.groupby("video_path")["split"].nunique() > 1).sum())
        if not out.empty else 0
    )

    duplicated_final_paths = (
        int(out["video_path"].duplicated().sum())
        if not out.empty else 0
    )

    train_class_ids = set(
        out.loc[out["split"] == "train", "class_id"].unique()
    ) if not out.empty else set()

    zero_train = [i for i in range(262) if i not in train_class_ids]

    missing_count = len(missing_df)

    print("\n" + "=" * 84)
    print("FINAL VALIDATION")
    print("=" * 84)
    print(f"Matched videos          : {len(out)} / {len(official_df)}")
    print(f"Missing videos          : {missing_count}")
    print(f"Represented classes     : {represented} / 262")
    print(f"Split counts            : {split_counts}")
    print(f"Expected split counts   : {expected_counts}")
    print(f"Cross-split leakage     : {leakage}")
    print(f"Duplicate final paths   : {duplicated_final_paths}")
    print(f"Classes with 0 train    : {len(zero_train)}")

    success = (
        len(out) == 4284
        and len(out) == len(official_df)
        and missing_count == 0
        and represented == 262
        and split_counts == expected_counts
        and leakage == 0
        and duplicated_final_paths == 0
        and len(zero_train) == 0
        and len(official_conflicts) == 0
        and len(local_suspicious) == 0
    )

    summary_lines = [
        "=" * 84,
        "INCLUDE 262-CLASS OFFICIAL SPLIT - FINAL SUMMARY",
        "=" * 84,
        "",
        "Match rule:",
        "  official class + raw filename stem",
        "  extension ignored",
        "  underscores/punctuation preserved",
        "",
        "This safely distinguishes:",
        "  MVI_005_6  vs MVI_0056",
        "  MVI_0057_  vs MVI_0057",
        "  MVI_0060_  vs MVI_0060",
        "",
        f"Official target videos: {len(official_df)}",
        f"Matched videos: {len(out)}",
        f"Missing videos: {missing_count}",
        f"Local physical videos: {len(physical_videos)}",
        f"Unique local identities: {len(selected_local)}",
        f"Duplicate extracted copies removed: {duplicate_copies_removed}",
        "",
        f"Represented classes: {represented} / 262",
        f"Split counts: {split_counts}",
        f"Expected split counts: {expected_counts}",
        f"Cross-split leakage: {leakage}",
        f"Duplicate final paths: {duplicated_final_paths}",
        f"Classes with zero training examples: {len(zero_train)}",
        "",
        f"Metadata: {OUTPUT}",
        f"Class mapping: {CLASS_MAP}",
        f"Missing report: {MISSING}",
        f"Conflict report: {CONFLICTS}",
        "",
        "STATUS:",
    ]

    if success:
        summary_lines.append("CLEAN 262-CLASS OFFICIAL-SPLIT DATASET READY ✅")
        summary_lines.append(
            "All 4284 available official identities were matched correctly."
        )
        summary_lines.append(
            "NEXT: video integrity verification, then 64-frame MediaPipe extraction."
        )
    else:
        summary_lines.append("FINAL VALIDATION FAILED ⚠️")
        summary_lines.append("Do not start feature extraction yet.")

    SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "\n".join(summary_lines))
    print("=" * 84)

    if not success:
        raise RuntimeError(
            "Final validation failed. Review the printed summary."
        )


if __name__ == "__main__":
    main()