from pathlib import Path
import re
import urllib.request
from collections import defaultdict, Counter

import pandas as pd

ROOT = Path.cwd()
DATASET = ROOT / "datasets" / "INCLUDE"
META = ROOT / "metadata"
RESULTS = ROOT / "results"

OUT_MISSING = META / "include262_missing_diagnostic_v4.csv"
OUT_UNMATCHED = META / "include262_unmatched_local_diagnostic_v4.csv"
OUT_CLASS_COUNTS = META / "include262_class_count_comparison_v4.csv"
OUT_SUMMARY = RESULTS / "include262_mismatch_diagnostic_v4_summary.txt"

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


def file_stem_norm(filename):
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
    filename = parts[-1].strip().lower()
    return {
        "split": split,
        "official_relpath": relpath,
        "label": label,
        "filename": filename,
        "stem_norm": file_stem_norm(filename),
        "pair_key": f"{label}|{filename}",
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
    filename = parts[-1].strip().lower()

    return {
        "label": label,
        "filename": filename,
        "stem_norm": file_stem_norm(filename),
        "pair_key": f"{label}|{filename}",
        "video_path": str(video.resolve()),
        "relative_local_path": str(rel),
        "class_folder": parts[idx],
    }


def main():
    print("=" * 80)
    print("INCLUDE STEP 8 - DIAGNOSE THE 605 OFFICIAL/LOCAL MISMATCHES")
    print("=" * 80)

    META.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    official = []
    for split, url in SPLIT_URLS.items():
        print(f"Downloading official {split} split...")
        for relpath in download_entries(split, url):
            rec = official_record(relpath, split)
            if rec is not None and rec["label"] != "secondnumber":
                official.append(rec)

    official_df = pd.DataFrame(official)

    local_all = []
    for p in DATASET.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            rec = local_record(p)
            if rec is not None:
                local_all.append(rec)

    # Collapse duplicate extracted copies by class + filename.
    grouped = defaultdict(list)
    for row in local_all:
        grouped[row["pair_key"]].append(row)

    local_unique = []
    for key, rows in grouped.items():
        rows = sorted(
            rows,
            key=lambda r: (len(r["relative_local_path"]), r["relative_local_path"].lower())
        )
        local_unique.append(rows[0])

    local_df = pd.DataFrame(local_unique)

    official_keys = set(official_df["pair_key"])
    local_keys = set(local_df["pair_key"])

    missing_off = official_df[~official_df["pair_key"].isin(local_keys)].copy()
    unmatched_local = local_df[~local_df["pair_key"].isin(official_keys)].copy()

    # Lookup maps independent of class label.
    local_by_filename = defaultdict(list)
    local_by_stem = defaultdict(list)
    for _, row in local_df.iterrows():
        local_by_filename[row["filename"]].append(row.to_dict())
        local_by_stem[row["stem_norm"]].append(row.to_dict())

    off_by_filename = defaultdict(list)
    off_by_stem = defaultdict(list)
    for _, row in official_df.iterrows():
        off_by_filename[row["filename"]].append(row.to_dict())
        off_by_stem[row["stem_norm"]].append(row.to_dict())

    missing_diag = []
    category_counts = Counter()

    for _, row in missing_off.iterrows():
        exact = local_by_filename.get(row["filename"], [])
        stem = local_by_stem.get(row["stem_norm"], [])

        exact_other = [x for x in exact if x["label"] != row["label"]]
        stem_other = [x for x in stem if x["label"] != row["label"]]

        if exact:
            status = "filename_exists_under_different_label"
            category_counts[status] += 1
        elif stem:
            status = "same_stem_but_extension_or_filename_format_diff"
            category_counts[status] += 1
        else:
            status = "filename_not_present_locally"
            category_counts[status] += 1

        missing_diag.append({
            **row.to_dict(),
            "diagnosis": status,
            "local_exact_candidate_count": len(exact),
            "local_exact_labels": ";".join(sorted({x["label"] for x in exact})),
            "local_exact_paths": " || ".join(x["relative_local_path"] for x in exact[:8]),
            "local_stem_candidate_count": len(stem),
            "local_stem_labels": ";".join(sorted({x["label"] for x in stem})),
            "local_stem_paths": " || ".join(x["relative_local_path"] for x in stem[:8]),
        })

    unmatched_diag = []
    for _, row in unmatched_local.iterrows():
        exact = off_by_filename.get(row["filename"], [])
        stem = off_by_stem.get(row["stem_norm"], [])

        if exact:
            status = "filename_exists_officially_under_different_label"
        elif stem:
            status = "same_stem_but_extension_or_filename_format_diff"
        else:
            status = "local_filename_not_in_official_metadata"

        unmatched_diag.append({
            **row.to_dict(),
            "diagnosis": status,
            "official_exact_candidate_count": len(exact),
            "official_exact_labels": ";".join(sorted({x["label"] for x in exact})),
            "official_exact_paths": " || ".join(x["official_relpath"] for x in exact[:8]),
            "official_stem_candidate_count": len(stem),
            "official_stem_labels": ";".join(sorted({x["label"] for x in stem})),
            "official_stem_paths": " || ".join(x["official_relpath"] for x in stem[:8]),
        })

    missing_diag_df = pd.DataFrame(missing_diag)
    unmatched_diag_df = pd.DataFrame(unmatched_diag)

    missing_diag_df.to_csv(OUT_MISSING, index=False)
    unmatched_diag_df.to_csv(OUT_UNMATCHED, index=False)

    # Per-class count comparison.
    off_counts = official_df.groupby("label").size().rename("official_count")
    loc_counts = local_df.groupby("label").size().rename("local_unique_count")
    matched_counts = (
        official_df[official_df["pair_key"].isin(local_keys)]
        .groupby("label").size().rename("matched_count")
    )

    class_counts = (
        pd.concat([off_counts, loc_counts, matched_counts], axis=1)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    class_counts["count_difference_local_minus_official"] = (
        class_counts["local_unique_count"] - class_counts["official_count"]
    )
    class_counts["missing_official_in_class"] = (
        class_counts["official_count"] - class_counts["matched_count"]
    )
    class_counts["unmatched_local_in_class"] = (
        class_counts["local_unique_count"] - class_counts["matched_count"]
    )
    class_counts.to_csv(OUT_CLASS_COUNTS, index=False)

    same_count_classes = int(
        (class_counts["official_count"] == class_counts["local_unique_count"]).sum()
    )
    differing_count_classes = len(class_counts) - same_count_classes

    print("\n" + "-" * 80)
    print("DIAGNOSTIC RESULT")
    print("-" * 80)
    print(f"Official target unique videos : {len(official_df)}")
    print(f"Local unique videos           : {len(local_df)}")
    print(f"Matched class+filename        : {len(official_df) - len(missing_off)}")
    print(f"Missing official identities   : {len(missing_off)}")
    print(f"Unmatched local identities    : {len(unmatched_local)}")
    print()
    print("Missing-official diagnosis:")
    for k, v in category_counts.items():
        print(f"  {k}: {v}")
    print()
    print(f"Classes with same video count : {same_count_classes} / 262")
    print(f"Classes with different count  : {differing_count_classes} / 262")

    print("\nTop 15 classes with most mismatches:")
    top = class_counts.sort_values(
        ["missing_official_in_class", "unmatched_local_in_class"],
        ascending=False
    ).head(15)
    print(
        top[
            [
                "label",
                "official_count",
                "local_unique_count",
                "matched_count",
                "missing_official_in_class",
                "unmatched_local_in_class",
            ]
        ].to_string(index=False)
    )

    summary_lines = [
        "=" * 80,
        "INCLUDE 262 MISMATCH DIAGNOSTIC V4 SUMMARY",
        "=" * 80,
        "",
        f"Official target unique videos: {len(official_df)}",
        f"Local unique videos: {len(local_df)}",
        f"Matched class+filename: {len(official_df) - len(missing_off)}",
        f"Missing official identities: {len(missing_off)}",
        f"Unmatched local identities: {len(unmatched_local)}",
        "",
        "Missing-official diagnosis:",
    ]
    for k, v in category_counts.items():
        summary_lines.append(f"  {k}: {v}")
    summary_lines += [
        "",
        f"Classes with same video count: {same_count_classes} / 262",
        f"Classes with different count: {differing_count_classes} / 262",
        "",
        f"Missing diagnostics: {OUT_MISSING}",
        f"Unmatched-local diagnostics: {OUT_UNMATCHED}",
        f"Class-count comparison: {OUT_CLASS_COUNTS}",
        "",
        "NEXT DECISION:",
        "If most missing filenames exist locally under another label -> fix class mapping.",
        "If most missing filenames do not exist locally -> local ZIP release and GitHub split metadata differ.",
        "Do not extract features until this is resolved.",
    ]

    OUT_SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n" + "\n".join(summary_lines))
    print("=" * 80)


if __name__ == "__main__":
    main()