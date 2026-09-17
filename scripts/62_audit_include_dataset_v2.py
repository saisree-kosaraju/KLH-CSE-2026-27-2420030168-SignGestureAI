from pathlib import Path
import argparse
import pandas as pd

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}

PROJECT_ROOT = Path.cwd()
METADATA_DIR = PROJECT_ROOT / "metadata"
RESULTS_DIR = PROJECT_ROOT / "results"

SOURCE_METADATA = METADATA_DIR / "include_metadata.csv"
READY_METADATA = METADATA_DIR / "include_ready.csv"
FINAL_METADATA = METADATA_DIR / "include_final.csv"

CLASS_REPORT = METADATA_DIR / "include_audit_class_report_v2.csv"
MISSING_ROWS_REPORT = METADATA_DIR / "include_audit_missing_rows_v2.csv"
SUMMARY_REPORT = RESULTS_DIR / "include_audit_v2_summary.txt"


def normalize_text_series(series):
    return series.astype(str).str.strip()


def choose_column(df, candidates, required=False, purpose="column"):
    lower_map = {str(c).lower().strip(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for col in df.columns:
        low = str(col).lower()
        for cand in candidates:
            if cand.lower() in low:
                return col

    if required:
        raise RuntimeError(
            f"Could not find {purpose}. Available columns:\n{list(df.columns)}"
        )
    return None


def detect_video_column(df):
    return choose_column(
        df,
        [
            "video_path",
            "local_video_path",
            "filepath",
            "file_path",
            "path",
            "video",
            "filename",
            "file_name",
        ],
    )


def detect_class_id_column(df):
    return choose_column(
        df,
        [
            "class_id",
            "label_id",
            "label",
            "target",
            "class",
            "category_id",
        ],
    )


def detect_gloss_column(df):
    return choose_column(
        df,
        [
            "gloss",
            "word",
            "sign",
            "class_name",
            "label_name",
            "category",
        ],
    )


def detect_split_column(df):
    return choose_column(
        df,
        ["split", "subset", "partition", "set"],
    )


def resolve_video_path(raw_value):
    if pd.isna(raw_value):
        return None

    raw = str(raw_value).strip()
    if not raw or raw.lower() == "nan":
        return None

    p = Path(raw)

    if p.is_absolute():
        return p

    candidates = [
        PROJECT_ROOT / p,
        PROJECT_ROOT / "datasets" / p,
        PROJECT_ROOT / "datasets" / "INCLUDE" / p,
        PROJECT_ROOT / "INCLUDE" / p,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return PROJECT_ROOT / p


def find_include_dataset_dirs():
    candidates = [
        PROJECT_ROOT / "datasets" / "INCLUDE",
        PROJECT_ROOT / "datasets" / "include",
        PROJECT_ROOT / "INCLUDE",
        PROJECT_ROOT / "include",
        PROJECT_ROOT / "data" / "INCLUDE",
        PROJECT_ROOT / "data" / "include",
    ]

    return [p for p in candidates if p.exists() and p.is_dir()]


def count_video_files(root):
    return sum(
        1
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def summarize_metadata(path, display_name):
    df = pd.read_csv(path)

    video_col = detect_video_column(df)
    class_col = detect_class_id_column(df)
    gloss_col = detect_gloss_column(df)
    split_col = detect_split_column(df)

    result = {
        "name": display_name,
        "path": path,
        "df": df,
        "video_col": video_col,
        "class_col": class_col,
        "gloss_col": gloss_col,
        "split_col": split_col,
        "rows": len(df),
    }

    result["unique_classes_by_id"] = (
        df[class_col].dropna().nunique()
        if class_col is not None
        else None
    )

    result["unique_classes_by_gloss"] = (
        normalize_text_series(df[gloss_col].dropna()).nunique()
        if gloss_col is not None
        else None
    )

    result["split_counts"] = (
        normalize_text_series(df[split_col])
        .str.lower()
        .value_counts(dropna=False)
        .to_dict()
        if split_col is not None
        else None
    )

    return result


def build_class_key(df, class_col, gloss_col):
    if gloss_col is not None:
        return normalize_text_series(df[gloss_col]).str.lower()

    if class_col is not None:
        return normalize_text_series(df[class_col]).str.lower()

    raise RuntimeError(
        "No class/gloss column found. Cannot compare class coverage."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-video-open",
        action="store_true",
        help="Also open every resolved video with OpenCV. Slower.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("INCLUDE STEP 1 - DATASET / METADATA AUDIT V2")
    print("=" * 78)
    print(f"\nProject root: {PROJECT_ROOT}")

    required = [SOURCE_METADATA, READY_METADATA, FINAL_METADATA]
    missing_required = [p for p in required if not p.exists()]

    if missing_required:
        print("\nMissing expected INCLUDE metadata:")
        for p in missing_required:
            print(f"  - {p}")
        raise FileNotFoundError(
            "Required INCLUDE metadata is missing. "
            "Do not train until the audit inputs are restored."
        )

    source = summarize_metadata(SOURCE_METADATA, "include_metadata.csv")
    ready = summarize_metadata(READY_METADATA, "include_ready.csv")
    final = summarize_metadata(FINAL_METADATA, "include_final.csv")

    summaries = [source, ready, final]

    print("\n" + "-" * 78)
    print("METADATA SUMMARY")
    print("-" * 78)

    for item in summaries:
        print(f"\n{item['name']}")
        print(f"Rows                : {item['rows']}")
        print(f"Class-ID column     : {item['class_col']}")
        print(f"Gloss column        : {item['gloss_col']}")
        print(f"Video-path column   : {item['video_col']}")
        print(f"Split column        : {item['split_col']}")
        print(f"Unique class IDs    : {item['unique_classes_by_id']}")
        print(f"Unique glosses      : {item['unique_classes_by_gloss']}")
        print(f"Split counts        : {item['split_counts']}")

    dataset_dirs = find_include_dataset_dirs()

    print("\n" + "-" * 78)
    print("DATASET FOLDER CHECK")
    print("-" * 78)

    if dataset_dirs:
        for d in dataset_dirs:
            print(f"{d} -> {count_video_files(d)} video files")
    else:
        print("No standard INCLUDE dataset folder was auto-detected.")
        print("This is okay if include_final.csv stores absolute video paths.")

    source_df = source["df"].copy()
    final_df = final["df"].copy()

    source_df["_audit_class_key"] = build_class_key(
        source_df, source["class_col"], source["gloss_col"]
    )
    final_df["_audit_class_key"] = build_class_key(
        final_df, final["class_col"], final["gloss_col"]
    )

    source_classes = set(source_df["_audit_class_key"].dropna())
    final_classes = set(final_df["_audit_class_key"].dropna())

    missing_classes = sorted(source_classes - final_classes)
    extra_classes = sorted(final_classes - source_classes)

    print("\n" + "-" * 78)
    print("CLASS COVERAGE")
    print("-" * 78)
    print(f"Classes in source metadata : {len(source_classes)}")
    print(f"Classes in final metadata  : {len(final_classes)}")
    print(f"Classes lost from source   : {len(missing_classes)}")
    print(f"Unexpected extra classes   : {len(extra_classes)}")

    if missing_classes:
        print("\nClasses present in include_metadata.csv but absent from include_final.csv:")
        for cls in missing_classes:
            print(f"  - {cls}")

    source_counts = source_df["_audit_class_key"].value_counts()
    final_counts = final_df["_audit_class_key"].value_counts()

    all_classes = sorted(source_classes | final_classes)

    class_report = pd.DataFrame(
        {
            "class_key": all_classes,
            "source_rows": [int(source_counts.get(c, 0)) for c in all_classes],
            "final_rows": [int(final_counts.get(c, 0)) for c in all_classes],
        }
    )
    class_report["dropped_rows"] = (
        class_report["source_rows"] - class_report["final_rows"]
    )
    class_report["present_in_final"] = class_report["final_rows"] > 0

    class_report.to_csv(CLASS_REPORT, index=False)

    existing_count = None
    missing_count = None

    if final["video_col"] is None:
        print("\nWARNING: Could not locate a video-path column in include_final.csv.")
    else:
        print("\n" + "-" * 78)
        print("FINAL VIDEO PATH CHECK")
        print("-" * 78)

        resolved_paths = []
        exists_flags = []

        for value in final_df[final["video_col"]]:
            resolved = resolve_video_path(value)
            resolved_paths.append("" if resolved is None else str(resolved))
            exists_flags.append(bool(resolved is not None and resolved.exists()))

        final_df["_resolved_video_path"] = resolved_paths
        final_df["_video_exists"] = exists_flags

        existing_count = int(sum(exists_flags))
        missing_count = int(len(exists_flags) - existing_count)

        print(f"Rows in include_final.csv : {len(final_df)}")
        print(f"Existing video paths      : {existing_count}")
        print(f"Missing video paths       : {missing_count}")

        final_df.loc[
            ~final_df["_video_exists"]
        ].to_csv(MISSING_ROWS_REPORT, index=False)

        if args.check_video_open:
            import cv2

            print("\nOpening every resolved video with OpenCV...")
            bad_open = []
            total = len(final_df)

            for idx, row in final_df.iterrows():
                if not row["_video_exists"]:
                    continue

                p = Path(row["_resolved_video_path"])
                cap = cv2.VideoCapture(str(p))
                ok = cap.isOpened()

                if ok:
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    ok = frame_count > 0

                cap.release()

                if not ok:
                    bad_open.append(str(p))

                if (idx + 1) % 250 == 0 or (idx + 1) == total:
                    print(f"Checked {idx + 1}/{total}")

            print(f"Unreadable/corrupt videos : {len(bad_open)}")

            (
                RESULTS_DIR / "include_audit_unreadable_v2.txt"
            ).write_text(
                "\n".join(bad_open),
                encoding="utf-8",
            )

    print("\n" + "-" * 78)
    print("SPLIT SAFETY")
    print("-" * 78)

    split_counts = final["split_counts"]

    if split_counts is None:
        print("No split column found in include_final.csv.")
    else:
        for k, v in sorted(split_counts.items()):
            print(f"{str(k):12s}: {v}")

        if final["video_col"] is not None and final["split_col"] is not None:
            tmp = final_df.copy()
            tmp["_video_key"] = normalize_text_series(
                tmp[final["video_col"]]
            ).str.lower()

            grouped = (
                tmp.groupby("_video_key")[final["split_col"]]
                .nunique()
            )
            leakage_count = int((grouped > 1).sum())
            print(f"Videos appearing in >1 split: {leakage_count}")

    summary_lines = [
        "=" * 78,
        "INCLUDE STEP 1 - AUDIT V2 SUMMARY",
        "=" * 78,
        "",
        f"Source metadata rows: {source['rows']}",
        f"Final metadata rows: {final['rows']}",
        f"Source classes: {len(source_classes)}",
        f"Final classes: {len(final_classes)}",
        f"Classes lost from source: {len(missing_classes)}",
        "",
        "Missing classes:",
        *([f"- {x}" for x in missing_classes] if missing_classes else ["None"]),
        "",
        f"Existing final video paths: {existing_count}",
        f"Missing final video paths: {missing_count}",
        "",
        f"Final split counts: {split_counts}",
        "",
        f"Class report: {CLASS_REPORT}",
        f"Missing rows report: {MISSING_ROWS_REPORT}",
        "",
        "STATUS:",
    ]

    if len(missing_classes) == 0 and (missing_count in [0, None]):
        summary_lines.append(
            "AUDIT PASSED FOR CURRENT METADATA ✅"
        )
    else:
        summary_lines.append(
            "AUDIT FOUND CLASS/PATH GAPS - FIX BEFORE NEW TRAINING ⚠️"
        )

    SUMMARY_REPORT.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n" + "\n".join(summary_lines))
    print("=" * 78)


if __name__ == "__main__":
    main()