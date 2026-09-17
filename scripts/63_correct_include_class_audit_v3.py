from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
META = ROOT / "metadata"
RESULTS = ROOT / "results"

SOURCE = META / "include_metadata.csv"
FINAL = META / "include_final.csv"

CLASS_REPORT = META / "include_class_audit_v3.csv"
SUMMARY = RESULTS / "include_class_audit_v3_summary.txt"


def norm(s):
    return s.astype(str).str.strip()


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("INCLUDE STEP 2 - CORRECT CLASS AUDIT V3")
    print("=" * 78)

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not FINAL.exists():
        raise FileNotFoundError(FINAL)

    src = pd.read_csv(SOURCE)
    fin = pd.read_csv(FINAL)

    print("\nSOURCE COLUMNS:")
    print(src.columns.tolist())

    print("\nFINAL COLUMNS:")
    print(fin.columns.tolist())

    if "label" not in src.columns:
        raise RuntimeError(
            "include_metadata.csv must contain the original 'label' column."
        )

    if "class_id" not in fin.columns:
        raise RuntimeError(
            "include_final.csv must contain the mapped 'class_id' column."
        )

    src["label"] = norm(src["label"])
    fin["class_id"] = pd.to_numeric(fin["class_id"], errors="raise").astype(int)

    source_labels = sorted(src["label"].dropna().unique().tolist())
    final_class_ids = sorted(fin["class_id"].dropna().unique().tolist())

    print("\n" + "-" * 78)
    print("TRUE CLASS COUNTS")
    print("-" * 78)
    print(f"Source original labels : {len(source_labels)}")
    print(f"Final mapped class IDs : {len(final_class_ids)}")

    if "category" in src.columns:
        src["category"] = norm(src["category"])
        print(f"Broad categories       : {src['category'].nunique()}")
        print("\nNOTE:")
        print("'category' is a broad group field, NOT the sign-class/gloss field.")

    print("\n" + "-" * 78)
    print("FINAL CLASS-ID RANGE")
    print("-" * 78)

    if final_class_ids:
        print(f"Minimum class_id : {min(final_class_ids)}")
        print(f"Maximum class_id : {max(final_class_ids)}")
        expected = list(range(min(final_class_ids), max(final_class_ids) + 1))
        missing_ids = sorted(set(expected) - set(final_class_ids))
    else:
        missing_ids = []

    print(f"Missing IDs inside range: {len(missing_ids)}")
    if missing_ids:
        print(missing_ids)

    print("\n" + "-" * 78)
    print("VIDEOS PER FINAL CLASS")
    print("-" * 78)

    counts = (
        fin.groupby("class_id")
        .size()
        .rename("total_videos")
        .reset_index()
        .sort_values(["total_videos", "class_id"])
        .reset_index(drop=True)
    )

    if "split" in fin.columns:
        split_table = (
            fin.pivot_table(
                index="class_id",
                columns="split",
                values=fin.columns[0],
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
        )
        counts = counts.merge(split_table, on="class_id", how="left")

    print(counts.head(20).to_string(index=False))
    print("\n...")
    print(counts.tail(10).to_string(index=False))

    print(f"\nMinimum videos/class : {counts['total_videos'].min()}")
    print(f"Maximum videos/class : {counts['total_videos'].max()}")
    print(f"Median videos/class  : {counts['total_videos'].median():.1f}")

    rare = counts[counts["total_videos"] < 5].copy()
    print(f"Classes with <5 videos: {len(rare)}")

    # If final still has original label, verify one-to-one label -> class_id.
    mapping_status = "NOT CHECKED"
    mapping_rows = None

    if "label" in fin.columns:
        fin["label"] = norm(fin["label"])

        mapping_rows = (
            fin[["label", "class_id"]]
            .drop_duplicates()
            .sort_values(["class_id", "label"])
            .reset_index(drop=True)
        )

        label_to_ids = fin.groupby("label")["class_id"].nunique()
        id_to_labels = fin.groupby("class_id")["label"].nunique()

        bad_label = int((label_to_ids > 1).sum())
        bad_id = int((id_to_labels > 1).sum())

        mapping_status = (
            "PASS"
            if bad_label == 0 and bad_id == 0
            else f"FAIL (label->ids conflicts={bad_label}, id->labels conflicts={bad_id})"
        )

        print("\n" + "-" * 78)
        print("LABEL -> CLASS_ID MAPPING")
        print("-" * 78)
        print(f"Mapping status: {mapping_status}")
        print(f"Unique labels in final: {fin['label'].nunique()}")

    counts.to_csv(CLASS_REPORT, index=False)

    summary_lines = [
        "=" * 78,
        "INCLUDE CLASS AUDIT V3 SUMMARY",
        "=" * 78,
        "",
        f"Rows in source metadata: {len(src)}",
        f"Rows in final metadata: {len(fin)}",
        f"TRUE source sign labels: {len(source_labels)}",
        f"Final mapped class IDs: {len(final_class_ids)}",
        f"Broad categories: {src['category'].nunique() if 'category' in src.columns else 'N/A'}",
        f"Missing IDs inside final range: {len(missing_ids)}",
        f"Minimum videos/class: {counts['total_videos'].min()}",
        f"Maximum videos/class: {counts['total_videos'].max()}",
        f"Median videos/class: {counts['total_videos'].median():.1f}",
        f"Classes with <5 videos: {len(rare)}",
        f"Final label->class mapping check: {mapping_status}",
        "",
        "IMPORTANT:",
        "The earlier audit incorrectly treated 'category' as the sign-class/gloss field.",
        "'category' has only broad groups; 'label' / 'class_id' are the actual class fields.",
        "",
        f"Class report: {CLASS_REPORT}",
        "",
        "STATUS:",
    ]

    if len(source_labels) == len(final_class_ids) == 257 and len(missing_ids) == 0:
        summary_lines.append(
            "CURRENT PREPARED DATASET IS INTERNALLY CONSISTENT AT 257 CLASSES ✅"
        )
        summary_lines.append(
            "NEXT: identify the 6 classes absent relative to the intended 263-class INCLUDE benchmark."
        )
    else:
        summary_lines.append(
            "CLASS COUNT / MAPPING NEEDS INVESTIGATION BEFORE TRAINING ⚠️"
        )

    SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "\n".join(summary_lines))
    print("=" * 78)


if __name__ == "__main__":
    main()