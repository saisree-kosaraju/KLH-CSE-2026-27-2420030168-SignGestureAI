import os
import pandas as pd


# =========================================================
# PATHS
# =========================================================

WLASL_CLASS_FILE = "datasets/WLASL/wlasl_class_list.txt"

WLASL_METADATA = "metadata/wlasl300_available.csv"
INCLUDE_METADATA = "metadata/include_metadata.csv"

OUTPUT_DIR = "metadata"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD WLASL CLASS LIST
# =========================================================

print("=" * 60)
print("PREPARING WLASL LABELS")
print("=" * 60)


class_mapping = {}


with open(
    WLASL_CLASS_FILE,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        line = line.strip()

        if not line:
            continue

        # WLASL class list normally contains:
        #
        # 0    book
        # 1    drink
        # 2    computer
        #
        # Split only once so signs containing spaces
        # remain intact.

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            continue

        class_name = parts[1].strip().lower()

        class_mapping[class_id] = class_name


print(
    "Classes found in class list:",
    len(class_mapping)
)


# =========================================================
# LOAD AVAILABLE WLASL300 METADATA
# =========================================================

wlasl_df = pd.read_csv(
    WLASL_METADATA
)


# Make sure IDs are integers

wlasl_df["label"] = (
    pd.to_numeric(
        wlasl_df["label"],
        errors="coerce"
    )
)


# Remove invalid label rows if any

wlasl_df = wlasl_df.dropna(
    subset=["label"]
).copy()


wlasl_df["label"] = (
    wlasl_df["label"]
    .astype(int)
)


# =========================================================
# MAP CLASS ID -> WORD
# =========================================================

wlasl_df["class_id"] = wlasl_df["label"]


wlasl_df["class_name"] = (
    wlasl_df["class_id"]
    .map(class_mapping)
)


# =========================================================
# CHECK FOR MISSING CLASS NAMES
# =========================================================

missing_class_names = (
    wlasl_df["class_name"]
    .isna()
    .sum()
)


print(
    "Videos with missing class names:",
    missing_class_names
)


print(
    "Unique class IDs:",
    wlasl_df["class_id"].nunique()
)


print(
    "Unique mapped class names:",
    wlasl_df["class_name"].nunique()
)


# =========================================================
# DISPLAY SAMPLE WLASL CLASSES
# =========================================================

print()
print("First 20 WLASL classes:")
print()


sample_classes = (
    wlasl_df[
        [
            "class_id",
            "class_name"
        ]
    ]
    .drop_duplicates()
    .sort_values("class_id")
    .head(20)
)


print(
    sample_classes.to_string(
        index=False
    )
)


# =========================================================
# SAVE FINAL WLASL READY METADATA
# =========================================================

wlasl_ready = wlasl_df[
    [
        "video_id",
        "video_path",
        "class_id",
        "class_name",
        "split",
        "dataset"
    ]
].copy()


WLASL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "wlasl300_ready.csv"
)


wlasl_ready.to_csv(
    WLASL_OUTPUT,
    index=False
)


print()
print(
    "Saved:",
    WLASL_OUTPUT
)


# =========================================================
# WLASL CLASS DISTRIBUTION
# =========================================================

print()
print("Smallest WLASL classes:")
print()


print(
    wlasl_ready[
        "class_name"
    ]
    .value_counts()
    .sort_values()
    .head(20)
)


# =========================================================
# INCLUDE LABEL AUDIT
# =========================================================

print()
print("=" * 60)
print("CHECKING INCLUDE LABELS")
print("=" * 60)


include_df = pd.read_csv(
    INCLUDE_METADATA
)


print(
    "Total INCLUDE videos:",
    len(include_df)
)


print(
    "Unique INCLUDE labels:",
    include_df["label"].nunique()
)


# =========================================================
# FIND SUSPICIOUS LABELS
# =========================================================

print()
print("Checking label: extra")
print()


extra_rows = include_df[
    include_df["label"]
    .astype(str)
    .str.lower()
    .eq("extra")
]


print(
    "Videos labelled 'extra':",
    len(extra_rows)
)


if len(extra_rows) > 0:

    print()
    print("Paths containing 'extra':")
    print()

    for path in (
        extra_rows["video_path"]
        .head(30)
    ):

        print(path)


# =========================================================
# CHECK EMPTY LABELS
# =========================================================

empty_labels = include_df[
    include_df["label"]
    .isna()
]


print()
print(
    "Empty INCLUDE labels:",
    len(empty_labels)
)


# =========================================================
# CHECK VERY SMALL CLASSES
# =========================================================

include_counts = (
    include_df["label"]
    .value_counts()
)


small_classes = include_counts[
    include_counts < 5
]


print()
print(
    "INCLUDE classes with fewer than 5 videos:",
    len(small_classes)
)


if len(small_classes) > 0:

    print()
    print(small_classes)


# =========================================================
# SAVE INCLUDE READY FILE
# =========================================================

INCLUDE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "include_ready.csv"
)


include_df.to_csv(
    INCLUDE_OUTPUT,
    index=False
)


print()
print(
    "Saved:",
    INCLUDE_OUTPUT
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 60)
print("LABEL PREPARATION COMPLETED")
print("=" * 60)

print()
print("WLASL usable videos:", len(wlasl_ready))

print(
    "WLASL classes:",
    wlasl_ready["class_name"].nunique()
)

print(
    "INCLUDE usable videos:",
    len(include_df)
)

print(
    "INCLUDE classes:",
    include_df["label"].nunique()
)