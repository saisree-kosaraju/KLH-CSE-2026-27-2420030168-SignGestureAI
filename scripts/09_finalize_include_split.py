import pandas as pd
import numpy as np
import re
from pathlib import Path

print("=" * 75)
print("FINALIZING INCLUDE TRAIN / VAL / TEST SPLIT")
print("=" * 75)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_PATH = Path("metadata/include_ready.csv")
OUTPUT_PATH = Path("metadata/include_final.csv")

RANDOM_SEED = 42

VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------
# Load metadata
# ---------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print(f"\nRows loaded: {len(df)}")
print(f"Original classes: {df['label'].nunique()}")

# ---------------------------------------------------------
# Normalization functions
# ---------------------------------------------------------

def normalize_text(value):
    value = str(value).strip().lower()

    value = value.replace("_", " ")
    value = value.replace("-", " ")

    value = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def normalize_category(value):
    value = normalize_text(value)

    aliases = {
        "days and time": "day and time",
        "day and time": "day and time",

        "means of transport":
            "means of transportation",

        "means of transportation":
            "means of transportation"
    }

    return aliases.get(
        value,
        value
    )


# ---------------------------------------------------------
# Normalize labels and categories
# ---------------------------------------------------------

df["category_norm"] = (
    df["category"]
    .apply(normalize_category)
)

df["label_norm"] = (
    df["label"]
    .apply(normalize_text)
)

df["video_norm"] = (
    df["video_name"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# ---------------------------------------------------------
# Create class key
# ---------------------------------------------------------

df["class_key"] = (
    df["category_norm"]
    + "|"
    + df["label_norm"]
)

print(
    f"Unique normalized labels: "
    f"{df['label_norm'].nunique()}"
)

print(
    f"Unique category+label classes: "
    f"{df['class_key'].nunique()}"
)

# ---------------------------------------------------------
# Create recording group key
# ---------------------------------------------------------

df["group_key"] = (
    df["class_key"]
    + "|"
    + df["video_norm"]
)

print(
    f"Unique recording groups: "
    f"{df['group_key'].nunique()}"
)

duplicate_rows = (
    df["group_key"]
    .duplicated(keep=False)
    .sum()
)

print(
    f"Rows belonging to duplicated groups: "
    f"{duplicate_rows}"
)

# ---------------------------------------------------------
# Validate that every group belongs to exactly one class
# ---------------------------------------------------------

group_class_counts = (
    df.groupby("group_key")["class_key"]
    .nunique()
)

bad_groups = (
    group_class_counts[
        group_class_counts > 1
    ]
)

if len(bad_groups) > 0:

    raise ValueError(
        "A recording group belongs to more than one class."
    )

print(
    "Recording-group class consistency: OK"
)

# ---------------------------------------------------------
# Create class IDs
# ---------------------------------------------------------

class_keys = sorted(
    df["class_key"].unique()
)

class_id_map = {
    class_key: class_id
    for class_id, class_key
    in enumerate(class_keys)
}

df["class_id"] = (
    df["class_key"]
    .map(class_id_map)
)

print(
    f"Class IDs created: "
    f"{df['class_id'].nunique()}"
)

# ---------------------------------------------------------
# Create unique group table
# ---------------------------------------------------------

group_df = (
    df[
        [
            "group_key",
            "class_key",
            "class_id",
            "label_norm",
            "category_norm"
        ]
    ]
    .drop_duplicates(
        subset="group_key"
    )
    .reset_index(drop=True)
)

print(
    f"Unique groups available for splitting: "
    f"{len(group_df)}"
)

# ---------------------------------------------------------
# Function to calculate split counts
# ---------------------------------------------------------

def calculate_split_counts(n_groups):

    if n_groups < 4:
        raise ValueError(
            f"Class has only {n_groups} groups. "
            "At least 4 are required."
        )

    n_val = max(
        1,
        int(round(n_groups * VAL_RATIO))
    )

    n_test = max(
        1,
        int(round(n_groups * TEST_RATIO))
    )

    # Always preserve at least 2 training groups
    while (
        n_groups - n_val - n_test
    ) < 2:

        if n_val > 1:
            n_val -= 1

        elif n_test > 1:
            n_test -= 1

        else:
            raise ValueError(
                "Unable to create safe split."
            )

    n_train = (
        n_groups
        - n_val
        - n_test
    )

    return (
        n_train,
        n_val,
        n_test
    )


# ---------------------------------------------------------
# Perform deterministic per-class group split
# ---------------------------------------------------------

rng = np.random.default_rng(
    RANDOM_SEED
)

split_assignments = {}

class_split_summary = []

for class_key in class_keys:

    class_groups = (
        group_df[
            group_df["class_key"]
            == class_key
        ]["group_key"]
        .tolist()
    )

    # Sort first so result is reproducible
    class_groups = sorted(
        class_groups
    )

    # Shuffle deterministically
    class_groups = list(
        rng.permutation(
            class_groups
        )
    )

    n_groups = len(
        class_groups
    )

    (
        n_train,
        n_val,
        n_test
    ) = calculate_split_counts(
        n_groups
    )

    train_groups = (
        class_groups[
            :n_train
        ]
    )

    val_groups = (
        class_groups[
            n_train:
            n_train + n_val
        ]
    )

    test_groups = (
        class_groups[
            n_train + n_val:
        ]
    )

    for group in train_groups:
        split_assignments[group] = "train"

    for group in val_groups:
        split_assignments[group] = "val"

    for group in test_groups:
        split_assignments[group] = "test"

    class_split_summary.append({
        "class_key": class_key,
        "total_groups": n_groups,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups)
    })


# ---------------------------------------------------------
# Apply split to every original row
# ---------------------------------------------------------

df["split"] = (
    df["group_key"]
    .map(split_assignments)
)

# ---------------------------------------------------------
# Add dataset name
# ---------------------------------------------------------

df["dataset"] = "INCLUDE"

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("FINAL SPLIT VALIDATION")
print("=" * 75)

missing_split = (
    df["split"]
    .isna()
    .sum()
)

print(
    f"\nRows without split: "
    f"{missing_split}"
)

if missing_split != 0:
    raise ValueError(
        "Some rows were not assigned to a split."
    )

# ---------------------------------------------------------
# Check recording leakage
# ---------------------------------------------------------

group_split_counts = (
    df.groupby("group_key")["split"]
    .nunique()
)

leaking_groups = (
    group_split_counts[
        group_split_counts > 1
    ]
)

print(
    f"Recording groups appearing in multiple splits: "
    f"{len(leaking_groups)}"
)

if len(leaking_groups) != 0:
    raise ValueError(
        "Recording leakage detected!"
    )

# ---------------------------------------------------------
# Row-level split distribution
# ---------------------------------------------------------

print("\nROW-LEVEL SPLIT DISTRIBUTION:")

row_distribution = (
    df["split"]
    .value_counts()
    .reindex(
        ["train", "val", "test"]
    )
)

print(
    row_distribution
)

print("\nRow percentages:")

row_percentages = (
    df["split"]
    .value_counts(
        normalize=True
    )
    .reindex(
        ["train", "val", "test"]
    )
    * 100
)

print(
    row_percentages.round(2)
)

# ---------------------------------------------------------
# Group-level distribution
# ---------------------------------------------------------

group_with_split = (
    df[
        [
            "group_key",
            "split"
        ]
    ]
    .drop_duplicates()
)

print("\nUNIQUE-GROUP SPLIT DISTRIBUTION:")

print(
    group_with_split[
        "split"
    ]
    .value_counts()
    .reindex(
        ["train", "val", "test"]
    )
)

# ---------------------------------------------------------
# Check class coverage
# ---------------------------------------------------------

print("\nCLASS COVERAGE:")

for split_name in [
    "train",
    "val",
    "test"
]:

    split_df = df[
        df["split"]
        == split_name
    ]

    print(
        f"{split_name}: "
        f"{split_df['class_id'].nunique()} classes"
    )

# ---------------------------------------------------------
# Classes missing from any split
# ---------------------------------------------------------

all_class_ids = set(
    df["class_id"].unique()
)

for split_name in [
    "train",
    "val",
    "test"
]:

    split_class_ids = set(
        df.loc[
            df["split"] == split_name,
            "class_id"
        ]
    )

    missing = (
        all_class_ids
        - split_class_ids
    )

    print(
        f"Classes missing from {split_name}: "
        f"{len(missing)}"
    )

# ---------------------------------------------------------
# Smallest class split examples
# ---------------------------------------------------------

summary_df = pd.DataFrame(
    class_split_summary
)

summary_df = (
    summary_df
    .sort_values(
        "total_groups"
    )
)

print(
    "\nSmallest classes after splitting:"
)

print(
    summary_df
    .head(20)
    .to_string(index=False)
)

# ---------------------------------------------------------
# Missing-value check
# ---------------------------------------------------------

print("\nMISSING VALUES:")

important_columns = [
    "video_name",
    "video_path",
    "label",
    "category",
    "class_id",
    "group_key",
    "split"
]

for column in important_columns:

    print(
        f"{column}: "
        f"{df[column].isna().sum()}"
    )

# ---------------------------------------------------------
# Save final INCLUDE metadata
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

# Save split summary too
summary_output = Path(
    "metadata/include_split_summary.csv"
)

summary_df.to_csv(
    summary_output,
    index=False
)

print("\n" + "=" * 75)
print("INCLUDE FINALIZATION COMPLETED")
print("=" * 75)

print(
    f"\nFinal metadata saved: "
    f"{OUTPUT_PATH}"
)

print(
    f"Split summary saved: "
    f"{summary_output}"
)

print(
    f"\nFinal rows: "
    f"{len(df)}"
)

print(
    f"Final classes: "
    f"{df['class_id'].nunique()}"
)

print(
    f"Unique recording groups: "
    f"{df['group_key'].nunique()}"
)