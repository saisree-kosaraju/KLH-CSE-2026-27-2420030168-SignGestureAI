import pandas as pd
import re
from pathlib import Path

print("=" * 75)
print("CHECKING INCLUDE USING CATEGORY + LABEL + VIDEO NAME")
print("=" * 75)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

local_path = Path("metadata/include_ready.csv")

train_path = Path(
    "metadata/include_split_metadata/train-00000-of-00001.csv"
)

val_path = Path(
    "metadata/include_split_metadata/val-00000-of-00001.csv"
)

test_path = Path(
    "metadata/include_split_metadata/test-00000-of-00001.csv"
)

# ---------------------------------------------------------
# Normalization functions
# ---------------------------------------------------------

def normalize_text(value):
    """
    Normalize text for safer matching.
    """

    value = str(value).strip().lower()

    # Replace common separators with spaces
    value = value.replace("_", " ")
    value = value.replace("-", " ")

    # Remove extra punctuation
    value = re.sub(r"[^a-z0-9 ]+", " ", value)

    # Remove repeated spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


def clean_hf_label(value):
    """
    Convert:
        '50. Yellow'
    into:
        'yellow'
    """

    value = str(value)

    value = re.sub(
        r"^\s*\d+\.\s*",
        "",
        value
    )

    return normalize_text(value)


def normalize_category(value):
    """
    Normalize category names and handle known naming differences.
    """

    value = normalize_text(value)

    aliases = {
        "days and time": "day and time",
        "day and time": "day and time",

        "means of transportation": "means of transportation",
        "means of transport": "means of transportation",
    }

    return aliases.get(value, value)


def normalize_filename(value):
    return str(value).strip().lower()


# ---------------------------------------------------------
# Load local INCLUDE
# ---------------------------------------------------------

local_df = pd.read_csv(local_path)

print(f"\nLocal INCLUDE rows: {len(local_df)}")
print(
    f"Local INCLUDE classes: "
    f"{local_df['label'].nunique()}"
)

# Normalize local data

local_df["category_norm"] = (
    local_df["category"]
    .apply(normalize_category)
)

local_df["label_norm"] = (
    local_df["label"]
    .apply(normalize_text)
)

local_df["video_norm"] = (
    local_df["video_name"]
    .apply(normalize_filename)
)

# Composite identifier

local_df["match_key"] = (
    local_df["category_norm"]
    + "|"
    + local_df["label_norm"]
    + "|"
    + local_df["video_norm"]
)

# ---------------------------------------------------------
# Check duplicate composite keys locally
# ---------------------------------------------------------

local_duplicate_mask = (
    local_df["match_key"]
    .duplicated(keep=False)
)

print(
    "Duplicate LOCAL composite keys: "
    f"{local_duplicate_mask.sum()}"
)

if local_duplicate_mask.sum() > 0:

    print("\nExample duplicate LOCAL composite keys:")

    print(
        local_df.loc[
            local_duplicate_mask,
            [
                "video_name",
                "label",
                "category",
                "video_path",
                "match_key"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# ---------------------------------------------------------
# Prepare Hugging Face split
# ---------------------------------------------------------

def prepare_hf_split(path, split_name):

    df = pd.read_csv(path)

    df["split"] = split_name

    # Extract filename from relative HF path
    df["video_name"] = (
        df["video_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[-1]
    )

    df["category_norm"] = (
        df["parent_label"]
        .apply(normalize_category)
    )

    df["label_norm"] = (
        df["label"]
        .apply(clean_hf_label)
    )

    df["video_norm"] = (
        df["video_name"]
        .apply(normalize_filename)
    )

    df["match_key"] = (
        df["category_norm"]
        + "|"
        + df["label_norm"]
        + "|"
        + df["video_norm"]
    )

    return df


# ---------------------------------------------------------
# Load HF metadata
# ---------------------------------------------------------

train_df = prepare_hf_split(
    train_path,
    "train"
)

val_df = prepare_hf_split(
    val_path,
    "val"
)

test_df = prepare_hf_split(
    test_path,
    "test"
)

hf_df = pd.concat(
    [
        train_df,
        val_df,
        test_df
    ],
    ignore_index=True
)

print("\nHF split sizes:")

print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")
print(f"Test:  {len(test_df)}")
print(f"Total: {len(hf_df)}")


# ---------------------------------------------------------
# Check duplicate HF composite keys
# ---------------------------------------------------------

hf_duplicate_mask = (
    hf_df["match_key"]
    .duplicated(keep=False)
)

print(
    "\nDuplicate HF composite keys: "
    f"{hf_duplicate_mask.sum()}"
)

if hf_duplicate_mask.sum() > 0:

    print("\nExample duplicate HF composite keys:")

    print(
        hf_df.loc[
            hf_duplicate_mask,
            [
                "video_name",
                "label",
                "parent_label",
                "split",
                "video_path"
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


# ---------------------------------------------------------
# Check same composite video across different splits
# ---------------------------------------------------------

split_count = (
    hf_df.groupby("match_key")["split"]
    .nunique()
)

multi_split_keys = (
    split_count[
        split_count > 1
    ]
)

print(
    "\nComposite keys occurring in multiple HF splits: "
    f"{len(multi_split_keys)}"
)

if len(multi_split_keys) > 0:

    examples = list(
        multi_split_keys.index[:20]
    )

    print("\nExamples:")

    print(
        hf_df[
            hf_df["match_key"].isin(examples)
        ][
            [
                "video_name",
                "label",
                "parent_label",
                "split",
                "match_key"
            ]
        ]
        .sort_values("match_key")
        .to_string(index=False)
    )


# ---------------------------------------------------------
# Exact composite matching
# ---------------------------------------------------------

local_keys = set(
    local_df["match_key"]
)

hf_keys = set(
    hf_df["match_key"]
)

matched_keys = (
    local_keys & hf_keys
)

missing_local_keys = (
    local_keys - hf_keys
)

hf_only_keys = (
    hf_keys - local_keys
)

print("\n" + "=" * 75)
print("COMPOSITE MATCH RESULTS")
print("=" * 75)

print(
    f"Unique LOCAL composite keys: "
    f"{len(local_keys)}"
)

print(
    f"Unique HF composite keys: "
    f"{len(hf_keys)}"
)

print(
    f"Matched composite keys: "
    f"{len(matched_keys)}"
)

print(
    f"Local composite keys NOT found in HF: "
    f"{len(missing_local_keys)}"
)

print(
    f"HF composite keys NOT found locally: "
    f"{len(hf_only_keys)}"
)


# ---------------------------------------------------------
# Count matched LOCAL rows
# ---------------------------------------------------------

matched_local_df = local_df[
    local_df["match_key"].isin(
        matched_keys
    )
]

missing_local_df = local_df[
    ~local_df["match_key"].isin(
        matched_keys
    )
]

print(
    f"\nMatched LOCAL rows: "
    f"{len(matched_local_df)}"
)

print(
    f"Unmatched LOCAL rows: "
    f"{len(missing_local_df)}"
)


# ---------------------------------------------------------
# Match by split
# ---------------------------------------------------------

print("\nMatched LOCAL rows by HF split:")

for split_name in [
    "train",
    "val",
    "test"
]:

    split_keys = set(
        hf_df.loc[
            hf_df["split"] == split_name,
            "match_key"
        ]
    )

    count = (
        local_df["match_key"]
        .isin(split_keys)
        .sum()
    )

    print(
        f"{split_name}: {count}"
    )


# ---------------------------------------------------------
# Display unmatched local videos
# ---------------------------------------------------------

if len(missing_local_df) > 0:

    print("\n" + "=" * 75)
    print("UNMATCHED LOCAL VIDEOS")
    print("=" * 75)

    print(
        missing_local_df[
            [
                "video_name",
                "label",
                "category",
                "video_path"
            ]
        ]
        .head(50)
        .to_string(index=False)
    )


# ---------------------------------------------------------
# Category comparison
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("CATEGORY CHECK")
print("=" * 75)

print("\nLocal normalized categories:")

print(
    sorted(
        local_df["category_norm"]
        .unique()
    )
)

print("\nHF normalized categories:")

print(
    sorted(
        hf_df["category_norm"]
        .unique()
    )
)


# ---------------------------------------------------------
# Save diagnostic matches
# ---------------------------------------------------------

hf_unique = (
    hf_df
    .drop_duplicates(
        subset="match_key",
        keep=False
    )
)

match_table = local_df.merge(
    hf_unique[
        [
            "match_key",
            "split",
            "parent_label",
            "label",
            "video_path"
        ]
    ],
    on="match_key",
    how="left",
    suffixes=(
        "_local",
        "_hf"
    )
)

diagnostic_output = Path(
    "metadata/include_composite_match_check.csv"
)

match_table.to_csv(
    diagnostic_output,
    index=False
)

print("\n" + "=" * 75)
print("CHECK COMPLETED")
print("=" * 75)

print(
    f"\nDiagnostic file saved: "
    f"{diagnostic_output}"
)