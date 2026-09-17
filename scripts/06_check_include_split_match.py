import pandas as pd
from pathlib import Path

print("=" * 70)
print("CHECKING INCLUDE SPLIT METADATA AGAINST LOCAL DATASET")
print("=" * 70)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

local_metadata_path = Path("metadata/include_ready.csv")

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
# Load local INCLUDE metadata
# ---------------------------------------------------------

local_df = pd.read_csv(local_metadata_path)

print(f"\nLocal INCLUDE videos: {len(local_df)}")
print(f"Local INCLUDE classes: {local_df['label'].nunique()}")

# Normalize local video names
local_df["video_key"] = (
    local_df["video_name"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# ---------------------------------------------------------
# Check duplicate local filenames
# ---------------------------------------------------------

local_duplicates = (
    local_df["video_key"]
    .duplicated(keep=False)
)

print(
    f"Duplicate video filenames in local dataset: "
    f"{local_duplicates.sum()}"
)

if local_duplicates.sum() > 0:
    print("\nExample duplicate local filenames:")
    print(
        local_df.loc[
            local_duplicates,
            ["video_name", "label", "video_path"]
        ]
        .head(20)
        .to_string(index=False)
    )

# ---------------------------------------------------------
# Function to prepare HF split metadata
# ---------------------------------------------------------

def prepare_split(path, split_name):

    df = pd.read_csv(path)

    df["split"] = split_name

    # Extract video filename from HF relative path
    df["video_name"] = (
        df["video_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[-1]
    )

    df["video_key"] = (
        df["video_name"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


# ---------------------------------------------------------
# Load HF train / val / test metadata
# ---------------------------------------------------------

train_df = prepare_split(train_path, "train")
val_df = prepare_split(val_path, "val")
test_df = prepare_split(test_path, "test")

print("\nHugging Face split sizes:")
print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")
print(f"Test:  {len(test_df)}")

hf_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

print(f"Total HF metadata entries: {len(hf_df)}")

# ---------------------------------------------------------
# Check duplicate video filenames in HF metadata
# ---------------------------------------------------------

hf_duplicates = (
    hf_df["video_key"]
    .duplicated(keep=False)
)

print(
    f"Duplicate HF video filenames: "
    f"{hf_duplicates.sum()}"
)

# ---------------------------------------------------------
# Check whether same video occurs in multiple HF splits
# ---------------------------------------------------------

split_counts = (
    hf_df.groupby("video_key")["split"]
    .nunique()
)

multi_split_videos = split_counts[split_counts > 1]

print(
    f"Video filenames occurring in multiple HF splits: "
    f"{len(multi_split_videos)}"
)

if len(multi_split_videos) > 0:

    print("\nExample videos occurring in multiple splits:")

    examples = list(
        multi_split_videos.index[:20]
    )

    print(
        hf_df[
            hf_df["video_key"].isin(examples)
        ][
            ["video_name", "label", "split"]
        ]
        .sort_values("video_name")
        .to_string(index=False)
    )

# ---------------------------------------------------------
# Match local videos against HF metadata
# ---------------------------------------------------------

hf_video_set = set(hf_df["video_key"])
local_video_set = set(local_df["video_key"])

matched_local = local_video_set & hf_video_set
missing_from_hf = local_video_set - hf_video_set
hf_not_local = hf_video_set - local_video_set

print("\n" + "=" * 70)
print("MATCH RESULTS")
print("=" * 70)

print(
    f"Unique local video filenames: "
    f"{len(local_video_set)}"
)

print(
    f"Unique HF video filenames: "
    f"{len(hf_video_set)}"
)

print(
    f"Local videos found in HF metadata: "
    f"{len(matched_local)}"
)

print(
    f"Local videos NOT found in HF metadata: "
    f"{len(missing_from_hf)}"
)

print(
    f"HF videos not present locally: "
    f"{len(hf_not_local)}"
)

# ---------------------------------------------------------
# Match counts by split
# ---------------------------------------------------------

print("\nMatched local videos by HF split:")

for split_name in ["train", "val", "test"]:

    split_keys = set(
        hf_df.loc[
            hf_df["split"] == split_name,
            "video_key"
        ]
    )

    count = len(
        local_video_set & split_keys
    )

    print(
        f"{split_name}: {count}"
    )

# ---------------------------------------------------------
# Show local videos missing from HF metadata
# ---------------------------------------------------------

if len(missing_from_hf) > 0:

    print("\nFirst 30 local videos missing from HF metadata:")

    missing_rows = local_df[
        local_df["video_key"].isin(missing_from_hf)
    ]

    print(
        missing_rows[
            ["video_name", "label", "category", "video_path"]
        ]
        .head(30)
        .to_string(index=False)
    )

# ---------------------------------------------------------
# Check label agreement for matched videos
# ---------------------------------------------------------

hf_unique = (
    hf_df.sort_values("split")
    .drop_duplicates(
        subset="video_key",
        keep="first"
    )
)

comparison = local_df.merge(
    hf_unique[
        [
            "video_key",
            "label",
            "parent_label",
            "split"
        ]
    ],
    on="video_key",
    how="inner",
    suffixes=("_local", "_hf")
)

# Remove numeric prefix from HF label
# Example:
# "50. Yellow" -> "Yellow"

comparison["hf_clean_label"] = (
    comparison["label_hf"]
    .astype(str)
    .str.replace(
        r"^\s*\d+\.\s*",
        "",
        regex=True
    )
    .str.strip()
    .str.lower()
)

comparison["local_clean_label"] = (
    comparison["label_local"]
    .astype(str)
    .str.strip()
    .str.lower()
)

label_match = (
    comparison["hf_clean_label"]
    ==
    comparison["local_clean_label"]
)

print("\n" + "=" * 70)
print("LABEL CHECK")
print("=" * 70)

print(
    f"Matched videos checked: "
    f"{len(comparison)}"
)

print(
    f"Matching labels: "
    f"{label_match.sum()}"
)

print(
    f"Label mismatches: "
    f"{(~label_match).sum()}"
)

if (~label_match).sum() > 0:

    print("\nFirst 30 label mismatches:")

    print(
        comparison.loc[
            ~label_match,
            [
                "video_name",
                "label_local",
                "label_hf",
                "split"
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

print("\n" + "=" * 70)
print("CHECK COMPLETED")
print("=" * 70)