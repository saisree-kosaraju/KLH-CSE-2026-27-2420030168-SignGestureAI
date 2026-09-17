import pandas as pd
import re
from pathlib import Path

print("=" * 70)
print("CHECKING INCLUDE UNIQUE RECORDINGS PER CLASS")
print("=" * 70)

input_path = Path("metadata/include_ready.csv")

df = pd.read_csv(input_path)

# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------

def normalize_text(value):
    value = str(value).strip().lower()
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_category(value):
    value = normalize_text(value)

    aliases = {
        "days and time": "day and time",
        "day and time": "day and time",
        "means of transport": "means of transportation",
        "means of transportation": "means of transportation"
    }

    return aliases.get(value, value)


df["category_norm"] = df["category"].apply(normalize_category)
df["label_norm"] = df["label"].apply(normalize_text)

df["video_norm"] = (
    df["video_name"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Composite recording key
df["group_key"] = (
    df["category_norm"]
    + "|"
    + df["label_norm"]
    + "|"
    + df["video_norm"]
)

# ---------------------------------------------------------
# Basic information
# ---------------------------------------------------------

print(f"\nTotal rows: {len(df)}")
print(f"Classes: {df['label_norm'].nunique()}")
print(f"Unique recording groups: {df['group_key'].nunique()}")

duplicate_rows = df["group_key"].duplicated(keep=False).sum()

print(f"Rows belonging to duplicated groups: {duplicate_rows}")

# ---------------------------------------------------------
# Unique groups per class
# ---------------------------------------------------------

group_counts = (
    df.groupby("label_norm")["group_key"]
    .nunique()
    .sort_values()
)

row_counts = (
    df.groupby("label_norm")
    .size()
)

summary = pd.DataFrame({
    "rows": row_counts,
    "unique_groups": group_counts
})

summary["duplicate_difference"] = (
    summary["rows"] - summary["unique_groups"]
)

summary = summary.sort_values(
    ["unique_groups", "rows"]
)

print("\nSmallest classes by UNIQUE recording groups:")
print(summary.head(30))

# ---------------------------------------------------------
# Important thresholds
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SMALL CLASS CHECK")
print("=" * 70)

for threshold in [3, 4, 5, 6]:

    small = summary[
        summary["unique_groups"] < threshold
    ]

    print(
        f"\nClasses with fewer than {threshold} "
        f"unique recording groups: {len(small)}"
    )

    if len(small) > 0:
        print(small)

# ---------------------------------------------------------
# Duplicate group sizes
# ---------------------------------------------------------

group_sizes = (
    df.groupby("group_key")
    .size()
    .sort_values(ascending=False)
)

duplicated_groups = group_sizes[
    group_sizes > 1
]

print("\n" + "=" * 70)
print("DUPLICATE GROUP INFORMATION")
print("=" * 70)

print(
    f"\nNumber of duplicated recording groups: "
    f"{len(duplicated_groups)}"
)

print(
    f"Largest number of copies of one recording: "
    f"{group_sizes.max()}"
)

if len(duplicated_groups) > 0:

    print("\nLargest duplicate groups:")

    print(
        duplicated_groups
        .head(20)
    )

# ---------------------------------------------------------
# Save summary
# ---------------------------------------------------------

output_path = Path(
    "metadata/include_class_group_counts.csv"
)

summary.to_csv(output_path)

print("\n" + "=" * 70)
print("CHECK COMPLETED")
print("=" * 70)

print(f"\nSaved: {output_path}")