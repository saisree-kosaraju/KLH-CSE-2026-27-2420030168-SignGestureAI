import os
import re
import json
import pandas as pd


# =========================================================
# PATHS
# =========================================================

WLASL_PATH = "datasets/WLASL"
WLASL_VIDEO_PATH = os.path.join(WLASL_PATH, "videos")
WLASL_JSON_PATH = os.path.join(WLASL_PATH, "nslt_300.json")

INCLUDE_PATH = "datasets/INCLUDE"

OUTPUT_DIR = "metadata"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# WLASL300 METADATA
# =========================================================

print("=" * 60)
print("CREATING WLASL300 METADATA")
print("=" * 60)


with open(WLASL_JSON_PATH, "r", encoding="utf-8") as file:
    wlasl_data = json.load(file)


wlasl_rows = []


for video_id, information in wlasl_data.items():

    video_filename = f"{video_id}.mp4"

    video_path = os.path.join(
        WLASL_VIDEO_PATH,
        video_filename
    )

    exists = os.path.exists(video_path)

    # -----------------------------------------------------
    # WLASL "action" is stored as a list.
    # The first value represents the class ID.
    # -----------------------------------------------------

    action = information.get("action", None)

    if isinstance(action, list):

        if len(action) > 0:
            label = action[0]
        else:
            label = None

    else:
        label = action


    split = information.get(
        "subset",
        "unknown"
    )


    wlasl_rows.append(
        {
            "video_id": video_id,
            "video_path": video_path,
            "label": label,
            "split": split,
            "exists": exists,
            "dataset": "WLASL300"
        }
    )


# =========================================================
# CONVERT WLASL DATA TO DATAFRAME
# =========================================================

wlasl_df = pd.DataFrame(
    wlasl_rows
)


# =========================================================
# SAVE COMPLETE WLASL300 METADATA
# =========================================================

wlasl_output = os.path.join(
    OUTPUT_DIR,
    "wlasl300_metadata.csv"
)


wlasl_df.to_csv(
    wlasl_output,
    index=False
)


print()
print("WLASL300 metadata created.")

print(
    "Total metadata entries:",
    len(wlasl_df)
)

print(
    "Videos found:",
    wlasl_df["exists"].sum()
)

print(
    "Videos missing:",
    (~wlasl_df["exists"]).sum()
)


# =========================================================
# CREATE CLEAN WLASL300 METADATA
# ONLY VIDEOS THAT ACTUALLY EXIST
# =========================================================

wlasl_available_df = wlasl_df[
    wlasl_df["exists"] == True
].copy()


clean_output = os.path.join(
    OUTPUT_DIR,
    "wlasl300_available.csv"
)


wlasl_available_df.to_csv(
    clean_output,
    index=False
)


print()

print(
    "Available WLASL300 videos:",
    len(wlasl_available_df)
)

print(
    "Unique WLASL300 class IDs:",
    wlasl_available_df["label"].nunique()
)


print()
print("Split distribution:")

print(
    wlasl_available_df["split"]
    .value_counts()
)


# =========================================================
# WLASL CLASS DISTRIBUTION
# =========================================================

print()
print("Top 10 WLASL class IDs by number of videos:")

print(
    wlasl_available_df["label"]
    .value_counts()
    .head(10)
)


# =========================================================
# INCLUDE METADATA
# =========================================================

print()
print("=" * 60)
print("CREATING INCLUDE METADATA")
print("=" * 60)


include_rows = []


for root, dirs, files in os.walk(INCLUDE_PATH):

    for filename in files:

        # -------------------------------------------------
        # ACCEPT VIDEO FILES
        # -------------------------------------------------

        if not filename.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        ):
            continue


        video_path = os.path.join(
            root,
            filename
        )


        relative_path = os.path.relpath(
            root,
            INCLUDE_PATH
        )


        parts = relative_path.split(
            os.sep
        )


        if len(parts) < 1:
            continue


        # -------------------------------------------------
        # FIRST FOLDER = CATEGORY
        #
        # Example:
        # INCLUDE/Animals/...
        #
        # category = Animals
        # -------------------------------------------------

        category = parts[0]


        # =================================================
        # FIND CORRECT SIGN LABEL
        # =================================================

        # Normally the folder directly containing
        # the video is the sign label.

        label_folder = os.path.basename(
            root
        )


        # -------------------------------------------------
        # SOME INCLUDE VIDEOS ARE INSIDE "Extra"
        #
        # Example:
        #
        # 5. Beautiful/
        #     Extra/
        #         MVI_9573.MOV
        #
        # "Extra" is NOT the class.
        #
        # Correct class = Beautiful
        # -------------------------------------------------

        if label_folder.lower() == "extra":

            label_folder = os.path.basename(
                os.path.dirname(root)
            )


        # -------------------------------------------------
        # REMOVE NUMBER PREFIX
        #
        # Examples:
        #
        # "1. loud"           -> "loud"
        # "5. Beautiful"      -> "Beautiful"
        # "21. Train Station" -> "Train Station"
        # -------------------------------------------------

        label = re.sub(
            r"^\s*\d+\s*[\.\-_]*\s*",
            "",
            label_folder
        ).strip()


        # Convert to lowercase for consistency

        label = label.lower()


        include_rows.append(
            {
                "video_name": filename,
                "video_path": video_path,
                "label": label,
                "category": category,
                "dataset": "INCLUDE"
            }
        )


# =========================================================
# CONVERT INCLUDE DATA TO DATAFRAME
# =========================================================

include_df = pd.DataFrame(
    include_rows
)


# =========================================================
# SAVE INCLUDE METADATA
# =========================================================

include_output = os.path.join(
    OUTPUT_DIR,
    "include_metadata.csv"
)


include_df.to_csv(
    include_output,
    index=False
)


print()
print("INCLUDE metadata created.")


print(
    "Total INCLUDE videos:",
    len(include_df)
)


print(
    "Number of unique INCLUDE labels:",
    include_df["label"].nunique()
)


# =========================================================
# CHECK IF "EXTRA" STILL EXISTS
# =========================================================

extra_count = (
    include_df["label"]
    .astype(str)
    .str.lower()
    .eq("extra")
    .sum()
)


print(
    "Videos incorrectly labelled 'extra':",
    extra_count
)


# =========================================================
# INCLUDE CATEGORY DISTRIBUTION
# =========================================================

print()
print("INCLUDE category distribution:")

print(
    include_df["category"]
    .value_counts()
)


# =========================================================
# INCLUDE LABEL DISTRIBUTION
# =========================================================

print()
print("Top 10 INCLUDE labels by number of videos:")

print(
    include_df["label"]
    .value_counts()
    .head(10)
)


# =========================================================
# CHECK SMALL INCLUDE CLASSES
# =========================================================

include_class_counts = (
    include_df["label"]
    .value_counts()
)


small_classes = include_class_counts[
    include_class_counts < 5
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
# FINAL SUMMARY
# =========================================================

print()
print("=" * 60)
print("METADATA GENERATION COMPLETED")
print("=" * 60)


print()
print("Generated files:")


print(
    os.path.join(
        OUTPUT_DIR,
        "wlasl300_metadata.csv"
    )
)


print(
    os.path.join(
        OUTPUT_DIR,
        "wlasl300_available.csv"
    )
)


print(
    os.path.join(
        OUTPUT_DIR,
        "include_metadata.csv"
    )
)