import os
import json


# =========================================================
# PROJECT PATHS
# =========================================================

WLASL_PATH = "datasets/WLASL"
WLASL_VIDEO_PATH = os.path.join(WLASL_PATH, "videos")
WLASL_JSON_PATH = os.path.join(WLASL_PATH, "nslt_300.json")

INCLUDE_PATH = "datasets/INCLUDE"


# =========================================================
# VERIFY WLASL
# =========================================================

print("=" * 50)
print("VERIFYING WLASL300")
print("=" * 50)

if os.path.exists(WLASL_PATH):
    print("✓ WLASL folder found")
else:
    print("✗ WLASL folder NOT found")


if os.path.exists(WLASL_VIDEO_PATH):
    print("✓ WLASL videos folder found")

    video_files = [
        file
        for file in os.listdir(WLASL_VIDEO_PATH)
        if file.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    print("Number of WLASL videos:", len(video_files))

else:
    print("✗ WLASL videos folder NOT found")


if os.path.exists(WLASL_JSON_PATH):

    print("✓ nslt_300.json found")

    with open(WLASL_JSON_PATH, "r", encoding="utf-8") as file:
        wlasl_data = json.load(file)

    print("Number of entries in nslt_300.json:", len(wlasl_data))

else:
    print("✗ nslt_300.json NOT found")


# =========================================================
# VERIFY INCLUDE
# =========================================================

print()
print("=" * 50)
print("VERIFYING INCLUDE")
print("=" * 50)

if os.path.exists(INCLUDE_PATH):

    print("✓ INCLUDE folder found")

    categories = [
        folder
        for folder in os.listdir(INCLUDE_PATH)
        if os.path.isdir(os.path.join(INCLUDE_PATH, folder))
    ]

    print("Number of INCLUDE categories:", len(categories))

    print("\nCategories:")

    for category in sorted(categories):
        print(" -", category)

else:

    print("✗ INCLUDE folder NOT found")


# =========================================================
# COUNT INCLUDE VIDEOS
# =========================================================

include_video_count = 0

if os.path.exists(INCLUDE_PATH):

    for root, dirs, files in os.walk(INCLUDE_PATH):

        for file in files:

            if file.lower().endswith((".mp4", ".avi", ".mov")):
                include_video_count += 1


print("\nNumber of INCLUDE videos:", include_video_count)


print()
print("=" * 50)
print("DATASET VERIFICATION COMPLETED")
print("=" * 50)