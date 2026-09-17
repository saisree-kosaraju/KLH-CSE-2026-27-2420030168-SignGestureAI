import cv2
import pandas as pd
from pathlib import Path

print("=" * 75)
print("VIDEO INTEGRITY VERIFICATION")
print("=" * 75)

# =========================================================
# CONFIGURATION
# =========================================================

DATASETS = {
    "WLASL300": {
        "metadata": Path("metadata/wlasl300_final.csv"),
        "output": Path("metadata/wlasl300_video_check.csv")
    },

    "INCLUDE": {
        "metadata": Path("metadata/include_final.csv"),
        "output": Path("metadata/include_video_check.csv")
    }
}


# =========================================================
# VIDEO CHECK FUNCTION
# =========================================================

def check_video(video_path):

    result = {
        "file_exists": False,
        "can_open": False,
        "first_frame_ok": False,
        "frame_count": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "duration_seconds": 0.0,
        "valid_video": False,
        "error": ""
    }

    path = Path(video_path)

    # -----------------------------------------------------
    # Check file existence
    # -----------------------------------------------------

    if not path.exists():
        result["error"] = "File does not exist"
        return result

    result["file_exists"] = True

    # -----------------------------------------------------
    # Open video
    # -----------------------------------------------------

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        result["error"] = "OpenCV could not open video"
        cap.release()
        return result

    result["can_open"] = True

    # -----------------------------------------------------
    # Read video properties
    # -----------------------------------------------------

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    result["frame_count"] = frame_count
    result["fps"] = fps
    result["width"] = width
    result["height"] = height

    # -----------------------------------------------------
    # Calculate approximate duration
    # -----------------------------------------------------

    if fps > 0 and frame_count > 0:
        result["duration_seconds"] = (
            frame_count / fps
        )

    # -----------------------------------------------------
    # Decode first frame
    # -----------------------------------------------------

    success, frame = cap.read()

    if success and frame is not None:
        result["first_frame_ok"] = True

    cap.release()

    # -----------------------------------------------------
    # Final validity decision
    # -----------------------------------------------------

    if (
        result["file_exists"]
        and result["can_open"]
        and result["first_frame_ok"]
        and result["frame_count"] > 0
        and result["width"] > 0
        and result["height"] > 0
    ):
        result["valid_video"] = True

    else:
        if result["error"] == "":
            result["error"] = (
                "Invalid video metadata or "
                "first frame could not be decoded"
            )

    return result


# =========================================================
# PROCESS ONE DATASET
# =========================================================

def verify_dataset(
    dataset_name,
    metadata_path,
    output_path
):

    print("\n" + "=" * 75)
    print(f"CHECKING {dataset_name}")
    print("=" * 75)

    if not metadata_path.exists():
        print(
            f"ERROR: Metadata file not found: "
            f"{metadata_path}"
        )
        return

    df = pd.read_csv(metadata_path)

    print(f"\nVideos listed: {len(df)}")

    if "video_path" not in df.columns:
        print(
            "ERROR: video_path column "
            "not found in metadata."
        )
        return

    results = []

    total = len(df)

    # -----------------------------------------------------
    # Check every video
    # -----------------------------------------------------

    for index, row in df.iterrows():

        video_path = row["video_path"]

        result = check_video(
            video_path
        )

        result["row_index"] = index
        result["video_path"] = video_path

        # WLASL identifier
        if "video_id" in df.columns:
            result["video_id"] = row["video_id"]

        # INCLUDE identifier
        if "video_name" in df.columns:
            result["video_name"] = row["video_name"]

        # Label
        if "class_name" in df.columns:
            result["label"] = row["class_name"]

        elif "label" in df.columns:
            result["label"] = row["label"]

        # Split
        if "split" in df.columns:
            result["split"] = row["split"]

        results.append(result)

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        checked = index + 1

        if (
            checked % 100 == 0
            or checked == total
        ):
            print(
                f"Checked {checked}/{total} videos..."
            )

    # -----------------------------------------------------
    # Convert results to DataFrame
    # -----------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    # -----------------------------------------------------
    # Save detailed report
    # -----------------------------------------------------

    result_df.to_csv(
        output_path,
        index=False
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    valid_count = (
        result_df["valid_video"]
        .sum()
    )

    invalid_count = (
        len(result_df)
        - valid_count
    )

    print("\n" + "-" * 75)
    print(f"{dataset_name} SUMMARY")
    print("-" * 75)

    print(
        f"Total videos: {len(result_df)}"
    )

    print(
        f"Files found: "
        f"{result_df['file_exists'].sum()}"
    )

    print(
        f"Videos OpenCV could open: "
        f"{result_df['can_open'].sum()}"
    )

    print(
        f"Videos with readable first frame: "
        f"{result_df['first_frame_ok'].sum()}"
    )

    print(
        f"Valid videos: {valid_count}"
    )

    print(
        f"Invalid videos: {invalid_count}"
    )

    # -----------------------------------------------------
    # Frame statistics
    # -----------------------------------------------------

    valid_df = result_df[
        result_df["valid_video"]
    ]

    if len(valid_df) > 0:

        print("\nFRAME STATISTICS:")

        print(
            f"Minimum frames: "
            f"{valid_df['frame_count'].min()}"
        )

        print(
            f"Maximum frames: "
            f"{valid_df['frame_count'].max()}"
        )

        print(
            f"Average frames: "
            f"{valid_df['frame_count'].mean():.2f}"
        )

        print("\nFPS STATISTICS:")

        print(
            f"Minimum FPS: "
            f"{valid_df['fps'].min():.2f}"
        )

        print(
            f"Maximum FPS: "
            f"{valid_df['fps'].max():.2f}"
        )

        print(
            f"Average FPS: "
            f"{valid_df['fps'].mean():.2f}"
        )

        print("\nDURATION STATISTICS:")

        print(
            f"Minimum duration: "
            f"{valid_df['duration_seconds'].min():.2f} sec"
        )

        print(
            f"Maximum duration: "
            f"{valid_df['duration_seconds'].max():.2f} sec"
        )

        print(
            f"Average duration: "
            f"{valid_df['duration_seconds'].mean():.2f} sec"
        )

    # -----------------------------------------------------
    # Invalid videos
    # -----------------------------------------------------

    if invalid_count > 0:

        print("\nINVALID VIDEOS:")

        invalid_df = result_df[
            ~result_df["valid_video"]
        ]

        columns = [
            "video_path",
            "error"
        ]

        print(
            invalid_df[
                columns
            ].to_string(index=False)
        )

    # -----------------------------------------------------
    # Split validation
    # -----------------------------------------------------

    if "split" in result_df.columns:

        print("\nVALID VIDEOS BY SPLIT:")

        print(
            result_df[
                result_df["valid_video"]
            ]["split"]
            .value_counts()
        )

    print(
        f"\nDetailed report saved: "
        f"{output_path}"
    )


# =========================================================
# RUN BOTH DATASETS SEPARATELY
# =========================================================

for dataset_name, config in DATASETS.items():

    verify_dataset(
        dataset_name,
        config["metadata"],
        config["output"]
    )


print("\n" + "=" * 75)
print("VIDEO INTEGRITY VERIFICATION COMPLETED")
print("=" * 75)

print(
    "\nWLASL300 and INCLUDE were checked "
    "separately."
)