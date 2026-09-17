import cv2
import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 75)
print("TESTING FIXED 32-FRAME SAMPLING")
print("=" * 75)

SEQUENCE_LENGTH = 32


# =========================================================
# FRAME SAMPLING FUNCTION
# =========================================================

def sample_video_frames(video_path, sequence_length=32):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None, None

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        return None, None

    # -----------------------------------------------------
    # Uniformly select exactly 32 frame positions
    # -----------------------------------------------------

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        sequence_length
    )

    frame_indices = np.round(
        frame_indices
    ).astype(int)

    sampled_frames = []

    # -----------------------------------------------------
    # Read selected frames
    # -----------------------------------------------------

    for frame_index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = cap.read()

        if not success or frame is None:
            cap.release()
            return None, frame_indices

        sampled_frames.append(frame)

    cap.release()

    return sampled_frames, frame_indices


# =========================================================
# TEST FUNCTION
# =========================================================

def test_dataset(
    dataset_name,
    metadata_path,
    number_of_tests=5
):

    print("\n" + "=" * 75)
    print(f"TESTING {dataset_name}")
    print("=" * 75)

    df = pd.read_csv(
        metadata_path
    )

    # -----------------------------------------------------
    # Select videos with different lengths
    # -----------------------------------------------------

    check_file = (
        "metadata/wlasl300_video_check.csv"
        if dataset_name == "WLASL300"
        else "metadata/include_video_check.csv"
    )

    check_df = pd.read_csv(
        check_file
    )

    check_df = check_df[
        check_df["valid_video"] == True
    ].copy()

    # Sort by frame count
    check_df = check_df.sort_values(
        "frame_count"
    ).reset_index(drop=True)

    # Pick videos across the distribution
    positions = np.linspace(
        0,
        len(check_df) - 1,
        number_of_tests
    ).astype(int)

    test_rows = check_df.iloc[
        positions
    ]

    successes = 0

    for test_number, (_, row) in enumerate(
        test_rows.iterrows(),
        start=1
    ):

        video_path = row["video_path"]
        total_frames = row["frame_count"]

        print(
            f"\nTest {test_number}"
        )

        print(
            f"Video: {video_path}"
        )

        print(
            f"Original frames: {total_frames}"
        )

        frames, indices = sample_video_frames(
            video_path,
            SEQUENCE_LENGTH
        )

        if frames is None:

            print(
                "Sampling: FAILED"
            )

            continue

        successes += 1

        print(
            f"Sampled frames: {len(frames)}"
        )

        print(
            f"First sampled index: {indices[0]}"
        )

        print(
            f"Last sampled index: {indices[-1]}"
        )

        print(
            "Sample indices:"
        )

        print(indices)

        # -------------------------------------------------
        # Check dimensions of first sampled frame
        # -------------------------------------------------

        first_frame = frames[0]

        height, width = (
            first_frame.shape[:2]
        )

        print(
            f"Frame resolution: "
            f"{width}x{height}"
        )

        # -------------------------------------------------
        # Detect repeated indices
        # -------------------------------------------------

        unique_indices = len(
            np.unique(indices)
        )

        repeated = (
            SEQUENCE_LENGTH
            - unique_indices
        )

        print(
            f"Unique sampled positions: "
            f"{unique_indices}"
        )

        print(
            f"Repeated positions: "
            f"{repeated}"
        )

        print(
            "Sampling: SUCCESS"
        )

    print("\n" + "-" * 75)

    print(
        f"{dataset_name}: "
        f"{successes}/{number_of_tests} "
        f"sampling tests passed"
    )


# =========================================================
# TEST WLASL300
# =========================================================

test_dataset(
    "WLASL300",
    "metadata/wlasl300_final.csv"
)


# =========================================================
# TEST INCLUDE
# =========================================================

test_dataset(
    "INCLUDE",
    "metadata/include_final.csv"
)


print("\n" + "=" * 75)
print("FRAME SAMPLING TEST COMPLETED")
print("=" * 75)

print(
    f"\nFixed sequence length: "
    f"{SEQUENCE_LENGTH} frames"
)