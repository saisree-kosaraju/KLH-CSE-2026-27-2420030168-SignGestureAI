import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path

print("=" * 75)
print("TESTING MEDIAPIPE HOLISTIC LANDMARK EXTRACTION")
print("=" * 75)

SEQUENCE_LENGTH = 32
MODEL_PATH = "models/holistic_landmarker.task"

# ---------------------------------------------------------
# MediaPipe aliases
# ---------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# ---------------------------------------------------------
# Sample exactly 32 frames
# ---------------------------------------------------------

def sample_video_frames(video_path, sequence_length=32):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None, None, None

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if total_frames <= 0:
        cap.release()
        return None, None, None

    if fps <= 0:
        fps = 25.0

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        sequence_length
    )

    frame_indices = np.round(
        frame_indices
    ).astype(int)

    # Create increasing timestamps across original duration
    duration_ms = (
        (total_frames - 1) / fps
    ) * 1000.0

    timestamps = np.linspace(
        0,
        duration_ms,
        sequence_length
    )

    timestamps = np.round(
        timestamps
    ).astype(int)

    frames = []

    for index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        success, frame = cap.read()

        if not success or frame is None:
            cap.release()
            return None, None, None

        frames.append(frame)

    cap.release()

    return frames, frame_indices, timestamps


# ---------------------------------------------------------
# Test one video
# ---------------------------------------------------------

def test_video(
    dataset_name,
    video_path
):

    print("\n" + "=" * 75)
    print(f"TESTING {dataset_name}")
    print("=" * 75)

    print(f"\nVideo: {video_path}")

    frames, indices, timestamps = sample_video_frames(
        video_path,
        SEQUENCE_LENGTH
    )

    if frames is None:
        print("ERROR: Frame sampling failed.")
        return

    print(
        f"Sampled frames: {len(frames)}"
    )

    print(
        f"First frame index: {indices[0]}"
    )

    print(
        f"Last frame index: {indices[-1]}"
    )

    # -----------------------------------------------------
    # Configure Holistic Landmarker
    # -----------------------------------------------------

    options = HolisticLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),

        running_mode=RunningMode.VIDEO,

        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,

        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,

        min_hand_landmarks_confidence=0.5,

        output_face_blendshapes=False,
        output_segmentation_mask=False
    )

    pose_detected_frames = 0
    face_detected_frames = 0
    left_hand_detected_frames = 0
    right_hand_detected_frames = 0

    pose_counts = []
    face_counts = []
    left_counts = []
    right_counts = []

    # -----------------------------------------------------
    # Run MediaPipe
    # -----------------------------------------------------

    with HolisticLandmarker.create_from_options(
        options
    ) as landmarker:

        for frame_number, frame in enumerate(frames):

            # OpenCV uses BGR.
            # MediaPipe SRGB expects RGB.
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            rgb_frame = np.ascontiguousarray(
                rgb_frame
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                timestamps[frame_number]
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # ---------------------------------------------
            # Count landmarks
            # ---------------------------------------------

            pose_count = len(
                result.pose_landmarks
            )

            face_count = len(
                result.face_landmarks
            )

            left_count = len(
                result.left_hand_landmarks
            )

            right_count = len(
                result.right_hand_landmarks
            )

            pose_counts.append(
                pose_count
            )

            face_counts.append(
                face_count
            )

            left_counts.append(
                left_count
            )

            right_counts.append(
                right_count
            )

            if pose_count > 0:
                pose_detected_frames += 1

            if face_count > 0:
                face_detected_frames += 1

            if left_count > 0:
                left_hand_detected_frames += 1

            if right_count > 0:
                right_hand_detected_frames += 1

            print(
                f"Frame {frame_number + 1:02d}/32 | "
                f"Pose: {pose_count:3d} | "
                f"Face: {face_count:3d} | "
                f"Left hand: {left_count:2d} | "
                f"Right hand: {right_count:2d}"
            )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "-" * 75)
    print(f"{dataset_name} LANDMARK SUMMARY")
    print("-" * 75)

    print(
        f"Pose detected in: "
        f"{pose_detected_frames}/32 frames"
    )

    print(
        f"Face detected in: "
        f"{face_detected_frames}/32 frames"
    )

    print(
        f"Left hand detected in: "
        f"{left_hand_detected_frames}/32 frames"
    )

    print(
        f"Right hand detected in: "
        f"{right_hand_detected_frames}/32 frames"
    )

    print(
        "\nMaximum landmarks detected in one frame:"
    )

    print(
        f"Pose landmarks: "
        f"{max(pose_counts)}"
    )

    print(
        f"Face landmarks: "
        f"{max(face_counts)}"
    )

    print(
        f"Left-hand landmarks: "
        f"{max(left_counts)}"
    )

    print(
        f"Right-hand landmarks: "
        f"{max(right_counts)}"
    )


# ---------------------------------------------------------
# Load one WLASL video
# ---------------------------------------------------------

wlasl = pd.read_csv(
    "metadata/wlasl300_final.csv"
)

wlasl_video = wlasl.iloc[0][
    "video_path"
]


# ---------------------------------------------------------
# Load one INCLUDE video
# ---------------------------------------------------------

include = pd.read_csv(
    "metadata/include_final.csv"
)

include_video = include.iloc[0][
    "video_path"
]


# ---------------------------------------------------------
# Test separately
# ---------------------------------------------------------

test_video(
    "WLASL300",
    wlasl_video
)

test_video(
    "INCLUDE",
    include_video
)


print("\n" + "=" * 75)
print("HOLISTIC LANDMARK TEST COMPLETED")
print("=" * 75)