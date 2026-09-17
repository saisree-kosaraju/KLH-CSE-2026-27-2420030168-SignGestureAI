import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

print("=" * 75)
print("TESTING FIXED MEDIAPIPE FEATURE VECTOR")
print("=" * 75)

SEQUENCE_LENGTH = 32
MODEL_PATH = "models/holistic_landmarker.task"

# 40 official MediaPipe lip-region landmarks
LIP_INDICES = [
    0, 13, 14, 17, 37, 39, 40, 61,
    78, 80, 81, 82, 84, 87, 88, 91,
    95, 146, 178, 181, 185, 191,
    267, 269, 270, 291, 308, 310,
    311, 312, 314, 317, 318, 321,
    324, 375, 402, 405, 409, 415
]

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# =========================================================
# FRAME SAMPLING
# =========================================================

def sample_video_frames(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None, None

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if total_frames <= 0:
        cap.release()
        return None, None

    if fps <= 0:
        fps = 25.0

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    )

    frame_indices = np.round(
        frame_indices
    ).astype(int)

    duration_ms = (
        (total_frames - 1) / fps
    ) * 1000

    timestamps = np.linspace(
        0,
        duration_ms,
        SEQUENCE_LENGTH
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
            return None, None

        frames.append(frame)

    cap.release()

    return frames, timestamps


# =========================================================
# LANDMARK HELPERS
# =========================================================

def extract_xyz(landmarks, expected_count):

    output = np.zeros(
        expected_count * 3,
        dtype=np.float32
    )

    if landmarks is None:
        return output

    if len(landmarks) == 0:
        return output

    values = []

    for landmark in landmarks[:expected_count]:

        values.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    values = np.array(
        values,
        dtype=np.float32
    )

    output[:len(values)] = values

    return output


def extract_pose(landmarks):

    output = np.zeros(
        33 * 4,
        dtype=np.float32
    )

    if landmarks is None:
        return output

    if len(landmarks) == 0:
        return output

    values = []

    for landmark in landmarks[:33]:

        visibility = getattr(
            landmark,
            "visibility",
            0.0
        )

        if visibility is None:
            visibility = 0.0

        values.extend([
            landmark.x,
            landmark.y,
            landmark.z,
            visibility
        ])

    values = np.array(
        values,
        dtype=np.float32
    )

    output[:len(values)] = values

    return output


def extract_lips(face_landmarks):

    output = np.zeros(
        len(LIP_INDICES) * 3,
        dtype=np.float32
    )

    if face_landmarks is None:
        return output

    if len(face_landmarks) == 0:
        return output

    values = []

    for index in LIP_INDICES:

        landmark = face_landmarks[index]

        values.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.array(
        values,
        dtype=np.float32
    )


# =========================================================
# CREATE ONE FRAME FEATURE VECTOR
# =========================================================

def create_feature_vector(result):

    pose_detected = (
        len(result.pose_landmarks) > 0
    )

    face_detected = (
        len(result.face_landmarks) > 0
    )

    left_detected = (
        len(result.left_hand_landmarks) > 0
    )

    right_detected = (
        len(result.right_hand_landmarks) > 0
    )

    pose = extract_pose(
        result.pose_landmarks
    )

    left_hand = extract_xyz(
        result.left_hand_landmarks,
        21
    )

    right_hand = extract_xyz(
        result.right_hand_landmarks,
        21
    )

    lips = extract_lips(
        result.face_landmarks
    )

    detection_mask = np.array(
        [
            float(pose_detected),
            float(face_detected),
            float(left_detected),
            float(right_detected)
        ],
        dtype=np.float32
    )

    feature_vector = np.concatenate([
        pose,
        left_hand,
        right_hand,
        lips,
        detection_mask
    ])

    return feature_vector


# =========================================================
# EXTRACT VIDEO SEQUENCE
# =========================================================

def extract_video_features(video_path):

    frames, timestamps = sample_video_frames(
        video_path
    )

    if frames is None:
        return None

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

    sequence = []

    with HolisticLandmarker.create_from_options(
        options
    ) as landmarker:

        for frame_number, frame in enumerate(frames):

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            rgb = np.ascontiguousarray(
                rgb
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect_for_video(
                mp_image,
                int(timestamps[frame_number])
            )

            feature_vector = create_feature_vector(
                result
            )

            sequence.append(
                feature_vector
            )

    return np.stack(
        sequence
    ).astype(np.float32)


# =========================================================
# TEST WLASL300 + INCLUDE SEPARATELY
# =========================================================

wlasl = pd.read_csv(
    "metadata/wlasl300_final.csv"
)

include = pd.read_csv(
    "metadata/include_final.csv"
)

test_videos = {
    "WLASL300": wlasl.iloc[0]["video_path"],
    "INCLUDE": include.iloc[0]["video_path"]
}


for dataset_name, video_path in test_videos.items():

    print("\n" + "=" * 75)
    print(dataset_name)
    print("=" * 75)

    print(
        f"Video: {video_path}"
    )

    features = extract_video_features(
        video_path
    )

    if features is None:

        print(
            "Feature extraction FAILED"
        )

        continue

    print(
        f"Feature shape: {features.shape}"
    )

    print(
        f"Data type: {features.dtype}"
    )

    print(
        f"NaN values: {np.isnan(features).sum()}"
    )

    print(
        f"Infinite values: "
        f"{np.isinf(features).sum()}"
    )

    print(
        f"Minimum value: {features.min():.5f}"
    )

    print(
        f"Maximum value: {features.max():.5f}"
    )

    print(
        "Feature extraction: SUCCESS"
    )


print("\n" + "=" * 75)
print("EXPECTED FEATURE SHAPE: (32, 382)")
print("=" * 75)