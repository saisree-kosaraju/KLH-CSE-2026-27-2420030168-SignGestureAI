import cv2
import numpy as np
import mediapipe as mp

print("=" * 75)
print("TESTING ROBUST MEDIAPIPE IMAGE MODE")
print("=" * 75)

SEQUENCE_LENGTH = 32

# Every frame sent to MediaPipe will have exactly this size
INPUT_SIZE = 512

MODEL_PATH = "models/holistic_landmarker.task"

LIP_INDICES = [
    0, 13, 14, 17, 37, 39, 40, 61,
    78, 80, 81, 82, 84, 87, 88, 91,
    95, 146, 178, 181, 185, 191,
    267, 269, 270, 291, 308, 310,
    311, 312, 314, 317, 318, 321,
    324, 375, 402, 405, 409, 415
]

BaseOptions = mp.tasks.BaseOptions

HolisticLandmarker = (
    mp.tasks.vision.HolisticLandmarker
)

HolisticLandmarkerOptions = (
    mp.tasks.vision.HolisticLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


# =========================================================
# RESIZE WITH ASPECT RATIO PRESERVED
# =========================================================

def letterbox_frame(frame, target_size=512):

    height, width = frame.shape[:2]

    if height <= 0 or width <= 0:
        return None

    scale = min(
        target_size / width,
        target_size / height
    )

    new_width = max(
        1,
        int(round(width * scale))
    )

    new_height = max(
        1,
        int(round(height * scale))
    )

    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (
            target_size,
            target_size,
            3
        ),
        dtype=np.uint8
    )

    x_offset = (
        target_size - new_width
    ) // 2

    y_offset = (
        target_size - new_height
    ) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas


# =========================================================
# ROBUST SEQUENTIAL FRAME SAMPLING
# =========================================================

def sample_frames(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            "Could not open video."
        )

        return None

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:

        cap.release()

        return None

    target_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    )

    target_indices = np.round(
        target_indices
    ).astype(int)

    frames = []

    target_pointer = 0
    frame_index = 0
    last_frame = None

    while True:

        success, frame = cap.read()

        if not success:
            break

        last_frame = frame

        # ---------------------------------------------
        # Some short videos have repeated target indices.
        # Add the same decoded frame for every duplicate.
        # ---------------------------------------------

        while (
            target_pointer < SEQUENCE_LENGTH
            and
            target_indices[target_pointer]
            == frame_index
        ):

            processed_frame = (
                letterbox_frame(
                    frame,
                    INPUT_SIZE
                )
            )

            if processed_frame is None:

                cap.release()

                return None

            frames.append(
                processed_frame.copy()
            )

            target_pointer += 1

        if (
            target_pointer
            >= SEQUENCE_LENGTH
        ):
            break

        frame_index += 1

    cap.release()

    # -----------------------------------------------------
    # Fallback in case OpenCV reports slightly more frames
    # than it can actually decode.
    # -----------------------------------------------------

    if (
        len(frames) < SEQUENCE_LENGTH
        and
        last_frame is not None
    ):

        fallback = letterbox_frame(
            last_frame,
            INPUT_SIZE
        )

        while (
            len(frames)
            < SEQUENCE_LENGTH
        ):

            frames.append(
                fallback.copy()
            )

    if len(frames) != SEQUENCE_LENGTH:

        print(
            f"Sampling produced "
            f"{len(frames)} frames "
            f"instead of 32."
        )

        return None

    return frames


# =========================================================
# XYZ LANDMARK FEATURES
# =========================================================

def extract_xyz(
    landmarks,
    expected_count
):

    output = np.zeros(
        expected_count * 3,
        dtype=np.float32
    )

    if (
        landmarks is None
        or
        len(landmarks) == 0
    ):
        return output

    values = []

    for landmark in landmarks[
        :expected_count
    ]:

        values.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    values = np.array(
        values,
        dtype=np.float32
    )

    output[
        :len(values)
    ] = values

    return output


# =========================================================
# POSE FEATURES
# =========================================================

def extract_pose(
    landmarks
):

    output = np.zeros(
        33 * 4,
        dtype=np.float32
    )

    if (
        landmarks is None
        or
        len(landmarks) == 0
    ):
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

    output[
        :len(values)
    ] = values

    return output


# =========================================================
# LIP FEATURES
# =========================================================

def extract_lips(
    face_landmarks
):

    output = np.zeros(
        len(LIP_INDICES) * 3,
        dtype=np.float32
    )

    if (
        face_landmarks is None
        or
        len(face_landmarks) == 0
    ):
        return output

    values = []

    for index in LIP_INDICES:

        if index >= len(
            face_landmarks
        ):
            continue

        landmark = (
            face_landmarks[index]
        )

        values.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    values = np.array(
        values,
        dtype=np.float32
    )

    output[
        :len(values)
    ] = values

    return output


# =========================================================
# 382-D FEATURE VECTOR
# =========================================================

def create_feature_vector(
    result
):

    pose_detected = (
        len(
            result.pose_landmarks
        ) > 0
    )

    face_detected = (
        len(
            result.face_landmarks
        ) > 0
    )

    left_detected = (
        len(
            result.left_hand_landmarks
        ) > 0
    )

    right_detected = (
        len(
            result.right_hand_landmarks
        ) > 0
    )

    pose = extract_pose(
        result.pose_landmarks
    )

    left = extract_xyz(
        result.left_hand_landmarks,
        21
    )

    right = extract_xyz(
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

    features = np.concatenate([
        pose,
        left,
        right,
        lips,
        detection_mask
    ])

    return features.astype(
        np.float32
    )


# =========================================================
# PROCESS ONE VIDEO
# =========================================================

def process_video(
    video_path,
    landmarker
):

    frames = sample_frames(
        video_path
    )

    if frames is None:
        return None

    sequence = []

    for frame in frames:

        # Every frame should now be 512 × 512

        if frame.shape != (
            INPUT_SIZE,
            INPUT_SIZE,
            3
        ):

            print(
                f"Unexpected frame shape: "
                f"{frame.shape}"
            )

            return None

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb = np.ascontiguousarray(
            rgb
        )

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=rgb
        )

        result = landmarker.detect(
            mp_image
        )

        features = (
            create_feature_vector(
                result
            )
        )

        sequence.append(
            features
        )

    sequence = np.stack(
        sequence
    ).astype(np.float32)

    return sequence


# =========================================================
# IMAGE MODE OPTIONS
# =========================================================

options = HolisticLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.IMAGE,

    min_face_detection_confidence=0.5,

    min_face_landmarks_confidence=0.5,

    min_pose_detection_confidence=0.5,

    min_pose_landmarks_confidence=0.5,

    min_hand_landmarks_confidence=0.5,

    output_face_blendshapes=False,

    output_segmentation_mask=False
)


# =========================================================
# PREVIOUSLY FAILED VIDEOS
# =========================================================

test_videos = [

    "datasets/WLASL/videos/10112.mp4",

    "datasets/WLASL/videos/55348.mp4",

    "datasets/WLASL/videos/65092.mp4",

    "datasets/WLASL/videos/55340.mp4",

    "datasets/WLASL/videos/55341.mp4",

    "datasets/WLASL/videos/52867.mp4",

    "datasets/WLASL/videos/52860.mp4",

    "datasets/WLASL/videos/52862.mp4",

    "datasets/WLASL/videos/52863.mp4",

    "datasets/WLASL/videos/69422.mp4"
]


success_count = 0
failed_count = 0


# =========================================================
# TEST
# =========================================================

with HolisticLandmarker.create_from_options(
    options
) as landmarker:

    for number, video_path in enumerate(
        test_videos,
        start=1
    ):

        print(
            "\n" + "-" * 75
        )

        print(
            f"Test {number}/"
            f"{len(test_videos)}"
        )

        print(
            f"Video: "
            f"{video_path}"
        )

        try:

            features = process_video(
                video_path,
                landmarker
            )

            if features is None:

                print(
                    "FAILED: extraction "
                    "returned None"
                )

                failed_count += 1

                continue

            print(
                f"Shape: "
                f"{features.shape}"
            )

            print(
                f"NaN values: "
                f"{np.isnan(features).sum()}"
            )

            print(
                f"Infinite values: "
                f"{np.isinf(features).sum()}"
            )

            if (
                features.shape
                == (32, 382)
                and
                np.isnan(
                    features
                ).sum() == 0
                and
                np.isinf(
                    features
                ).sum() == 0
            ):

                print(
                    "SUCCESS"
                )

                success_count += 1

            else:

                print(
                    "FAILED: invalid "
                    "feature array"
                )

                failed_count += 1

        except Exception as error:

            print(
                f"FAILED: {error}"
            )

            failed_count += 1


print(
    "\n" + "=" * 75
)

print(
    "ROBUST IMAGE MODE TEST SUMMARY"
)

print(
    "=" * 75
)

print(
    f"Successful videos: "
    f"{success_count}"
)

print(
    f"Failed videos: "
    f"{failed_count}"
)

print(
    f"MediaPipe input size: "
    f"{INPUT_SIZE}x{INPUT_SIZE}"
)