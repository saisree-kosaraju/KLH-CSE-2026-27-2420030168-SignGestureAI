import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import re
import sys
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

SEQUENCE_LENGTH = 32
FEATURE_DIM = 382
INPUT_SIZE = 512

MODEL_PATH = Path(
    "models/holistic_landmarker.task"
)

WLASL_METADATA = Path(
    "metadata/wlasl300_expanded_3392.csv"
)

INCLUDE_METADATA = Path(
    "metadata/include_final.csv"
)

# IMPORTANT:
# New clean folders so old experimental features
# are never mixed with final features.

WLASL_OUTPUT = Path(
    "processed_data/WLASL300/landmarks_512_image"
)

INCLUDE_OUTPUT = Path(
    "processed_data/INCLUDE/landmarks_512_image"
)

WLASL_MANIFEST = Path(
    "metadata/wlasl300_landmark_manifest_expanded_3392.csv"
)

INCLUDE_MANIFEST = Path(
    "metadata/include_landmark_manifest_final.csv"
)


# =========================================================
# LIP LANDMARK INDICES
# =========================================================

LIP_INDICES = [
    0, 13, 14, 17, 37, 39, 40, 61,
    78, 80, 81, 82, 84, 87, 88, 91,
    95, 146, 178, 181, 185, 191,
    267, 269, 270, 291, 308, 310,
    311, 312, 314, 317, 318, 321,
    324, 375, 402, 405, 409, 415
]


# =========================================================
# MEDIAPIPE ALIASES
# =========================================================

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
# CREATE LANDMARKER
# =========================================================

def create_landmarker():

    options = HolisticLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(
                MODEL_PATH
            )
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

    return HolisticLandmarker.create_from_options(
        options
    )


# =========================================================
# LETTERBOX FRAME TO 512 × 512
# =========================================================

def letterbox_frame(
    frame,
    target_size=INPUT_SIZE
):

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

    # Better interpolation depending on direction
    if scale < 1:

        interpolation = (
            cv2.INTER_AREA
        )

    else:

        interpolation = (
            cv2.INTER_LINEAR
        )

    resized = cv2.resize(
        frame,
        (
            new_width,
            new_height
        ),
        interpolation=interpolation
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
        y_offset:
        y_offset + new_height,

        x_offset:
        x_offset + new_width
    ] = resized

    return canvas


# =========================================================
# ROBUST SEQUENTIAL 32-FRAME SAMPLING
# =========================================================

def sample_video_frames(
    video_path
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

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

    current_index = 0

    last_frame = None

    # -----------------------------------------------------
    # Sequential decoding
    # -----------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        last_frame = frame

        # Some short WLASL videos have repeated
        # target indices.
        while (
            target_pointer
            < SEQUENCE_LENGTH

            and

            target_indices[
                target_pointer
            ] == current_index
        ):

            prepared = letterbox_frame(
                frame,
                INPUT_SIZE
            )

            if prepared is None:

                cap.release()

                return None

            frames.append(
                prepared.copy()
            )

            target_pointer += 1

        if (
            target_pointer
            >= SEQUENCE_LENGTH
        ):
            break

        current_index += 1

    cap.release()

    # -----------------------------------------------------
    # Fallback for videos whose reported frame count
    # is slightly larger than decodable frame count.
    # -----------------------------------------------------

    if (
        len(frames)
        < SEQUENCE_LENGTH

        and

        last_frame is not None
    ):

        fallback = letterbox_frame(
            last_frame,
            INPUT_SIZE
        )

        if fallback is None:
            return None

        while (
            len(frames)
            < SEQUENCE_LENGTH
        ):

            frames.append(
                fallback.copy()
            )

    if (
        len(frames)
        != SEQUENCE_LENGTH
    ):

        return None

    return frames


# =========================================================
# EXTRACT XYZ
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

    values = np.asarray(
        values,
        dtype=np.float32
    )

    output[
        :len(values)
    ] = values

    return output


# =========================================================
# EXTRACT POSE
# =========================================================

def extract_pose(
    landmarks
):

    # 33 landmarks ×
    # x, y, z, visibility
    # = 132

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

    values = np.asarray(
        values,
        dtype=np.float32
    )

    output[
        :len(values)
    ] = values

    return output


# =========================================================
# EXTRACT LIPS
# =========================================================

def extract_lips(
    face_landmarks
):

    # 40 × xyz = 120

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

        if (
            index
            >= len(face_landmarks)
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

    values = np.asarray(
        values,
        dtype=np.float32
    )

    output[
        :len(values)
    ] = values

    return output


# =========================================================
# CREATE 382-D FEATURE VECTOR
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

    # 132
    pose = extract_pose(
        result.pose_landmarks
    )

    # 63
    left_hand = extract_xyz(
        result.left_hand_landmarks,
        21
    )

    # 63
    right_hand = extract_xyz(
        result.right_hand_landmarks,
        21
    )

    # 120
    lips = extract_lips(
        result.face_landmarks
    )

    # 4
    detection_mask = np.array(
        [
            float(pose_detected),
            float(face_detected),
            float(left_detected),
            float(right_detected)
        ],
        dtype=np.float32
    )

    # 132 + 63 + 63 + 120 + 4
    # = 382

    features = np.concatenate([
        pose,
        left_hand,
        right_hand,
        lips,
        detection_mask
    ])

    return features.astype(
        np.float32
    )


# =========================================================
# EXTRACT ONE VIDEO
# =========================================================

def extract_video(
    video_path,
    landmarker
):

    frames = sample_video_frames(
        video_path
    )

    if frames is None:

        return None

    sequence = []

    for frame in frames:

        # Every MediaPipe input must be identical size

        if frame.shape != (
            INPUT_SIZE,
            INPUT_SIZE,
            3
        ):

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

        feature_vector = (
            create_feature_vector(
                result
            )
        )

        sequence.append(
            feature_vector
        )

    sequence = np.stack(
        sequence
    ).astype(np.float32)

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURE_DIM
    ):

        return None

    if np.isnan(
        sequence
    ).any():

        return None

    if np.isinf(
        sequence
    ).any():

        return None

    return sequence


# =========================================================
# SAFE FILE NAME
# =========================================================

def safe_name(
    value
):

    return re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        str(value)
    )


# =========================================================
# PROCESS DATASET
# =========================================================

def process_dataset(
    dataset_name,
    metadata_path,
    output_dir,
    manifest_path
):

    print(
        "\n" + "=" * 75
    )

    print(
        f"EXTRACTING {dataset_name}"
    )

    print(
        "=" * 75
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(
        metadata_path,
        dtype={
            "video_id": str
        }
    )

    total = len(df)

    print(
        f"\nVideos: {total}"
    )

    print(
        f"Output: {output_dir}"
    )

    print(
        f"Input size: "
        f"{INPUT_SIZE}x{INPUT_SIZE}"
    )

    print(
        f"Sequence shape: "
        f"({SEQUENCE_LENGTH}, "
        f"{FEATURE_DIM})"
    )

    processed = 0
    skipped = 0
    failed = 0

    manifest = []

    # -----------------------------------------------------
    # One landmarker can now safely be reused because
    # every input is exactly 512 × 512.
    # -----------------------------------------------------

    with create_landmarker() as landmarker:

        for index, row in df.iterrows():

            video_path = (
                row["video_path"]
            )

            # =============================================
            # SAMPLE ID
            # =============================================

            if dataset_name == "WLASL300":

                video_id = str(
                    row["video_id"]
                ).zfill(5)

                sample_id = (
                    f"wlasl_{video_id}"
                )

            else:

                sample_id = (
                    f"include_{index:05d}"
                )

            sample_id = safe_name(
                sample_id
            )

            output_file = (
                output_dir
                /
                f"{sample_id}.npy"
            )

            # =============================================
            # RESTART / RESUME SUPPORT
            # =============================================

            if output_file.exists():

                try:

                    existing = np.load(
                        output_file,
                        mmap_mode="r"
                    )

                    if existing.shape == (
                        SEQUENCE_LENGTH,
                        FEATURE_DIM
                    ):

                        skipped += 1

                        manifest.append({

                            "sample_id":
                                sample_id,

                            "video_path":
                                video_path,

                            "feature_path":
                                str(
                                    output_file
                                ),

                            "status":
                                "existing",

                            "error":
                                "",

                            "split":
                                row.get(
                                    "split",
                                    ""
                                ),

                            "class_id":
                                row.get(
                                    "class_id",
                                    ""
                                ),

                            "label":
                                row.get(
                                    "class_name",
                                    row.get(
                                        "label",
                                        ""
                                    )
                                )
                        })

                        if (
                            (index + 1)
                            % 25 == 0
                            or
                            (index + 1)
                            == total
                        ):

                            pd.DataFrame(
                                manifest
                            ).to_csv(
                                manifest_path,
                                index=False
                            )

                            print(
                                f"Processed "
                                f"{index + 1}/"
                                f"{total}"
                                f" | New: "
                                f"{processed}"
                                f" | Skipped: "
                                f"{skipped}"
                                f" | Failed: "
                                f"{failed}"
                            )

                        continue

                except Exception:

                    # Invalid or corrupt npy:
                    # simply extract again.
                    pass

            # =============================================
            # EXTRACT
            # =============================================

            status = "failed"

            error_message = ""

            try:

                sequence = extract_video(
                    video_path,
                    landmarker
                )

                if sequence is None:

                    failed += 1

                    error_message = (
                        "Extraction returned None"
                    )

                else:

                    np.save(
                        output_file,
                        sequence
                    )

                    processed += 1

                    status = "success"

            except Exception as error:

                failed += 1

                error_message = str(
                    error
                )

            # =============================================
            # MANIFEST
            # =============================================

            manifest.append({

                "sample_id":
                    sample_id,

                "video_path":
                    video_path,

                "feature_path":
                    (
                        str(output_file)
                        if status
                        == "success"
                        else ""
                    ),

                "status":
                    status,

                "error":
                    error_message,

                "split":
                    row.get(
                        "split",
                        ""
                    ),

                "class_id":
                    row.get(
                        "class_id",
                        ""
                    ),

                "label":
                    row.get(
                        "class_name",
                        row.get(
                            "label",
                            ""
                        )
                    )
            })

            # =============================================
            # SAVE PROGRESS
            # =============================================

            if (
                (index + 1)
                % 25 == 0

                or

                (index + 1)
                == total
            ):

                pd.DataFrame(
                    manifest
                ).to_csv(
                    manifest_path,
                    index=False
                )

                print(
                    f"Processed "
                    f"{index + 1}/"
                    f"{total}"
                    f" | New: "
                    f"{processed}"
                    f" | Skipped: "
                    f"{skipped}"
                    f" | Failed: "
                    f"{failed}"
                )

    # =====================================================
    # FINAL MANIFEST
    # =====================================================

    manifest_df = pd.DataFrame(
        manifest
    )

    manifest_df.to_csv(
        manifest_path,
        index=False
    )

    print(
        "\n" + "-" * 75
    )

    print(
        f"{dataset_name} SUMMARY"
    )

    print(
        "-" * 75
    )

    print(
        f"Total videos: "
        f"{total}"
    )

    print(
        f"Newly extracted: "
        f"{processed}"
    )

    print(
        f"Already existing: "
        f"{skipped}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print(
        f"Manifest: "
        f"{manifest_path}"
    )


# =========================================================
# MAIN
# =========================================================

print(
    "=" * 75
)

print(
    "FINAL MEDIAPIPE LANDMARK EXTRACTION"
)

print(
    "=" * 75
)

print(
    f"\nInput frame size: "
    f"{INPUT_SIZE}x{INPUT_SIZE}"
)

print(
    f"Sequence length: "
    f"{SEQUENCE_LENGTH}"
)

print(
    f"Feature dimension: "
    f"{FEATURE_DIM}"
)

print(
    "Mode: MediaPipe IMAGE"
)

print(
    "Sampling: sequential + uniform"
)


# =========================================================
# COMMAND-LINE DATASET SELECTION
# =========================================================

if len(sys.argv) < 2:

    print(
        "\nPlease specify which dataset "
        "to process."
    )

    print(
        "\nUse one of:"
    )

    print(
        "python scripts/"
        "14_extract_all_landmarks.py "
        "WLASL300"
    )

    print(
        "python scripts/"
        "14_extract_all_landmarks.py "
        "INCLUDE"
    )

    sys.exit(0)


dataset_choice = (
    sys.argv[1]
    .strip()
    .upper()
)


# =========================================================
# WLASL300
# =========================================================

if dataset_choice == "WLASL300":

    process_dataset(
        "WLASL300",
        WLASL_METADATA,
        WLASL_OUTPUT,
        WLASL_MANIFEST
    )


# =========================================================
# INCLUDE
# =========================================================

elif dataset_choice == "INCLUDE":

    process_dataset(
        "INCLUDE",
        INCLUDE_METADATA,
        INCLUDE_OUTPUT,
        INCLUDE_MANIFEST
    )


else:

    print(
        f"\nUnknown dataset: "
        f"{dataset_choice}"
    )

    print(
        "Use WLASL300 or INCLUDE."
    )

    sys.exit(1)


print(
    "\n" + "=" * 75
)

print(
    "LANDMARK EXTRACTION COMPLETED"
)

print(
    "=" * 75
)