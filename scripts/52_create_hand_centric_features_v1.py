from pathlib import Path
import time

import numpy as np
import pandas as pd


# ============================================================
# STEP 52 - WLASL300 HAND-CENTRIC FEATURE EXTRACTION V1
# ============================================================

NUM_FRAMES = 32
RAW_DIM = 382

EXPECTED_TRAIN = 2294
EXPECTED_VAL = 600
EXPECTED_TOTAL = 2894

LEFT_HAND_SLICE = slice(132, 195)
RIGHT_HAND_SLICE = slice(195, 258)

LEFT_MASK_INDEX = 380
RIGHT_MASK_INDEX = 381

HAND_LANDMARKS = 21

# Per-hand feature size:
# normalized coordinates       = 63
# bone vectors                 = 60
# joint angles                 = 15
# fingertip pair distances     = 10
# fingertip->wrist distances   = 5
# fingertip->palm distances    = 5
# palm normal                  = 3
# openness                     = 1
# TOTAL                        = 162
PER_HAND_DIM = 162

# Cross-hand:
# wrist relative xyz           = 3
# wrist distance               = 1
# corresponding tip relative   = 15
# corresponding tip distance   = 5
# TOTAL                        = 24
CROSS_HAND_DIM = 24

STATIC_GEOMETRY_DIM = (
    PER_HAND_DIM
    + PER_HAND_DIM
    + CROSS_HAND_DIM
)  # 348

MASK_DIM = 3  # left, right, both

STATIC_WITH_MASKS_DIM = (
    STATIC_GEOMETRY_DIM
    + MASK_DIM
)  # 351

VELOCITY_DIM = STATIC_GEOMETRY_DIM
ACCELERATION_DIM = STATIC_GEOMETRY_DIM

TOTAL_FEATURE_DIM = (
    STATIC_WITH_MASKS_DIM
    + VELOCITY_DIM
    + ACCELERATION_DIM
)  # 1047


# ============================================================
# MEDIAPIPE HAND TOPOLOGY
# ============================================================

FINGERTIPS = [4, 8, 12, 16, 20]

MCP_IDS = [5, 9, 13, 17]

BONE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),

    (0, 5, 6), (5, 6, 7), (6, 7, 8),

    (0, 9, 10), (9, 10, 11), (10, 11, 12),

    (0, 13, 14), (13, 14, 15), (14, 15, 16),

    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]


# ============================================================
# PATHS
# ============================================================

INPUT_MANIFEST = Path(
    "metadata/wlasl300_hand_rescue_manifest_v1.csv"
)

OUTPUT_DIR = Path(
    "processed_data/WLASL300/hand_centric_features_v1"
)

OUTPUT_MANIFEST = Path(
    "metadata/wlasl300_hand_centric_manifest_v1.csv"
)

SUMMARY_PATH = Path(
    "results/wlasl300_hand_centric_features_v1_summary.txt"
)

FAILED_PATH = Path(
    "results/wlasl300_hand_centric_features_v1_failed.csv"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

EPS = 1e-6


def safe_norm(vector):

    return float(
        np.linalg.norm(
            vector
        )
    )


def safe_unit(vector):

    norm = np.linalg.norm(
        vector
    )

    if norm < EPS:

        return np.zeros_like(
            vector,
            dtype=np.float32
        )

    return (
        vector / norm
    ).astype(
        np.float32
    )


def angle_at(
    a,
    b,
    c
):

    v1 = a - b
    v2 = c - b

    n1 = np.linalg.norm(
        v1
    )

    n2 = np.linalg.norm(
        v2
    )


    if (
        n1 < EPS
        or
        n2 < EPS
    ):

        return 0.0


    cosine = float(
        np.dot(
            v1,
            v2
        )
        /
        (
            n1
            *
            n2
        )
    )


    cosine = float(
        np.clip(
            cosine,
            -1.0,
            1.0
        )
    )


    # Normalize angle to 0-1
    return float(
        np.arccos(
            cosine
        )
        /
        np.pi
    )


# ============================================================
# NORMALIZE ONE HAND
# ============================================================

def normalize_hand(
    hand_xyz
):

    # Wrist-centered
    centered = (

        hand_xyz

        -

        hand_xyz[
            0:1
        ]

    ).astype(
        np.float32
    )


    # Scale using wrist -> MCP distances
    scales = [

        np.linalg.norm(
            centered[
                mcp_id
            ]
        )

        for mcp_id
        in MCP_IDS
    ]


    scale = float(
        np.mean(
            scales
        )
    )


    if (
        not np.isfinite(
            scale
        )
        or
        scale < EPS
    ):

        scale = 1.0


    normalized = (

        centered

        /

        scale

    ).astype(
        np.float32
    )


    normalized = np.clip(

        normalized,

        -8.0,

        8.0
    )


    return normalized


# ============================================================
# FEATURES FOR ONE HAND
# ============================================================

def hand_features(
    hand_xyz
):

    normalized = normalize_hand(
        hand_xyz
    )


    features = []


    # ========================================================
    # 1. NORMALIZED LANDMARKS
    # 21 x 3 = 63
    # ========================================================

    features.extend(

        normalized
        .reshape(
            -1
        )
        .tolist()
    )


    # ========================================================
    # 2. BONE VECTORS
    # 20 x 3 = 60
    # ========================================================

    for (
        start_id,
        end_id
    ) in BONE_EDGES:

        vector = (

            normalized[
                end_id
            ]

            -

            normalized[
                start_id
            ]
        )


        features.extend(
            vector.tolist()
        )


    # ========================================================
    # 3. FINGER JOINT ANGLES
    # 15
    # ========================================================

    for (
        a,
        b,
        c
    ) in ANGLE_TRIPLETS:

        features.append(

            angle_at(

                normalized[
                    a
                ],

                normalized[
                    b
                ],

                normalized[
                    c
                ]
            )
        )


    # ========================================================
    # 4. FINGERTIP PAIRWISE DISTANCES
    # 10
    # ========================================================

    for i in range(
        len(
            FINGERTIPS
        )
    ):

        for j in range(
            i + 1,
            len(
                FINGERTIPS
            )
        ):

            features.append(

                safe_norm(

                    normalized[
                        FINGERTIPS[
                            i
                        ]
                    ]

                    -

                    normalized[
                        FINGERTIPS[
                            j
                        ]
                    ]
                )
            )


    # ========================================================
    # 5. FINGERTIP -> WRIST DISTANCES
    # 5
    # ========================================================

    for tip_id in FINGERTIPS:

        features.append(

            safe_norm(

                normalized[
                    tip_id
                ]

                -

                normalized[
                    0
                ]
            )
        )


    # ========================================================
    # 6. FINGERTIP -> PALM CENTER DISTANCES
    # 5
    # ========================================================

    palm_ids = [
        0,
        5,
        9,
        13,
        17
    ]


    palm_center = np.mean(

        normalized[
            palm_ids
        ],

        axis=0
    )


    for tip_id in FINGERTIPS:

        features.append(

            safe_norm(

                normalized[
                    tip_id
                ]

                -

                palm_center
            )
        )


    # ========================================================
    # 7. PALM ORIENTATION
    # 3
    # ========================================================

    palm_vector_a = (

        normalized[
            5
        ]

        -

        normalized[
            0
        ]
    )


    palm_vector_b = (

        normalized[
            17
        ]

        -

        normalized[
            0
        ]
    )


    palm_normal = safe_unit(

        np.cross(

            palm_vector_a,

            palm_vector_b
        )
    )


    features.extend(
        palm_normal.tolist()
    )


    # ========================================================
    # 8. HAND OPENNESS
    # 1
    # ========================================================

    openness = float(

        np.mean(

            [

                safe_norm(
                    normalized[
                        tip_id
                    ]
                )

                for tip_id
                in FINGERTIPS
            ]
        )
    )


    features.append(
        openness
    )


    result = np.asarray(

        features,

        dtype=np.float32
    )


    if result.shape != (
        PER_HAND_DIM,
    ):

        raise RuntimeError(

            f"Per-hand feature shape "
            f"is {result.shape}, "
            f"expected ({PER_HAND_DIM},)."
        )


    return result


# ============================================================
# CROSS-HAND FEATURES
# ============================================================

def cross_hand_features(
    left_xyz,
    right_xyz
):

    features = []


    left_wrist = left_xyz[
        0
    ]

    right_wrist = right_xyz[
        0
    ]


    wrist_relative = (

        right_wrist

        -

        left_wrist
    )


    # Wrist relative xyz
    features.extend(
        wrist_relative.tolist()
    )


    # Wrist distance
    features.append(

        safe_norm(
            wrist_relative
        )
    )


    # Corresponding fingertip relative xyz
    for tip_id in FINGERTIPS:

        relative = (

            right_xyz[
                tip_id
            ]

            -

            left_xyz[
                tip_id
            ]
        )


        features.extend(
            relative.tolist()
        )


    # Corresponding fingertip distances
    for tip_id in FINGERTIPS:

        features.append(

            safe_norm(

                right_xyz[
                    tip_id
                ]

                -

                left_xyz[
                    tip_id
                ]
            )
        )


    result = np.asarray(

        features,

        dtype=np.float32
    )


    if result.shape != (
        CROSS_HAND_DIM,
    ):

        raise RuntimeError(

            f"Cross-hand feature shape "
            f"is {result.shape}, "
            f"expected ({CROSS_HAND_DIM},)."
        )


    return result


# ============================================================
# STATIC FEATURES FOR ONE FRAME
# ============================================================

def frame_static_geometry(
    raw_frame
):

    left_detected = bool(

        raw_frame[
            LEFT_MASK_INDEX
        ]

        >
        0.5
    )


    right_detected = bool(

        raw_frame[
            RIGHT_MASK_INDEX
        ]

        >
        0.5
    )


    both_detected = (

        left_detected

        and

        right_detected
    )


    left_features = np.zeros(

        PER_HAND_DIM,

        dtype=np.float32
    )


    right_features = np.zeros(

        PER_HAND_DIM,

        dtype=np.float32
    )


    cross_features = np.zeros(

        CROSS_HAND_DIM,

        dtype=np.float32
    )


    left_xyz = None
    right_xyz = None


    if left_detected:

        left_xyz = (

            raw_frame[
                LEFT_HAND_SLICE
            ]

            .reshape(
                HAND_LANDMARKS,
                3
            )

            .astype(
                np.float32
            )
        )


        left_features = hand_features(
            left_xyz
        )


    if right_detected:

        right_xyz = (

            raw_frame[
                RIGHT_HAND_SLICE
            ]

            .reshape(
                HAND_LANDMARKS,
                3
            )

            .astype(
                np.float32
            )
        )


        right_features = hand_features(
            right_xyz
        )


    if both_detected:

        cross_features = (
            cross_hand_features(

                left_xyz,

                right_xyz
            )
        )


    geometry = np.concatenate(

        [
            left_features,
            right_features,
            cross_features
        ],

        axis=0
    ).astype(
        np.float32
    )


    if geometry.shape != (
        STATIC_GEOMETRY_DIM,
    ):

        raise RuntimeError(

            f"Static geometry shape "
            f"is {geometry.shape}, "
            f"expected "
            f"({STATIC_GEOMETRY_DIM},)."
        )


    return (

        geometry,

        left_detected,

        right_detected,

        both_detected
    )


# ============================================================
# CREATE 32-FRAME HAND-CENTRIC FEATURES
# ============================================================

def create_sequence_features(
    rescued_landmarks
):

    static_geometry = np.zeros(

        (
            NUM_FRAMES,
            STATIC_GEOMETRY_DIM
        ),

        dtype=np.float32
    )


    masks = np.zeros(

        (
            NUM_FRAMES,
            MASK_DIM
        ),

        dtype=np.float32
    )


    left_valid = np.zeros(

        NUM_FRAMES,

        dtype=bool
    )


    right_valid = np.zeros(

        NUM_FRAMES,

        dtype=bool
    )


    both_valid = np.zeros(

        NUM_FRAMES,

        dtype=bool
    )


    # ========================================================
    # STATIC FEATURES
    # ========================================================

    for frame_index in range(
        NUM_FRAMES
    ):

        (
            geometry,
            left_detected,
            right_detected,
            both_detected

        ) = frame_static_geometry(

            rescued_landmarks[
                frame_index
            ]
        )


        static_geometry[
            frame_index
        ] = geometry


        left_valid[
            frame_index
        ] = left_detected


        right_valid[
            frame_index
        ] = right_detected


        both_valid[
            frame_index
        ] = both_detected


        masks[
            frame_index
        ] = [

            float(
                left_detected
            ),

            float(
                right_detected
            ),

            float(
                both_detected
            )
        ]


    # ========================================================
    # TEMPORAL MOTION
    # ========================================================

    velocity = np.zeros_like(
        static_geometry
    )


    acceleration = np.zeros_like(
        static_geometry
    )


    left_slice = slice(

        0,

        PER_HAND_DIM
    )


    right_slice = slice(

        PER_HAND_DIM,

        PER_HAND_DIM * 2
    )


    cross_slice = slice(

        PER_HAND_DIM * 2,

        STATIC_GEOMETRY_DIM
    )


    # ========================================================
    # VELOCITY
    # ========================================================

    for t in range(
        1,
        NUM_FRAMES
    ):

        if (
            left_valid[
                t
            ]

            and

            left_valid[
                t - 1
            ]
        ):

            velocity[
                t,
                left_slice
            ] = (

                static_geometry[
                    t,
                    left_slice
                ]

                -

                static_geometry[
                    t - 1,
                    left_slice
                ]
            )


        if (
            right_valid[
                t
            ]

            and

            right_valid[
                t - 1
            ]
        ):

            velocity[
                t,
                right_slice
            ] = (

                static_geometry[
                    t,
                    right_slice
                ]

                -

                static_geometry[
                    t - 1,
                    right_slice
                ]
            )


        if (
            both_valid[
                t
            ]

            and

            both_valid[
                t - 1
            ]
        ):

            velocity[
                t,
                cross_slice
            ] = (

                static_geometry[
                    t,
                    cross_slice
                ]

                -

                static_geometry[
                    t - 1,
                    cross_slice
                ]
            )


    # ========================================================
    # ACCELERATION
    # ========================================================

    for t in range(
        2,
        NUM_FRAMES
    ):

        if (
            left_valid[
                t
            ]

            and

            left_valid[
                t - 1
            ]

            and

            left_valid[
                t - 2
            ]
        ):

            acceleration[
                t,
                left_slice
            ] = (

                velocity[
                    t,
                    left_slice
                ]

                -

                velocity[
                    t - 1,
                    left_slice
                ]
            )


        if (
            right_valid[
                t
            ]

            and

            right_valid[
                t - 1
            ]

            and

            right_valid[
                t - 2
            ]
        ):

            acceleration[
                t,
                right_slice
            ] = (

                velocity[
                    t,
                    right_slice
                ]

                -

                velocity[
                    t - 1,
                    right_slice
                ]
            )


        if (
            both_valid[
                t
            ]

            and

            both_valid[
                t - 1
            ]

            and

            both_valid[
                t - 2
            ]
        ):

            acceleration[
                t,
                cross_slice
            ] = (

                velocity[
                    t,
                    cross_slice
                ]

                -

                velocity[
                    t - 1,
                    cross_slice
                ]
            )


    # ========================================================
    # FINAL FEATURE VECTOR
    # ========================================================

    output = np.concatenate(

        [
            static_geometry,
            masks,
            velocity,
            acceleration
        ],

        axis=1
    ).astype(
        np.float32
    )


    if output.shape != (

        NUM_FRAMES,

        TOTAL_FEATURE_DIM
    ):

        raise RuntimeError(

            f"Final feature shape "
            f"is {output.shape}, "
            f"expected "
            f"({NUM_FRAMES}, "
            f"{TOTAL_FEATURE_DIM})."
        )


    if not np.isfinite(
        output
    ).all():

        raise RuntimeError(
            "NaN/Inf found."
        )


    return output


# ============================================================
# LOAD STEP 51 MANIFEST
# ============================================================

print("=" * 78)

print(
    "STEP 52 - WLASL300 HAND-CENTRIC FEATURE EXTRACTION V1"
)

print("=" * 78)


if not INPUT_MANIFEST.exists():

    raise FileNotFoundError(

        f"Missing Step 51 manifest:\n"
        f"{INPUT_MANIFEST}"
    )


df = pd.read_csv(

    INPUT_MANIFEST,

    dtype={
        "video_id":
            str
    }
)


required_columns = [

    "video_id",

    "class_id",

    "split",

    "rescued_landmark_path"
]


missing_columns = [

    column

    for column
    in required_columns

    if column
    not in df.columns
]


if missing_columns:

    raise RuntimeError(

        f"Missing columns: "
        f"{missing_columns}"
    )


df["video_id"] = (

    df[
        "video_id"
    ]

    .astype(
        str
    )

    .str.strip()

    .str.zfill(
        5
    )
)


df["class_id"] = (

    df[
        "class_id"
    ]

    .astype(
        int
    )
)


df["split"] = (

    df[
        "split"
    ]

    .astype(
        str
    )

    .str.strip()

    .str.lower()
)


# ============================================================
# SPLIT SAFETY
# ============================================================

train_count = int(

    (
        df[
            "split"
        ]
        ==
        "train"
    ).sum()
)


val_count = int(

    (
        df[
            "split"
        ]
        ==
        "val"
    ).sum()
)


test_count = int(

    (
        df[
            "split"
        ]
        ==
        "test"
    ).sum()
)


print(
    "\nInput manifest:"
)

print(
    f"Train : {train_count}"
)

print(
    f"Val   : {val_count}"
)

print(
    f"Test  : {test_count}"
)


if train_count != EXPECTED_TRAIN:

    raise RuntimeError(

        f"Expected "
        f"{EXPECTED_TRAIN} "
        f"train videos, "
        f"found {train_count}."
    )


if val_count != EXPECTED_VAL:

    raise RuntimeError(

        f"Expected "
        f"{EXPECTED_VAL} "
        f"validation videos, "
        f"found {val_count}."
    )


if test_count != 0:

    raise RuntimeError(
        "TEST DATA ENTERED STEP 52!"
    )


if len(
    df
) != EXPECTED_TOTAL:

    raise RuntimeError(

        f"Expected "
        f"{EXPECTED_TOTAL} rows, "
        f"found {len(df)}."
    )


if df[
    "video_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate video IDs detected."
    )


print(
    "Test videos processed: 0 ✅"
)


print(
    f"\nOutput feature shape: "
    f"({NUM_FRAMES}, "
    f"{TOTAL_FEATURE_DIM})"
)


print(
    f"Static geometry dim : "
    f"{STATIC_GEOMETRY_DIM}"
)


print(
    f"Detection mask dim  : "
    f"{MASK_DIM}"
)


print(
    f"Velocity dim        : "
    f"{VELOCITY_DIM}"
)


print(
    f"Acceleration dim    : "
    f"{ACCELERATION_DIM}"
)


# ============================================================
# MAIN EXTRACTION
# ============================================================

rows = []

failed = []

start_time = time.time()


print(
    "\n" + "=" * 78
)

print(
    "STARTING HAND-CENTRIC FEATURE CREATION"
)

print(
    "=" * 78
)


for index, row in df.iterrows():

    video_id = row[
        "video_id"
    ]


    source_path = Path(

        row[
            "rescued_landmark_path"
        ]
    )


    output_path = (

        OUTPUT_DIR

        /

        f"{video_id}.npy"
    )


    try:

        if not source_path.exists():

            raise FileNotFoundError(

                f"Missing rescued landmarks: "
                f"{source_path}"
            )


        # =====================================================
        # RESUME SUPPORT
        # =====================================================

        if output_path.exists():

            features = np.load(

                output_path

            ).astype(
                np.float32
            )


            if features.shape != (

                NUM_FRAMES,

                TOTAL_FEATURE_DIM
            ):

                raise RuntimeError(

                    f"Existing output "
                    f"has shape "
                    f"{features.shape}"
                )


            if not np.isfinite(
                features
            ).all():

                raise RuntimeError(

                    "Existing output "
                    "contains NaN/Inf."
                )


            reused = True


        else:

            rescued = np.load(

                source_path

            ).astype(
                np.float32
            )


            if rescued.shape != (

                NUM_FRAMES,

                RAW_DIM
            ):

                raise RuntimeError(

                    f"Rescued landmark "
                    f"shape is "
                    f"{rescued.shape}"
                )


            if not np.isfinite(
                rescued
            ).all():

                raise RuntimeError(

                    "Rescued landmarks "
                    "contain NaN/Inf."
                )


            features = (
                create_sequence_features(
                    rescued
                )
            )


            np.save(

                output_path,

                features.astype(
                    np.float32
                )
            )


            reused = False


        rows.append(
            {

                "video_id":
                    video_id,

                "class_id":
                    int(
                        row[
                            "class_id"
                        ]
                    ),

                "split":
                    row[
                        "split"
                    ],

                "rescued_landmark_path":
                    str(
                        source_path
                    ),

                "hand_centric_feature_path":
                    str(
                        output_path
                    ),

                "feature_frames":
                    NUM_FRAMES,

                "feature_dim":
                    TOTAL_FEATURE_DIM,

                "reused":
                    reused
            }
        )


    except Exception as error:

        failed.append(
            {

                "video_id":
                    video_id,

                "split":
                    row[
                        "split"
                    ],

                "position":
                    index + 1,

                "error":
                    str(
                        error
                    )
            }
        )


        print(

            f"\nFAILED "
            f"{index + 1}/"
            f"{len(df)} "

            f"| Video="
            f"{video_id} "

            f"| Error="
            f"{error}"
        )


    processed = (
        index
        +
        1
    )


    if (

        processed
        ==
        1

        or

        processed % 100
        ==
        0

        or

        processed
        ==
        len(
            df
        )
    ):

        elapsed = (

            time.time()

            -

            start_time
        )


        print(

            f"Processed "
            f"{processed}/"
            f"{len(df)} "

            f"| Success="
            f"{len(rows)} "

            f"| Failed="
            f"{len(failed)} "

            f"| Time="
            f"{elapsed:.1f}s"
        )


    if (

        processed % 100
        ==
        0

        or

        processed
        ==
        len(
            df
        )
    ):

        pd.DataFrame(
            rows
        ).to_csv(

            OUTPUT_MANIFEST,

            index=False
        )


        if failed:

            pd.DataFrame(
                failed
            ).to_csv(

                FAILED_PATH,

                index=False
            )


# ============================================================
# FINAL VALIDATION
# ============================================================

result_df = pd.DataFrame(
    rows
)


result_df.to_csv(

    OUTPUT_MANIFEST,

    index=False
)


if failed:

    pd.DataFrame(
        failed
    ).to_csv(

        FAILED_PATH,

        index=False
    )


generated_train = int(

    (
        result_df[
            "split"
        ]
        ==
        "train"
    ).sum()

) if len(
    result_df
) else 0


generated_val = int(

    (
        result_df[
            "split"
        ]
        ==
        "val"
    ).sum()

) if len(
    result_df
) else 0


generated_test = int(

    (
        result_df[
            "split"
        ]
        ==
        "test"
    ).sum()

) if len(
    result_df
) else 0


elapsed = (

    time.time()

    -

    start_time
)


summary_lines = [

    "=" * 78,

    "STEP 52 HAND-CENTRIC FEATURE SUMMARY",

    "=" * 78,

    "",

    f"Requested videos: "
    f"{len(df)}",

    f"Successfully processed: "
    f"{len(result_df)}",

    f"Failed videos: "
    f"{len(failed)}",

    "",

    f"Train videos: "
    f"{generated_train}",

    f"Validation videos: "
    f"{generated_val}",

    f"Test videos: "
    f"{generated_test}",

    "",

    f"Sequence length: "
    f"{NUM_FRAMES}",

    f"Per-hand feature dim: "
    f"{PER_HAND_DIM}",

    f"Cross-hand feature dim: "
    f"{CROSS_HAND_DIM}",

    f"Static geometry dim: "
    f"{STATIC_GEOMETRY_DIM}",

    f"Detection mask dim: "
    f"{MASK_DIM}",

    f"Velocity dim: "
    f"{VELOCITY_DIM}",

    f"Acceleration dim: "
    f"{ACCELERATION_DIM}",

    f"Final feature dim: "
    f"{TOTAL_FEATURE_DIM}",

    f"Final feature shape: "
    f"({NUM_FRAMES}, "
    f"{TOTAL_FEATURE_DIM})",

    "",

    f"Total processing time: "
    f"{elapsed:.2f} seconds",

    "",

    "Output folder:",

    str(
        OUTPUT_DIR
    ),

    "",

    "Manifest:",

    str(
        OUTPUT_MANIFEST
    ),

    "",

    "Summary:",

    str(
        SUMMARY_PATH
    ),
]


with open(

    SUMMARY_PATH,

    "w",

    encoding="utf-8"

) as file:

    file.write(

        "\n".join(
            summary_lines
        )
    )


print(

    "\n"

    +

    "\n".join(
        summary_lines
    )
)


if (

    len(
        result_df
    )
    ==
    EXPECTED_TOTAL

    and

    len(
        failed
    )
    ==
    0

    and

    generated_train
    ==
    EXPECTED_TRAIN

    and

    generated_val
    ==
    EXPECTED_VAL

    and

    generated_test
    ==
    0
):

    print(

        "\nSTATUS: STEP 52 "
        "HAND-CENTRIC FEATURES "
        "COMPLETED ✅"
    )


else:

    print(

        "\nSTATUS: STEP 52 finished, "
        "but review counts/failures."
    )


print("=" * 78)