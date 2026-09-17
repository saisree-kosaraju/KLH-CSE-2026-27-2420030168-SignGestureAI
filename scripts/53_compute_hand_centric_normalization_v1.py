from pathlib import Path
import time

import numpy as np
import pandas as pd


# ============================================================
# STEP 53 - WLASL300 HAND-CENTRIC TRAIN-ONLY NORMALIZATION
# ============================================================

NUM_FRAMES = 32
TOTAL_DIM = 1047

EXPECTED_TRAIN = 2294
EXPECTED_VAL = 600
EXPECTED_TOTAL = 2894

PER_HAND_DIM = 162
CROSS_HAND_DIM = 24

STATIC_GEOMETRY_DIM = 348
MASK_DIM = 3

# ============================================================
# FEATURE LAYOUT FROM STEP 52
# ============================================================

# Static
STATIC_LEFT = slice(0, 162)
STATIC_RIGHT = slice(162, 324)
STATIC_CROSS = slice(324, 348)

# Detection masks
MASK_SLICE = slice(348, 351)

LEFT_MASK_INDEX = 348
RIGHT_MASK_INDEX = 349
BOTH_MASK_INDEX = 350

# Velocity
VELOCITY_START = 351

VELOCITY_LEFT = slice(
    VELOCITY_START,
    VELOCITY_START + 162
)  # 351:513

VELOCITY_RIGHT = slice(
    VELOCITY_START + 162,
    VELOCITY_START + 324
)  # 513:675

VELOCITY_CROSS = slice(
    VELOCITY_START + 324,
    VELOCITY_START + 348
)  # 675:699

# Acceleration
ACCELERATION_START = 699

ACCELERATION_LEFT = slice(
    ACCELERATION_START,
    ACCELERATION_START + 162
)  # 699:861

ACCELERATION_RIGHT = slice(
    ACCELERATION_START + 162,
    ACCELERATION_START + 324
)  # 861:1023

ACCELERATION_CROSS = slice(
    ACCELERATION_START + 324,
    ACCELERATION_START + 348
)  # 1023:1047


# ============================================================
# PATHS
# ============================================================

MANIFEST_PATH = Path(
    "metadata/wlasl300_hand_centric_manifest_v1.csv"
)

OUTPUT_PATH = Path(
    "metadata/wlasl300_hand_centric_normalization_v1.npz"
)

SUMMARY_PATH = Path(
    "results/wlasl300_hand_centric_normalization_v1_summary.txt"
)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("=" * 78)

print(
    "STEP 53 - WLASL300 HAND-CENTRIC TRAIN-ONLY NORMALIZATION V1"
)

print("=" * 78)


if not MANIFEST_PATH.exists():

    raise FileNotFoundError(
        f"Missing manifest:\n{MANIFEST_PATH}"
    )


# ============================================================
# LOAD MANIFEST
# ============================================================

df = pd.read_csv(
    MANIFEST_PATH,
    dtype={
        "video_id": str
    }
)


required_columns = [
    "video_id",
    "class_id",
    "split",
    "hand_centric_feature_path"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise RuntimeError(
        f"Missing columns: {missing_columns}"
    )


df["video_id"] = (
    df["video_id"]
    .astype(str)
    .str.strip()
    .str.zfill(5)
)


df["class_id"] = (
    df["class_id"]
    .astype(int)
)


df["split"] = (
    df["split"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# SPLIT CHECK
# ============================================================

train_df = (
    df[
        df["split"]
        ==
        "train"
    ]
    .copy()
    .reset_index(drop=True)
)


val_df = (
    df[
        df["split"]
        ==
        "val"
    ]
    .copy()
    .reset_index(drop=True)
)


test_df = (
    df[
        df["split"]
        ==
        "test"
    ]
    .copy()
    .reset_index(drop=True)
)


print("\nManifest:")

print(
    f"Train : {len(train_df)}"
)

print(
    f"Val   : {len(val_df)}"
)

print(
    f"Test  : {len(test_df)}"
)


if len(df) != EXPECTED_TOTAL:

    raise RuntimeError(
        f"Expected {EXPECTED_TOTAL} rows, "
        f"found {len(df)}."
    )


if len(train_df) != EXPECTED_TRAIN:

    raise RuntimeError(
        f"Expected {EXPECTED_TRAIN} train rows, "
        f"found {len(train_df)}."
    )


if len(val_df) != EXPECTED_VAL:

    raise RuntimeError(
        f"Expected {EXPECTED_VAL} validation rows, "
        f"found {len(val_df)}."
    )


if len(test_df) != 0:

    raise RuntimeError(
        "TEST DATA ENTERED STEP 53!"
    )


print(
    "\nNormalization statistics will use:"
)

print(
    f"TRAIN videos only: {len(train_df)} ✅"
)

print(
    "Validation videos used for statistics: 0 ✅"
)

print(
    "Test videos used for statistics: 0 ✅"
)


# ============================================================
# VERIFY ALL FEATURE FILES
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "VERIFYING STEP 52 FEATURE FILES"
)

print(
    "=" * 78
)


invalid_files = []


for index, row in df.iterrows():

    path = Path(
        row[
            "hand_centric_feature_path"
        ]
    )


    try:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing file: {path}"
            )


        features = np.load(
            path,
            mmap_mode="r"
        )


        if features.shape != (
            NUM_FRAMES,
            TOTAL_DIM
        ):

            raise RuntimeError(
                f"Incorrect shape: {features.shape}"
            )


        if not np.isfinite(
            features
        ).all():

            raise RuntimeError(
                "NaN/Inf found."
            )


    except Exception as error:

        invalid_files.append(
            {
                "video_id":
                    row[
                        "video_id"
                    ],

                "error":
                    str(
                        error
                    )
            }
        )


    processed = index + 1


    if (
        processed == 1
        or
        processed % 500 == 0
        or
        processed == len(df)
    ):

        print(
            f"Verified "
            f"{processed}/{len(df)}"
            f" | Invalid="
            f"{len(invalid_files)}"
        )


if invalid_files:

    print(
        "\nINVALID FILES:"
    )

    for item in invalid_files[:20]:

        print(
            item
        )


    raise RuntimeError(
        "Feature validation failed."
    )


print(
    "\nFeature verification: PASS ✅"
)


# ============================================================
# ACCUMULATORS
# ============================================================

feature_sum = np.zeros(
    TOTAL_DIM,
    dtype=np.float64
)


feature_sq_sum = np.zeros(
    TOTAL_DIM,
    dtype=np.float64
)


feature_count = np.zeros(
    TOTAL_DIM,
    dtype=np.int64
)


# ============================================================
# ACCUMULATION HELPER
# ============================================================

def accumulate_slice(
    features,
    feature_slice,
    valid_mask
):

    global feature_sum
    global feature_sq_sum
    global feature_count


    valid_count = int(
        valid_mask.sum()
    )


    if valid_count == 0:

        return


    values = (
        features[
            valid_mask,
            feature_slice
        ]
        .astype(
            np.float64
        )
    )


    feature_sum[
        feature_slice
    ] += values.sum(
        axis=0
    )


    feature_sq_sum[
        feature_slice
    ] += np.square(
        values
    ).sum(
        axis=0
    )


    feature_count[
        feature_slice
    ] += valid_count


# ============================================================
# TEMPORAL VALIDITY
# ============================================================

def velocity_valid(
    detected
):

    valid = np.zeros(
        NUM_FRAMES,
        dtype=bool
    )


    valid[1:] = (
        detected[1:]
        &
        detected[:-1]
    )


    return valid


def acceleration_valid(
    detected
):

    valid = np.zeros(
        NUM_FRAMES,
        dtype=bool
    )


    valid[2:] = (
        detected[2:]
        &
        detected[1:-1]
        &
        detected[:-2]
    )


    return valid


# ============================================================
# COMPUTE TRAIN-ONLY STATISTICS
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "COMPUTING TRAIN-ONLY NORMALIZATION STATISTICS"
)

print(
    "=" * 78
)


start_time = time.time()


left_frame_count = 0
right_frame_count = 0
both_frame_count = 0

left_velocity_count = 0
right_velocity_count = 0
both_velocity_count = 0

left_acceleration_count = 0
right_acceleration_count = 0
both_acceleration_count = 0


for index, row in train_df.iterrows():

    path = Path(
        row[
            "hand_centric_feature_path"
        ]
    )


    features = np.load(
        path
    ).astype(
        np.float32
    )


    # ========================================================
    # DETECTION MASKS
    # ========================================================

    left_detected = (
        features[
            :,
            LEFT_MASK_INDEX
        ]
        >
        0.5
    )


    right_detected = (
        features[
            :,
            RIGHT_MASK_INDEX
        ]
        >
        0.5
    )


    both_detected = (
        features[
            :,
            BOTH_MASK_INDEX
        ]
        >
        0.5
    )


    # ========================================================
    # STATIC FEATURES
    # ========================================================

    accumulate_slice(
        features,
        STATIC_LEFT,
        left_detected
    )


    accumulate_slice(
        features,
        STATIC_RIGHT,
        right_detected
    )


    accumulate_slice(
        features,
        STATIC_CROSS,
        both_detected
    )


    # ========================================================
    # TEMPORAL VALIDITY
    # ========================================================

    left_vel = velocity_valid(
        left_detected
    )


    right_vel = velocity_valid(
        right_detected
    )


    both_vel = velocity_valid(
        both_detected
    )


    left_acc = acceleration_valid(
        left_detected
    )


    right_acc = acceleration_valid(
        right_detected
    )


    both_acc = acceleration_valid(
        both_detected
    )


    # ========================================================
    # VELOCITY
    # ========================================================

    accumulate_slice(
        features,
        VELOCITY_LEFT,
        left_vel
    )


    accumulate_slice(
        features,
        VELOCITY_RIGHT,
        right_vel
    )


    accumulate_slice(
        features,
        VELOCITY_CROSS,
        both_vel
    )


    # ========================================================
    # ACCELERATION
    # ========================================================

    accumulate_slice(
        features,
        ACCELERATION_LEFT,
        left_acc
    )


    accumulate_slice(
        features,
        ACCELERATION_RIGHT,
        right_acc
    )


    accumulate_slice(
        features,
        ACCELERATION_CROSS,
        both_acc
    )


    # ========================================================
    # COUNTS FOR REPORT
    # ========================================================

    left_frame_count += int(
        left_detected.sum()
    )


    right_frame_count += int(
        right_detected.sum()
    )


    both_frame_count += int(
        both_detected.sum()
    )


    left_velocity_count += int(
        left_vel.sum()
    )


    right_velocity_count += int(
        right_vel.sum()
    )


    both_velocity_count += int(
        both_vel.sum()
    )


    left_acceleration_count += int(
        left_acc.sum()
    )


    right_acceleration_count += int(
        right_acc.sum()
    )


    both_acceleration_count += int(
        both_acc.sum()
    )


    processed = index + 1


    if (
        processed == 1
        or
        processed % 500 == 0
        or
        processed == len(train_df)
    ):

        elapsed = (
            time.time()
            -
            start_time
        )


        print(
            f"Processed "
            f"{processed}/{len(train_df)}"
            f" | Time="
            f"{elapsed:.1f}s"
        )


# ============================================================
# CALCULATE MEAN / STD
# ============================================================

mean = np.zeros(
    TOTAL_DIM,
    dtype=np.float32
)


std = np.ones(
    TOTAL_DIM,
    dtype=np.float32
)


valid_statistics = (
    feature_count
    >
    0
)


mean[
    valid_statistics
] = (

    feature_sum[
        valid_statistics
    ]

    /

    feature_count[
        valid_statistics
    ]

).astype(
    np.float32
)


variance = np.zeros(
    TOTAL_DIM,
    dtype=np.float64
)


variance[
    valid_statistics
] = (

    feature_sq_sum[
        valid_statistics
    ]

    /

    feature_count[
        valid_statistics
    ]

    -

    np.square(
        mean[
            valid_statistics
        ].astype(
            np.float64
        )
    )
)


variance = np.maximum(
    variance,
    0.0
)


computed_std = np.sqrt(
    variance
)


# Prevent division by tiny values.
computed_std[
    computed_std
    <
    1e-4
] = 1.0


std[
    valid_statistics
] = (
    computed_std[
        valid_statistics
    ]
    .astype(
        np.float32
    )
)


# ============================================================
# MASKS ARE NEVER NORMALIZED
# ============================================================

mean[
    MASK_SLICE
] = 0.0


std[
    MASK_SLICE
] = 1.0


feature_count[
    MASK_SLICE
] = (
    len(
        train_df
    )
    *
    NUM_FRAMES
)


# ============================================================
# SANITY CHECK
# ============================================================

if not np.isfinite(
    mean
).all():

    raise RuntimeError(
        "NaN/Inf in normalization mean."
    )


if not np.isfinite(
    std
).all():

    raise RuntimeError(
        "NaN/Inf in normalization std."
    )


if (
    std
    <=
    0
).any():

    raise RuntimeError(
        "Non-positive standard deviation."
    )


print(
    "\nNormalization arrays:"
)

print(
    f"Mean shape : {mean.shape}"
)

print(
    f"Std shape  : {std.shape}"
)

print(
    f"Count shape: {feature_count.shape}"
)


print(
    f"\nMinimum std: "
    f"{std.min():.8f}"
)

print(
    f"Maximum std: "
    f"{std.max():.8f}"
)


# ============================================================
# NORMALIZATION FUNCTION FOR VALIDATION TEST
# ============================================================

def normalize_features(
    features
):

    output = features.copy()


    left_detected = (
        features[
            :,
            LEFT_MASK_INDEX
        ]
        >
        0.5
    )


    right_detected = (
        features[
            :,
            RIGHT_MASK_INDEX
        ]
        >
        0.5
    )


    both_detected = (
        features[
            :,
            BOTH_MASK_INDEX
        ]
        >
        0.5
    )


    left_vel = velocity_valid(
        left_detected
    )


    right_vel = velocity_valid(
        right_detected
    )


    both_vel = velocity_valid(
        both_detected
    )


    left_acc = acceleration_valid(
        left_detected
    )


    right_acc = acceleration_valid(
        right_detected
    )


    both_acc = acceleration_valid(
        both_detected
    )


    groups = [
        (
            STATIC_LEFT,
            left_detected
        ),

        (
            STATIC_RIGHT,
            right_detected
        ),

        (
            STATIC_CROSS,
            both_detected
        ),

        (
            VELOCITY_LEFT,
            left_vel
        ),

        (
            VELOCITY_RIGHT,
            right_vel
        ),

        (
            VELOCITY_CROSS,
            both_vel
        ),

        (
            ACCELERATION_LEFT,
            left_acc
        ),

        (
            ACCELERATION_RIGHT,
            right_acc
        ),

        (
            ACCELERATION_CROSS,
            both_acc
        )
    ]


    for feature_slice, valid_mask in groups:

        if valid_mask.any():

            output[
                valid_mask,
                feature_slice
            ] = (

                output[
                    valid_mask,
                    feature_slice
                ]

                -

                mean[
                    feature_slice
                ]

            ) / std[
                feature_slice
            ]


        output[
            ~valid_mask,
            feature_slice
        ] = 0.0


    # Preserve binary masks exactly.
    output[
        :,
        MASK_SLICE
    ] = features[
        :,
        MASK_SLICE
    ]


    return output.astype(
        np.float32
    )


# ============================================================
# NORMALIZATION PRE-FLIGHT
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "NORMALIZATION PRE-FLIGHT TEST"
)

print(
    "=" * 78
)


sample_rows = []


# Test a few TRAIN and validation videos.
sample_rows.extend(
    [
        train_df.iloc[0],
        train_df.iloc[
            len(train_df) // 2
        ],
        train_df.iloc[-1]
    ]
)


sample_rows.extend(
    [
        val_df.iloc[0],
        val_df.iloc[
            len(val_df) // 2
        ],
        val_df.iloc[-1]
    ]
)


for row in sample_rows:

    path = Path(
        row[
            "hand_centric_feature_path"
        ]
    )


    features = np.load(
        path
    ).astype(
        np.float32
    )


    normalized = normalize_features(
        features
    )


    if normalized.shape != (
        NUM_FRAMES,
        TOTAL_DIM
    ):

        raise RuntimeError(
            "Normalized feature shape incorrect."
        )


    if not np.isfinite(
        normalized
    ).all():

        raise RuntimeError(
            "NaN/Inf after normalization."
        )


    if not np.array_equal(

        normalized[
            :,
            MASK_SLICE
        ],

        features[
            :,
            MASK_SLICE
        ]
    ):

        raise RuntimeError(
            "Detection masks changed "
            "during normalization."
        )


print(
    "Normalization pre-flight: PASS ✅"
)


# ============================================================
# SAVE NORMALIZATION
# ============================================================

np.savez(

    OUTPUT_PATH,

    mean=mean,

    std=std,

    count=feature_count,

    total_dim=np.array(
        TOTAL_DIM,
        dtype=np.int64
    ),

    sequence_length=np.array(
        NUM_FRAMES,
        dtype=np.int64
    ),

    static_geometry_dim=np.array(
        STATIC_GEOMETRY_DIM,
        dtype=np.int64
    ),

    mask_dim=np.array(
        MASK_DIM,
        dtype=np.int64
    ),

    velocity_start=np.array(
        VELOCITY_START,
        dtype=np.int64
    ),

    acceleration_start=np.array(
        ACCELERATION_START,
        dtype=np.int64
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

total_train_frames = (
    len(train_df)
    *
    NUM_FRAMES
)


elapsed = (
    time.time()
    -
    start_time
)


summary_lines = [

    "=" * 78,

    "STEP 53 HAND-CENTRIC NORMALIZATION SUMMARY",

    "=" * 78,

    "",

    f"Training videos used: "
    f"{len(train_df)}",

    f"Training frames: "
    f"{total_train_frames}",

    f"Validation videos used for statistics: 0",

    f"Test videos used for statistics: 0",

    "",

    f"Feature dimension: "
    f"{TOTAL_DIM}",

    f"Mean shape: "
    f"{mean.shape}",

    f"Std shape: "
    f"{std.shape}",

    "",

    f"Minimum std: "
    f"{std.min():.8f}",

    f"Maximum std: "
    f"{std.max():.8f}",

    "",

    "TRAIN detection-valid frames:",

    f"Left-hand static frames: "
    f"{left_frame_count}",

    f"Right-hand static frames: "
    f"{right_frame_count}",

    f"Both-hand static frames: "
    f"{both_frame_count}",

    "",

    f"Left-hand velocity frames: "
    f"{left_velocity_count}",

    f"Right-hand velocity frames: "
    f"{right_velocity_count}",

    f"Both-hand velocity frames: "
    f"{both_velocity_count}",

    "",

    f"Left-hand acceleration frames: "
    f"{left_acceleration_count}",

    f"Right-hand acceleration frames: "
    f"{right_acceleration_count}",

    f"Both-hand acceleration frames: "
    f"{both_acceleration_count}",

    "",

    f"Processing time: "
    f"{elapsed:.2f} seconds",

    "",

    "Normalization file:",

    str(
        OUTPUT_PATH
    ),

    "",

    "Summary file:",

    str(
        SUMMARY_PATH
    ),

    "",

    "TRAIN used for statistics: YES ✅",

    "VALIDATION used for statistics: NO ✅",

    "TEST used for statistics: NO ✅"
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


print(
    "\nSTATUS: STEP 53 "
    "TRAIN-ONLY NORMALIZATION COMPLETED ✅"
)

print("=" * 78)