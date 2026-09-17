import sys
import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# FEATURE LAYOUT
# =========================================================

# Current feature vector:
#
# Pose       = 132 features     [0:132]
# Left hand  = 63 features      [132:195]
# Right hand = 63 features      [195:258]
# Lips       = 120 features     [258:378]
# Mask       = 4 features       [378:382]
#
# Total = 382

FEATURE_DIM = 382
MASK_START = 378

GROUPS = {

    "pose": {
        "start": 0,
        "end": 132,
        "mask_index": 0
    },

    "left_hand": {
        "start": 132,
        "end": 195,
        "mask_index": 2
    },

    "right_hand": {
        "start": 195,
        "end": 258,
        "mask_index": 3
    },

    "lips": {
        "start": 258,
        "end": 378,
        "mask_index": 1
    }
}


# =========================================================
# DATASET CONFIGURATION
# =========================================================

DATASETS = {

    "WLASL300": {

        "manifest": Path(
            "metadata/wlasl300_landmark_manifest_final.csv"
        ),

        "stats": Path(
            "metadata/wlasl300_landmark_normalization.npz"
        )
    },

    "INCLUDE": {

        "manifest": Path(
            "metadata/include_landmark_manifest_final.csv"
        ),

        "stats": Path(
            "metadata/include_landmark_normalization.npz"
        )
    }
}


# =========================================================
# ARGUMENT CHECK
# =========================================================

if len(sys.argv) < 2:

    print(
        "Please specify dataset."
    )

    print()

    print(
        "Use:"
    )

    print(
        "python scripts/"
        "17_compute_landmark_normalization.py "
        "WLASL300"
    )

    print(
        "python scripts/"
        "17_compute_landmark_normalization.py "
        "INCLUDE"
    )

    sys.exit(0)


dataset_name = (
    sys.argv[1]
    .strip()
    .upper()
)


if dataset_name not in DATASETS:

    print(
        f"Unknown dataset: {dataset_name}"
    )

    print(
        "Use WLASL300 or INCLUDE."
    )

    sys.exit(1)


manifest_path = (
    DATASETS[dataset_name][
        "manifest"
    ]
)

stats_path = (
    DATASETS[dataset_name][
        "stats"
    ]
)


# =========================================================
# START
# =========================================================

print(
    "=" * 75
)

print(
    f"COMPUTING {dataset_name} "
    f"LANDMARK NORMALIZATION"
)

print(
    "=" * 75
)


# =========================================================
# LOAD MANIFEST
# =========================================================

if not manifest_path.exists():

    print(
        f"ERROR: Manifest not found:\n"
        f"{manifest_path}"
    )

    sys.exit(1)


manifest = pd.read_csv(
    manifest_path
)


print(
    f"\nManifest rows: "
    f"{len(manifest)}"
)


# =========================================================
# TRAINING DATA ONLY
# =========================================================

train_df = manifest[
    manifest["split"]
    .astype(str)
    .str.lower()
    == "train"
].copy()


print(
    f"Training rows: "
    f"{len(train_df)}"
)


if len(train_df) == 0:

    print(
        "ERROR: No training samples found."
    )

    sys.exit(1)


# =========================================================
# ACCUMULATORS
# =========================================================

feature_sum = np.zeros(
    FEATURE_DIM,
    dtype=np.float64
)

feature_squared_sum = np.zeros(
    FEATURE_DIM,
    dtype=np.float64
)

feature_count = np.zeros(
    FEATURE_DIM,
    dtype=np.int64
)


# Detection counts for debugging
detection_counts = {

    "pose": 0,
    "face": 0,
    "left_hand": 0,
    "right_hand": 0
}

total_training_frames = 0


# =========================================================
# PROCESS TRAINING ARRAYS
# =========================================================

for number, (_, row) in enumerate(
    train_df.iterrows(),
    start=1
):

    feature_path = Path(
        row["feature_path"]
    )

    if not feature_path.exists():

        print(
            f"\nERROR: Missing feature file:"
        )

        print(
            feature_path
        )

        sys.exit(1)


    features = np.load(
        feature_path
    )


    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    if features.shape != (
        32,
        FEATURE_DIM
    ):

        print(
            "\nERROR: Invalid feature shape:"
        )

        print(
            feature_path
        )

        print(
            features.shape
        )

        sys.exit(1)


    if (
        np.isnan(features).any()
        or
        np.isinf(features).any()
    ):

        print(
            "\nERROR: NaN/Inf detected:"
        )

        print(
            feature_path
        )

        sys.exit(1)


    total_training_frames += (
        features.shape[0]
    )


    # -----------------------------------------------------
    # Detection masks
    # -----------------------------------------------------

    masks = features[
        :,
        MASK_START:FEATURE_DIM
    ]


    detection_counts["pose"] += int(
        np.sum(
            masks[:, 0] > 0.5
        )
    )

    detection_counts["face"] += int(
        np.sum(
            masks[:, 1] > 0.5
        )
    )

    detection_counts["left_hand"] += int(
        np.sum(
            masks[:, 2] > 0.5
        )
    )

    detection_counts["right_hand"] += int(
        np.sum(
            masks[:, 3] > 0.5
        )
    )


    # -----------------------------------------------------
    # Calculate stats group-by-group
    #
    # IMPORTANT:
    # Missing landmarks contain zeros.
    # We exclude those frames from statistics.
    # -----------------------------------------------------

    for group_name, group in GROUPS.items():

        start = group["start"]

        end = group["end"]

        mask_index = (
            group["mask_index"]
        )


        detected_frames = (
            masks[:, mask_index]
            > 0.5
        )


        if not np.any(
            detected_frames
        ):
            continue


        values = features[
            detected_frames,
            start:end
        ].astype(
            np.float64
        )


        feature_sum[
            start:end
        ] += np.sum(
            values,
            axis=0
        )


        feature_squared_sum[
            start:end
        ] += np.sum(
            values * values,
            axis=0
        )


        feature_count[
            start:end
        ] += (
            values.shape[0]
        )


    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    if (
        number % 250 == 0
        or
        number == len(train_df)
    ):

        print(
            f"Processed training sample "
            f"{number}/{len(train_df)}"
        )


# =========================================================
# CALCULATE MEAN AND STANDARD DEVIATION
# =========================================================

mean = np.zeros(
    FEATURE_DIM,
    dtype=np.float32
)

std = np.ones(
    FEATURE_DIM,
    dtype=np.float32
)


for feature_index in range(
    MASK_START
):

    count = (
        feature_count[
            feature_index
        ]
    )

    if count == 0:

        mean[
            feature_index
        ] = 0.0

        std[
            feature_index
        ] = 1.0

        continue


    feature_mean = (
        feature_sum[
            feature_index
        ]
        /
        count
    )


    feature_variance = (

        feature_squared_sum[
            feature_index
        ]
        /
        count

    ) - (

        feature_mean
        *
        feature_mean
    )


    # Numerical safety
    feature_variance = max(
        feature_variance,
        0.0
    )


    feature_std = np.sqrt(
        feature_variance
    )


    # Avoid division by extremely
    # small values
    if feature_std < 1e-6:

        feature_std = 1.0


    mean[
        feature_index
    ] = feature_mean


    std[
        feature_index
    ] = feature_std


# =========================================================
# MASK FEATURES
# =========================================================

# Detection indicators stay unchanged.
#
# We therefore use:
# mean = 0
# std  = 1

mean[
    MASK_START:
] = 0.0

std[
    MASK_START:
] = 1.0


# =========================================================
# VALIDATE STATISTICS
# =========================================================

if (
    np.isnan(mean).any()
    or
    np.isnan(std).any()
    or
    np.isinf(mean).any()
    or
    np.isinf(std).any()
):

    print(
        "\nERROR: Invalid normalization "
        "statistics."
    )

    sys.exit(1)


if np.any(
    std <= 0
):

    print(
        "\nERROR: Invalid standard "
        "deviation found."
    )

    sys.exit(1)


# =========================================================
# SAVE
# =========================================================

np.savez(

    stats_path,

    mean=mean,

    std=std,

    feature_count=feature_count,

    sequence_length=np.array(
        [32],
        dtype=np.int32
    ),

    feature_dim=np.array(
        [FEATURE_DIM],
        dtype=np.int32
    )
)


# =========================================================
# SUMMARY
# =========================================================

print(
    "\n" + "=" * 75
)

print(
    "NORMALIZATION SUMMARY"
)

print(
    "=" * 75
)


print(
    f"Dataset: "
    f"{dataset_name}"
)

print(
    f"Training videos: "
    f"{len(train_df)}"
)

print(
    f"Training frames: "
    f"{total_training_frames}"
)


print(
    "\nDetection rates in "
    "training frames:"
)


for name, count in (
    detection_counts.items()
):

    percentage = (
        100.0
        * count
        / total_training_frames
    )

    print(
        f"{name:12s}: "
        f"{count:7d}/"
        f"{total_training_frames} "
        f"({percentage:.2f}%)"
    )


print(
    "\nFeature statistics:"
)

print(
    f"Mean shape: "
    f"{mean.shape}"
)

print(
    f"Std shape: "
    f"{std.shape}"
)

print(
    f"Minimum std: "
    f"{std[:MASK_START].min():.8f}"
)

print(
    f"Maximum std: "
    f"{std[:MASK_START].max():.8f}"
)


print(
    "\nMask normalization:"
)

print(
    f"Mean: "
    f"{mean[MASK_START:]}"
)

print(
    f"Std: "
    f"{std[MASK_START:]}"
)


print(
    "\nSaved normalization statistics:"
)

print(
    stats_path
)


print(
    "\nSTATUS: NORMALIZATION "
    "STATISTICS CREATED SUCCESSFULLY ✅"
)

print(
    "=" * 75
)