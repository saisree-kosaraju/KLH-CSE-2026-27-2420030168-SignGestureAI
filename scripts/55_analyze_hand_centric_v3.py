from pathlib import Path
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# STEP 54 - WLASL300 HAND-CENTRIC GEOMETRY V3
# ============================================================
#
# BASELINE:
#   Step 49 pretrained Geometry-First model = 47.00% validation
#
# NEW V3:
#   Existing pretrained geometry/raw/motion backbone
#                 +
#   1047-D rescued hand-centric features
#                 ↓
#       hand static / velocity / acceleration encoder
#                 ↓
#          hand gated fusion
#                 ↓
#   residual gated fusion with pretrained backbone
#                 ↓
#        pretrained temporal Transformer
#                 ↓
#          temporal attention
#                 ↓
#           300 classes
#
# TRAIN = 2294
# VAL   = 600
# TEST  = 0 (UNTOUCHED)
# ============================================================


# ============================================================
# CONFIG
# ============================================================

SEED = 42

NUM_CLASSES = 300
SEQUENCE_LENGTH = 32

EXPECTED_TRAIN = 2294
EXPECTED_VAL = 600

RAW_DIM = 382
GEOMETRY_DIM = 378
VELOCITY_DIM = 378
ACCELERATION_DIM = 378
MOTION_TOTAL_DIM = 1134

HAND_TOTAL_DIM = 1047
HAND_STATIC_DIM = 351
HAND_VELOCITY_DIM = 348
HAND_ACCELERATION_DIM = 348

D_MODEL = 160

NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 320

RAW_DROPOUT = 0.30
GEOMETRY_DROPOUT = 0.20
VELOCITY_DROPOUT = 0.30
ACCELERATION_DROPOUT = 0.35

TRANSFORMER_DROPOUT = 0.35
CLASSIFIER_DROPOUT = 0.35

HAND_STATIC_DROPOUT = 0.25
HAND_VELOCITY_DROPOUT = 0.30
HAND_ACCELERATION_DROPOUT = 0.35

RAW_MODALITY_DROP = 0.12
VELOCITY_MODALITY_DROP = 0.10
ACCELERATION_MODALITY_DROP = 0.12

HAND_VELOCITY_DROP = 0.10
HAND_ACCELERATION_DROP = 0.12

FRAME_DROP_PROB = 0.06

BATCH_SIZE = 16

WARMUP_EPOCHS = 3
FINETUNE_EPOCHS = 45
PATIENCE = 10

NEW_BRANCH_LR = 2e-4
PRETRAINED_LR = 2e-5
HEAD_LR = 6e-5

WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.10
HAND_AUX_WEIGHT = 0.15
GRAD_CLIP = 1.0


# ============================================================
# ORIGINAL FEATURE LAYOUT
# ============================================================

POSE_SLICE = slice(0, 132)
LEFT_HAND_SLICE = slice(132, 195)
RIGHT_HAND_SLICE = slice(195, 258)
LIPS_SLICE = slice(258, 378)

MASK_START = 378

POSE_MASK_INDEX = 378
FACE_MASK_INDEX = 379
LEFT_MASK_INDEX = 380
RIGHT_MASK_INDEX = 381

GEOMETRY_OFFSET = 0
VELOCITY_OFFSET = 378
ACCELERATION_OFFSET = 756


# ============================================================
# HAND-CENTRIC FEATURE LAYOUT
# Must match Steps 52 and 53
# ============================================================

HAND_STATIC_LEFT = slice(0, 162)
HAND_STATIC_RIGHT = slice(162, 324)
HAND_STATIC_CROSS = slice(324, 348)

HAND_MASK_SLICE = slice(348, 351)
HAND_LEFT_MASK_INDEX = 348
HAND_RIGHT_MASK_INDEX = 349
HAND_BOTH_MASK_INDEX = 350

HAND_VELOCITY_START = 351

HAND_VELOCITY_LEFT = slice(351, 513)
HAND_VELOCITY_RIGHT = slice(513, 675)
HAND_VELOCITY_CROSS = slice(675, 699)

HAND_ACCELERATION_START = 699

HAND_ACCELERATION_LEFT = slice(699, 861)
HAND_ACCELERATION_RIGHT = slice(861, 1023)
HAND_ACCELERATION_CROSS = slice(1023, 1047)


# ============================================================
# PATHS
# ============================================================

BASE_MANIFEST = Path(
    "metadata/wlasl300_enhanced_fusion_manifest_v1.csv"
)

HAND_MANIFEST = Path(
    "metadata/wlasl300_hand_centric_manifest_v1.csv"
)

STEP49_CHECKPOINT = Path(
    "models/checkpoints/wlasl300_pretrained_finetune_v1_best.pt"
)

RAW_NORMALIZATION_PATH = Path(
    "metadata/wlasl_pretraining_raw_normalization_8710.npz"
)

MOTION_NORMALIZATION_PATH = Path(
    "metadata/wlasl_pretraining_motion_geometry_normalization_8710.npz"
)

HAND_NORMALIZATION_PATH = Path(
    "metadata/wlasl300_hand_centric_normalization_v1.npz"
)

CHECKPOINT_DIR = Path(
    "models/checkpoints"
)

RESULTS_DIR = Path(
    "results"
)

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    /
    "wlasl300_hand_centric_geometry_v3_best.pt"
)

HISTORY_PATH = (
    RESULTS_DIR
    /
    "wlasl300_hand_centric_geometry_v3_history.csv"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SEED / DEVICE
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 78)
print("STEP 55 - WLASL300 HAND-CENTRIC V3 VALIDATION ANALYSIS")
print("=" * 78)

print(f"\nDevice: {DEVICE}")
print(f"PyTorch: {torch.__version__}")


for path in [
    BASE_MANIFEST,
    HAND_MANIFEST,
    RAW_NORMALIZATION_PATH,
    MOTION_NORMALIZATION_PATH,
    HAND_NORMALIZATION_PATH
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file missing:\n{path}"
        )


# ============================================================
# MERGE MANIFESTS
# ============================================================

base_df = pd.read_csv(
    BASE_MANIFEST,
    dtype={
        "video_id": str
    }
)

hand_df = pd.read_csv(
    HAND_MANIFEST,
    dtype={
        "video_id": str
    }
)


for dataframe in [
    base_df,
    hand_df
]:

    dataframe["video_id"] = (
        dataframe["video_id"]
        .astype(str)
        .str.strip()
        .str.zfill(5)
    )

    dataframe["class_id"] = (
        dataframe["class_id"]
        .astype(int)
    )

    dataframe["split"] = (
        dataframe["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )


required_base = [
    "video_id",
    "class_id",
    "split",
    "landmark_path",
    "motion_geometry_path"
]

required_hand = [
    "video_id",
    "class_id",
    "split",
    "hand_centric_feature_path"
]


for column in required_base:

    if column not in base_df.columns:

        raise RuntimeError(
            f"Base manifest missing: {column}"
        )


for column in required_hand:

    if column not in hand_df.columns:

        raise RuntimeError(
            f"Hand manifest missing: {column}"
        )


merged = pd.merge(

    base_df[
        required_base
    ],

    hand_df[
        required_hand
    ],

    on=[
        "video_id",
        "class_id",
        "split"
    ],

    how="inner",

    validate="one_to_one"
)


train_df = (
    merged[
        merged["split"]
        ==
        "train"
    ]
    .copy()
    .reset_index(drop=True)
)


val_df = (
    merged[
        merged["split"]
        ==
        "val"
    ]
    .copy()
    .reset_index(drop=True)
)


test_df = (
    merged[
        merged["split"]
        ==
        "test"
    ]
)


print("\nMerged dataset:")
print(f"Train : {len(train_df)}")
print(f"Val   : {len(val_df)}")
print(f"Test  : {len(test_df)}")


if len(train_df) != EXPECTED_TRAIN:

    raise RuntimeError(
        f"Expected {EXPECTED_TRAIN} train videos, "
        f"found {len(train_df)}."
    )


if len(val_df) != EXPECTED_VAL:

    raise RuntimeError(
        f"Expected {EXPECTED_VAL} validation videos, "
        f"found {len(val_df)}."
    )


if len(test_df) != 0:

    raise RuntimeError(
        "TEST DATA ENTERED STEP 55!"
    )


print("Test videos used: 0 ✅")


# ============================================================
# LOAD NORMALIZATION
# ============================================================

raw_norm = np.load(
    RAW_NORMALIZATION_PATH
)

RAW_MEAN = raw_norm["mean"].astype(
    np.float32
)

RAW_STD = raw_norm["std"].astype(
    np.float32
)


motion_norm = np.load(
    MOTION_NORMALIZATION_PATH
)

MOTION_MEAN = motion_norm["mean"].astype(
    np.float32
)

MOTION_STD = motion_norm["std"].astype(
    np.float32
)


hand_norm = np.load(
    HAND_NORMALIZATION_PATH
)

HAND_MEAN = hand_norm["mean"].astype(
    np.float32
)

HAND_STD = hand_norm["std"].astype(
    np.float32
)


if RAW_MEAN.shape != (
    RAW_DIM,
):

    raise RuntimeError(
        "Bad raw normalization shape."
    )


if MOTION_MEAN.shape != (
    MOTION_TOTAL_DIM,
):

    raise RuntimeError(
        "Bad motion normalization shape."
    )


if HAND_MEAN.shape != (
    HAND_TOTAL_DIM,
):

    raise RuntimeError(
        "Bad hand normalization shape."
    )


print("Normalization files loaded: PASS ✅")


# ============================================================
# ORIGINAL NORMALIZATION
# ============================================================

def normalize_raw(
    raw
):

    output = raw.copy()

    masks = raw[
        :,
        MASK_START:
    ].copy()


    groups = [

        (
            POSE_SLICE,
            masks[:, 0] > 0.5
        ),

        (
            LIPS_SLICE,
            masks[:, 1] > 0.5
        ),

        (
            LEFT_HAND_SLICE,
            masks[:, 2] > 0.5
        ),

        (
            RIGHT_HAND_SLICE,
            masks[:, 3] > 0.5
        )
    ]


    for feature_slice, detected in groups:

        if np.any(
            detected
        ):

            output[
                detected,
                feature_slice
            ] = (

                output[
                    detected,
                    feature_slice
                ]

                -

                RAW_MEAN[
                    feature_slice
                ]

            ) / RAW_STD[
                feature_slice
            ]


        output[
            ~detected,
            feature_slice
        ] = 0.0


    output[
        :,
        MASK_START:
    ] = masks


    return output.astype(
        np.float32
    )


def temporal_velocity_valid(
    detected
):

    valid = np.zeros(
        SEQUENCE_LENGTH,
        dtype=bool
    )

    valid[1:] = (
        detected[1:]
        &
        detected[:-1]
    )

    return valid


def temporal_acceleration_valid(
    detected
):

    valid = np.zeros(
        SEQUENCE_LENGTH,
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


def normalize_motion(
    features,
    raw_original
):

    output = features.copy()


    pose = (
        raw_original[
            :,
            POSE_MASK_INDEX
        ]
        >
        0.5
    )


    face = (
        raw_original[
            :,
            FACE_MASK_INDEX
        ]
        >
        0.5
    )


    left = (
        raw_original[
            :,
            LEFT_MASK_INDEX
        ]
        >
        0.5
    )


    right = (
        raw_original[
            :,
            RIGHT_MASK_INDEX
        ]
        >
        0.5
    )


    groups = [

        (
            POSE_SLICE,
            pose
        ),

        (
            LEFT_HAND_SLICE,
            left
        ),

        (
            RIGHT_HAND_SLICE,
            right
        ),

        (
            LIPS_SLICE,
            face
        )
    ]


    for feature_slice, detected in groups:

        geometry_slice = slice(
            GEOMETRY_OFFSET
            +
            feature_slice.start,

            GEOMETRY_OFFSET
            +
            feature_slice.stop
        )


        if np.any(
            detected
        ):

            output[
                detected,
                geometry_slice
            ] = (

                output[
                    detected,
                    geometry_slice
                ]

                -

                MOTION_MEAN[
                    geometry_slice
                ]

            ) / MOTION_STD[
                geometry_slice
            ]


        output[
            ~detected,
            geometry_slice
        ] = 0.0


        velocity_valid = (
            temporal_velocity_valid(
                detected
            )
        )


        velocity_slice = slice(
            VELOCITY_OFFSET
            +
            feature_slice.start,

            VELOCITY_OFFSET
            +
            feature_slice.stop
        )


        if np.any(
            velocity_valid
        ):

            output[
                velocity_valid,
                velocity_slice
            ] = (

                output[
                    velocity_valid,
                    velocity_slice
                ]

                -

                MOTION_MEAN[
                    velocity_slice
                ]

            ) / MOTION_STD[
                velocity_slice
            ]


        output[
            ~velocity_valid,
            velocity_slice
        ] = 0.0


        acceleration_valid = (
            temporal_acceleration_valid(
                detected
            )
        )


        acceleration_slice = slice(
            ACCELERATION_OFFSET
            +
            feature_slice.start,

            ACCELERATION_OFFSET
            +
            feature_slice.stop
        )


        if np.any(
            acceleration_valid
        ):

            output[
                acceleration_valid,
                acceleration_slice
            ] = (

                output[
                    acceleration_valid,
                    acceleration_slice
                ]

                -

                MOTION_MEAN[
                    acceleration_slice
                ]

            ) / MOTION_STD[
                acceleration_slice
            ]


        output[
            ~acceleration_valid,
            acceleration_slice
        ] = 0.0


    return output.astype(
        np.float32
    )


# ============================================================
# HAND-CENTRIC NORMALIZATION
# ============================================================

def normalize_hand_features(
    features
):

    output = features.copy()


    left_detected = (
        features[
            :,
            HAND_LEFT_MASK_INDEX
        ]
        >
        0.5
    )


    right_detected = (
        features[
            :,
            HAND_RIGHT_MASK_INDEX
        ]
        >
        0.5
    )


    both_detected = (
        features[
            :,
            HAND_BOTH_MASK_INDEX
        ]
        >
        0.5
    )


    left_vel = temporal_velocity_valid(
        left_detected
    )

    right_vel = temporal_velocity_valid(
        right_detected
    )

    both_vel = temporal_velocity_valid(
        both_detected
    )


    left_acc = temporal_acceleration_valid(
        left_detected
    )

    right_acc = temporal_acceleration_valid(
        right_detected
    )

    both_acc = temporal_acceleration_valid(
        both_detected
    )


    groups = [

        (
            HAND_STATIC_LEFT,
            left_detected
        ),

        (
            HAND_STATIC_RIGHT,
            right_detected
        ),

        (
            HAND_STATIC_CROSS,
            both_detected
        ),

        (
            HAND_VELOCITY_LEFT,
            left_vel
        ),

        (
            HAND_VELOCITY_RIGHT,
            right_vel
        ),

        (
            HAND_VELOCITY_CROSS,
            both_vel
        ),

        (
            HAND_ACCELERATION_LEFT,
            left_acc
        ),

        (
            HAND_ACCELERATION_RIGHT,
            right_acc
        ),

        (
            HAND_ACCELERATION_CROSS,
            both_acc
        )
    ]


    for feature_slice, valid_mask in groups:

        if np.any(
            valid_mask
        ):

            output[
                valid_mask,
                feature_slice
            ] = (

                output[
                    valid_mask,
                    feature_slice
                ]

                -

                HAND_MEAN[
                    feature_slice
                ]

            ) / HAND_STD[
                feature_slice
            ]


        output[
            ~valid_mask,
            feature_slice
        ] = 0.0


    output[
        :,
        HAND_MASK_SLICE
    ] = features[
        :,
        HAND_MASK_SLICE
    ]


    return output.astype(
        np.float32
    )


# ============================================================
# DATASET
# ============================================================

class WLASL300V3Dataset(
    Dataset
):

    def __init__(
        self,
        dataframe
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )


    def __len__(
        self
    ):

        return len(
            self.df
        )


    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]


        raw_original = np.load(
            row[
                "landmark_path"
            ]
        ).astype(
            np.float32
        )


        motion = np.load(
            row[
                "motion_geometry_path"
            ]
        ).astype(
            np.float32
        )


        hand = np.load(
            row[
                "hand_centric_feature_path"
            ]
        ).astype(
            np.float32
        )


        if raw_original.shape != (
            SEQUENCE_LENGTH,
            RAW_DIM
        ):

            raise RuntimeError(
                f"Bad raw shape for "
                f"{row['video_id']}: "
                f"{raw_original.shape}"
            )


        if motion.shape != (
            SEQUENCE_LENGTH,
            MOTION_TOTAL_DIM
        ):

            raise RuntimeError(
                f"Bad motion shape for "
                f"{row['video_id']}: "
                f"{motion.shape}"
            )


        if hand.shape != (
            SEQUENCE_LENGTH,
            HAND_TOTAL_DIM
        ):

            raise RuntimeError(
                f"Bad hand shape for "
                f"{row['video_id']}: "
                f"{hand.shape}"
            )


        raw = normalize_raw(
            raw_original
        )


        motion = normalize_motion(
            motion,
            raw_original
        )


        hand = normalize_hand_features(
            hand
        )


        geometry = motion[
            :,
            0:378
        ].copy()


        velocity = motion[
            :,
            378:756
        ].copy()


        acceleration = motion[
            :,
            756:1134
        ].copy()


        label = int(
            row[
                "class_id"
            ]
        )


        return (

            torch.from_numpy(
                raw
            ),

            torch.from_numpy(
                geometry
            ),

            torch.from_numpy(
                velocity
            ),

            torch.from_numpy(
                acceleration
            ),

            torch.from_numpy(
                hand
            ),

            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# LOADERS
# ============================================================

train_dataset = WLASL300V3Dataset(
    train_df
)

val_dataset = WLASL300V3Dataset(
    val_df
)


generator = torch.Generator()
generator.manual_seed(SEED)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    generator=generator
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


print(
    f"\nTrain batches: {len(train_loader)}"
)

print(
    f"Val batches  : {len(val_loader)}"
)


# ============================================================
# MODEL COMPONENTS
# ============================================================

class Projection(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        dropout
    ):

        super().__init__()


        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                D_MODEL
            ),

            nn.LayerNorm(
                D_MODEL
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            )
        )


    def forward(
        self,
        x
    ):

        return self.net(
            x
        )


class GeometryFirstFusion(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        self.gates = nn.Sequential(

            nn.LayerNorm(
                D_MODEL * 4
            ),

            nn.Linear(
                D_MODEL * 4,
                D_MODEL
            ),

            nn.GELU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                D_MODEL,
                3
            )
        )


        self.output_norm = nn.LayerNorm(
            D_MODEL
        )


    def modality_dropout(
        self,
        x,
        probability
    ):

        if (
            not self.training
            or
            probability <= 0
        ):

            return x


        keep = (

            torch.rand(
                x.size(0),
                1,
                1,
                device=x.device
            )

            >
            probability
        ).float()


        keep = (
            keep
            /
            (
                1.0
                -
                probability
            )
        )


        return x * keep


    def forward(
        self,
        geometry,
        raw,
        velocity,
        acceleration
    ):

        raw = self.modality_dropout(
            raw,
            RAW_MODALITY_DROP
        )


        velocity = self.modality_dropout(
            velocity,
            VELOCITY_MODALITY_DROP
        )


        acceleration = self.modality_dropout(
            acceleration,
            ACCELERATION_MODALITY_DROP
        )


        combined = torch.cat(

            [
                geometry,
                raw,
                velocity,
                acceleration
            ],

            dim=-1
        )


        gates = torch.sigmoid(

            self.gates(
                combined
            )
        )


        fused = (

            geometry

            +

            gates[
                :,
                :,
                0:1
            ]
            *
            raw

            +

            gates[
                :,
                :,
                1:2
            ]
            *
            velocity

            +

            gates[
                :,
                :,
                2:3
            ]
            *
            acceleration
        )


        return (
            self.output_norm(
                fused
            ),
            gates
        )


class HandFusion(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        self.gates = nn.Sequential(

            nn.LayerNorm(
                D_MODEL * 3
            ),

            nn.Linear(
                D_MODEL * 3,
                D_MODEL
            ),

            nn.GELU(),

            nn.Dropout(
                0.25
            ),

            nn.Linear(
                D_MODEL,
                2
            )
        )


        self.output_norm = nn.LayerNorm(
            D_MODEL
        )


    def modality_dropout(
        self,
        x,
        probability
    ):

        if (
            not self.training
            or
            probability <= 0
        ):

            return x


        keep = (

            torch.rand(
                x.size(0),
                1,
                1,
                device=x.device
            )

            >
            probability
        ).float()


        keep = (
            keep
            /
            (
                1.0
                -
                probability
            )
        )


        return x * keep


    def forward(
        self,
        static,
        velocity,
        acceleration
    ):

        velocity = self.modality_dropout(
            velocity,
            HAND_VELOCITY_DROP
        )


        acceleration = self.modality_dropout(
            acceleration,
            HAND_ACCELERATION_DROP
        )


        combined = torch.cat(

            [
                static,
                velocity,
                acceleration
            ],

            dim=-1
        )


        gates = torch.sigmoid(

            self.gates(
                combined
            )
        )


        fused = (

            static

            +

            gates[
                :,
                :,
                0:1
            ]
            *
            velocity

            +

            gates[
                :,
                :,
                1:2
            ]
            *
            acceleration
        )


        return (
            self.output_norm(
                fused
            ),
            gates
        )


class ResidualHandInjection(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        self.gate = nn.Sequential(

            nn.LayerNorm(
                D_MODEL * 2
            ),

            nn.Linear(
                D_MODEL * 2,
                D_MODEL // 2
            ),

            nn.GELU(),

            nn.Linear(
                D_MODEL // 2,
                1
            )
        )


        # Start close to the existing 47% Step-49 model.
        nn.init.zeros_(
            self.gate[-1].weight
        )

        nn.init.constant_(
            self.gate[-1].bias,
            -3.0
        )


    def forward(
        self,
        base,
        hand
    ):

        combined = torch.cat(
            [
                base,
                hand
            ],
            dim=-1
        )


        gate = torch.sigmoid(
            self.gate(
                combined
            )
        )


        fused = (
            base
            +
            gate
            *
            hand
        )


        return (
            fused,
            gate
        )


class TemporalAttentionPool(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        self.attention = nn.Sequential(

            nn.Linear(
                D_MODEL,
                D_MODEL // 2
            ),

            nn.Tanh(),

            nn.Linear(
                D_MODEL // 2,
                1
            )
        )


    def forward(
        self,
        x
    ):

        scores = (
            self.attention(
                x
            )
            .squeeze(-1)
        )


        weights = torch.softmax(
            scores,
            dim=1
        )


        pooled = (
            x
            *
            weights.unsqueeze(-1)
        ).sum(
            dim=1
        )


        return (
            pooled,
            weights
        )


# ============================================================
# V3 MODEL
# ============================================================

class HandCentricGeometryV3(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        # ----------------------------------------------------
        # PRETRAINED STEP-49 BRANCH
        # Names intentionally match Step 49 checkpoint.
        # ----------------------------------------------------

        self.geometry_projection = Projection(
            GEOMETRY_DIM,
            GEOMETRY_DROPOUT
        )

        self.raw_projection = Projection(
            RAW_DIM,
            RAW_DROPOUT
        )

        self.velocity_projection = Projection(
            VELOCITY_DIM,
            VELOCITY_DROPOUT
        )

        self.acceleration_projection = Projection(
            ACCELERATION_DIM,
            ACCELERATION_DROPOUT
        )

        self.fusion = GeometryFirstFusion()


        self.position_embedding = nn.Parameter(

            torch.zeros(
                1,
                SEQUENCE_LENGTH,
                D_MODEL
            )
        )


        encoder_layer = nn.TransformerEncoderLayer(

            d_model=D_MODEL,

            nhead=NUM_HEADS,

            dim_feedforward=FF_DIM,

            dropout=TRANSFORMER_DROPOUT,

            activation="gelu",

            batch_first=True,

            norm_first=True
        )


        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=NUM_LAYERS
        )


        self.final_norm = nn.LayerNorm(
            D_MODEL
        )


        self.temporal_pool = TemporalAttentionPool()


        self.classifier = nn.Sequential(

            nn.Dropout(
                CLASSIFIER_DROPOUT
            ),

            nn.Linear(
                D_MODEL,
                NUM_CLASSES
            )
        )


        self.geometry_classifier = nn.Sequential(

            nn.Dropout(
                0.25
            ),

            nn.Linear(
                D_MODEL,
                NUM_CLASSES
            )
        )


        # ----------------------------------------------------
        # NEW HAND-CENTRIC BRANCH
        # ----------------------------------------------------

        self.hand_static_projection = Projection(
            HAND_STATIC_DIM,
            HAND_STATIC_DROPOUT
        )

        self.hand_velocity_projection = Projection(
            HAND_VELOCITY_DIM,
            HAND_VELOCITY_DROPOUT
        )

        self.hand_acceleration_projection = Projection(
            HAND_ACCELERATION_DIM,
            HAND_ACCELERATION_DROPOUT
        )

        self.hand_fusion = HandFusion()

        self.hand_injection = ResidualHandInjection()


        self.hand_classifier = nn.Sequential(

            nn.Dropout(
                0.25
            ),

            nn.Linear(
                D_MODEL,
                NUM_CLASSES
            )
        )


    def forward(
        self,
        raw,
        geometry,
        velocity,
        acceleration,
        hand
    ):

        # ====================================================
        # PRETRAINED BASE
        # ====================================================

        geometry_encoded = self.geometry_projection(
            geometry
        )


        raw_encoded = self.raw_projection(
            raw
        )


        velocity_encoded = self.velocity_projection(
            velocity
        )


        acceleration_encoded = self.acceleration_projection(
            acceleration
        )


        geometry_summary = (
            geometry_encoded.mean(
                dim=1
            )
        )


        geometry_logits = self.geometry_classifier(
            geometry_summary
        )


        base_fused, base_gates = self.fusion(

            geometry_encoded,
            raw_encoded,
            velocity_encoded,
            acceleration_encoded
        )


        # ====================================================
        # HAND BRANCH
        # ====================================================

        hand_static = hand[
            :,
            :,
            0:351
        ]


        hand_velocity = hand[
            :,
            :,
            351:699
        ]


        hand_acceleration = hand[
            :,
            :,
            699:1047
        ]


        hand_static_encoded = self.hand_static_projection(
            hand_static
        )


        hand_velocity_encoded = self.hand_velocity_projection(
            hand_velocity
        )


        hand_acceleration_encoded = (
            self.hand_acceleration_projection(
                hand_acceleration
            )
        )


        hand_fused, hand_gates = self.hand_fusion(

            hand_static_encoded,
            hand_velocity_encoded,
            hand_acceleration_encoded
        )


        hand_summary = hand_fused.mean(
            dim=1
        )


        hand_logits = self.hand_classifier(
            hand_summary
        )


        # ====================================================
        # RESIDUAL HAND INJECTION
        # ====================================================

        fused, hand_injection_gate = self.hand_injection(

            base_fused,
            hand_fused
        )


        fused = (
            fused
            +
            self.position_embedding[
                :,
                :fused.size(1),
                :
            ]
        )


        encoded = self.transformer(
            fused
        )


        encoded = self.final_norm(
            encoded
        )


        pooled, attention = self.temporal_pool(
            encoded
        )


        logits = self.classifier(
            pooled
        )


        return (

            logits,
            geometry_logits,
            hand_logits,
            base_gates,
            hand_gates,
            hand_injection_gate,
            attention
        )



# ============================================================
# STEP 55 - V3 VALIDATION ANALYSIS + HAND ABLATIONS
# ============================================================

V3_CHECKPOINT_PATH = Path(
    "models/checkpoints/wlasl300_hand_centric_geometry_v3_best.pt"
)

PREDICTIONS_PATH = Path(
    "results/wlasl300_hand_centric_v3_val_predictions.csv"
)

CLASS_ACCURACY_PATH = Path(
    "results/wlasl300_hand_centric_v3_class_accuracy.csv"
)

CONFUSIONS_PATH = Path(
    "results/wlasl300_hand_centric_v3_top_confusions.csv"
)

HIGH_CONFIDENCE_ERRORS_PATH = Path(
    "results/wlasl300_hand_centric_v3_high_confidence_errors.csv"
)

ABLATION_PATH = Path(
    "results/wlasl300_hand_centric_v3_ablation.csv"
)

SUMMARY_PATH = Path(
    "results/wlasl300_hand_centric_v3_analysis.txt"
)

HIGH_CONFIDENCE_THRESHOLD = 0.50

if not V3_CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Missing Step 54 V3 checkpoint:\n{V3_CHECKPOINT_PATH}"
    )

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OPTIONAL GLOSS MAPPING
# ============================================================

def load_gloss_map():

    candidates = [
        Path("metadata/wlasl300_expanded_3392.csv"),
        Path("metadata/wlasl300_final.csv"),
        Path("metadata/wlasl300_available_from_full_inventory.csv")
    ]

    for path in candidates:

        if not path.exists():
            continue

        try:

            gloss_df = pd.read_csv(
                path
            )

            if (
                "class_id" in gloss_df.columns
                and
                "gloss" in gloss_df.columns
            ):

                gloss_df["class_id"] = (
                    gloss_df["class_id"]
                    .astype(int)
                )

                mapping = (
                    gloss_df[
                        [
                            "class_id",
                            "gloss"
                        ]
                    ]
                    .drop_duplicates(
                        subset=[
                            "class_id"
                        ]
                    )
                    .set_index(
                        "class_id"
                    )["gloss"]
                    .astype(str)
                    .to_dict()
                )

                if len(mapping) > 0:

                    print(
                        f"Gloss mapping loaded from: {path}"
                    )

                    return mapping

        except Exception as error:

            print(
                f"Gloss mapping warning for {path}: {error}"
            )


    print(
        "Gloss mapping file not found. "
        "Class IDs will still be analyzed."
    )

    return {}


GLOSS_MAP = load_gloss_map()


def gloss_of(
    class_id
):

    return GLOSS_MAP.get(
        int(
            class_id
        ),
        f"class_{int(class_id)}"
    )


# ============================================================
# CREATE MODEL + LOAD STEP 54 BEST CHECKPOINT
# ============================================================

model = HandCentricGeometryV3().to(
    DEVICE
)


checkpoint = torch.load(

    V3_CHECKPOINT_PATH,

    map_location=DEVICE,

    weights_only=False
)


if "model_state_dict" not in checkpoint:

    raise RuntimeError(
        "V3 checkpoint does not contain model_state_dict."
    )


model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ],

    strict=True
)


model.eval()


checkpoint_best_accuracy = checkpoint.get(
    "best_val_accuracy",
    None
)


print(
    "\nStep 54 V3 checkpoint loaded: PASS ✅"
)


if checkpoint_best_accuracy is not None:

    print(
        f"Checkpoint best validation accuracy: "
        f"{float(checkpoint_best_accuracy):.2f}%"
    )


print(
    f"Checkpoint global epoch: "
    f"{checkpoint.get('global_epoch', 'unknown')}"
)

print(
    f"Checkpoint stage: "
    f"{checkpoint.get('stage', 'unknown')}"
)


# ============================================================
# LOSS
# Same validation loss definition used during Step 54
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=LABEL_SMOOTHING
)


# ============================================================
# ABLATION MODES
# ============================================================

ABLATION_MODES = [
    "full_v3",
    "no_hand_branch",
    "no_hand_velocity",
    "no_hand_acceleration",
    "no_cross_hand",
    "static_hand_only"
]


def apply_hand_ablation(
    hand,
    mode
):

    if mode == "full_v3":

        return hand


    output = hand.clone()


    if mode == "no_hand_branch":

        output.zero_()


    elif mode == "no_hand_velocity":

        output[
            :,
            :,
            HAND_VELOCITY_START:
            HAND_ACCELERATION_START
        ] = 0.0


    elif mode == "no_hand_acceleration":

        output[
            :,
            :,
            HAND_ACCELERATION_START:
        ] = 0.0


    elif mode == "no_cross_hand":

        # Static cross-hand geometry
        output[
            :,
            :,
            HAND_STATIC_CROSS
        ] = 0.0

        # Cross-hand velocity
        output[
            :,
            :,
            HAND_VELOCITY_CROSS
        ] = 0.0

        # Cross-hand acceleration
        output[
            :,
            :,
            HAND_ACCELERATION_CROSS
        ] = 0.0


    elif mode == "static_hand_only":

        output[
            :,
            :,
            HAND_VELOCITY_START:
        ] = 0.0


    else:

        raise ValueError(
            f"Unknown ablation mode: {mode}"
        )


    return output


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_mode(
    mode,
    collect_predictions=False
):

    model.eval()


    running_combined_loss = 0.0

    top1_correct = 0
    top5_correct = 0
    total = 0


    base_gate_sum = np.zeros(
        3,
        dtype=np.float64
    )

    hand_gate_sum = np.zeros(
        2,
        dtype=np.float64
    )

    injection_sum = 0.0

    frame_count = 0


    prediction_rows = []

    sample_offset = 0


    for batch in val_loader:

        (
            raw,
            geometry,
            velocity,
            acceleration,
            hand,
            labels

        ) = batch


        batch_size = labels.size(
            0
        )


        # Keep a CPU copy of masks for diagnostics
        hand_cpu = hand.clone()


        raw = raw.to(
            DEVICE
        )

        geometry = geometry.to(
            DEVICE
        )

        velocity = velocity.to(
            DEVICE
        )

        acceleration = acceleration.to(
            DEVICE
        )

        hand = hand.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        ablated_hand = apply_hand_ablation(
            hand,
            mode
        )


        (
            logits,
            _,
            hand_logits,
            base_gates,
            hand_gates,
            injection_gate,
            _

        ) = model(

            raw,
            geometry,
            velocity,
            acceleration,
            ablated_hand
        )


        main_loss = criterion(
            logits,
            labels
        )


        hand_loss = criterion(
            hand_logits,
            labels
        )


        combined_loss = (
            main_loss
            +
            HAND_AUX_WEIGHT
            *
            hand_loss
        )


        running_combined_loss += (
            combined_loss.item()
            *
            batch_size
        )


        probabilities = torch.softmax(
            logits,
            dim=1
        )


        top1_probabilities, predictions = (
            probabilities.max(
                dim=1
            )
        )


        top5_probabilities, top5_indices = (
            probabilities.topk(
                5,
                dim=1
            )
        )


        top1_correct += (
            predictions
            .eq(
                labels
            )
            .sum()
            .item()
        )


        top5_hits = (
            top5_indices
            .eq(
                labels.view(
                    -1,
                    1
                )
            )
            .any(
                dim=1
            )
        )


        top5_correct += (
            top5_hits
            .sum()
            .item()
        )


        total += batch_size


        base_gate_sum += (
            base_gates
            .sum(
                dim=(0, 1)
            )
            .cpu()
            .numpy()
        )


        hand_gate_sum += (
            hand_gates
            .sum(
                dim=(0, 1)
            )
            .cpu()
            .numpy()
        )


        injection_sum += float(
            injection_gate
            .sum()
            .item()
        )


        frame_count += (
            injection_gate.size(0)
            *
            injection_gate.size(1)
        )


        if collect_predictions:

            predictions_cpu = (
                predictions
                .cpu()
                .numpy()
            )

            top1_probabilities_cpu = (
                top1_probabilities
                .cpu()
                .numpy()
            )

            top5_indices_cpu = (
                top5_indices
                .cpu()
                .numpy()
            )

            top5_probabilities_cpu = (
                top5_probabilities
                .cpu()
                .numpy()
            )

            labels_cpu = (
                labels
                .cpu()
                .numpy()
            )

            top5_hits_cpu = (
                top5_hits
                .cpu()
                .numpy()
            )


            for i in range(
                batch_size
            ):

                row = val_df.iloc[
                    sample_offset
                    +
                    i
                ]


                true_class = int(
                    labels_cpu[
                        i
                    ]
                )

                predicted_class = int(
                    predictions_cpu[
                        i
                    ]
                )


                top5_ids = [
                    int(
                        value
                    )
                    for value
                    in top5_indices_cpu[
                        i
                    ].tolist()
                ]


                top5_probs = [
                    float(
                        value
                    )
                    for value
                    in top5_probabilities_cpu[
                        i
                    ].tolist()
                ]


                left_coverage = float(
                    hand_cpu[
                        i,
                        :,
                        HAND_LEFT_MASK_INDEX
                    ]
                    .float()
                    .mean()
                    .item()
                )


                right_coverage = float(
                    hand_cpu[
                        i,
                        :,
                        HAND_RIGHT_MASK_INDEX
                    ]
                    .float()
                    .mean()
                    .item()
                )


                both_coverage = float(
                    hand_cpu[
                        i,
                        :,
                        HAND_BOTH_MASK_INDEX
                    ]
                    .float()
                    .mean()
                    .item()
                )


                prediction_rows.append(
                    {
                        "video_id":
                            row[
                                "video_id"
                            ],

                        "true_class_id":
                            true_class,

                        "true_gloss":
                            gloss_of(
                                true_class
                            ),

                        "pred_class_id":
                            predicted_class,

                        "pred_gloss":
                            gloss_of(
                                predicted_class
                            ),

                        "confidence":
                            float(
                                top1_probabilities_cpu[
                                    i
                                ]
                            ),

                        "correct":
                            int(
                                predicted_class
                                ==
                                true_class
                            ),

                        "top5_contains_true":
                            int(
                                top5_hits_cpu[
                                    i
                                ]
                            ),

                        "top5_class_ids":
                            ",".join(
                                str(
                                    value
                                )
                                for value
                                in top5_ids
                            ),

                        "top5_glosses":
                            " | ".join(
                                gloss_of(
                                    value
                                )
                                for value
                                in top5_ids
                            ),

                        "top5_probabilities":
                            ",".join(
                                f"{value:.6f}"
                                for value
                                in top5_probs
                            ),

                        "left_hand_coverage":
                            left_coverage,

                        "right_hand_coverage":
                            right_coverage,

                        "both_hands_coverage":
                            both_coverage
                    }
                )


        sample_offset += batch_size


    if total != EXPECTED_VAL:

        raise RuntimeError(
            f"Expected {EXPECTED_VAL} validation samples, "
            f"evaluated {total}."
        )


    results = {

        "mode":
            mode,

        "combined_val_loss":
            running_combined_loss
            /
            total,

        "top1_accuracy":
            100.0
            *
            top1_correct
            /
            total,

        "top5_accuracy":
            100.0
            *
            top5_correct
            /
            total,

        "base_raw_gate":
            base_gate_sum[0]
            /
            frame_count,

        "base_velocity_gate":
            base_gate_sum[1]
            /
            frame_count,

        "base_acceleration_gate":
            base_gate_sum[2]
            /
            frame_count,

        "hand_velocity_gate":
            hand_gate_sum[0]
            /
            frame_count,

        "hand_acceleration_gate":
            hand_gate_sum[1]
            /
            frame_count,

        "hand_injection_gate":
            injection_sum
            /
            frame_count
    }


    return (
        results,
        prediction_rows
    )


# ============================================================
# RUN FULL V3 FIRST
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "FULL V3 VALIDATION"
)

print(
    "=" * 78
)


start_time = time.time()


full_results, prediction_rows = evaluate_mode(

    "full_v3",

    collect_predictions=True
)


print(
    f"\nTop-1 Accuracy: "
    f"{full_results['top1_accuracy']:.2f}%"
)

print(
    f"Top-5 Accuracy: "
    f"{full_results['top5_accuracy']:.2f}%"
)

print(
    f"Combined Val Loss: "
    f"{full_results['combined_val_loss']:.4f}"
)

print(
    f"Hand Injection Gate: "
    f"{full_results['hand_injection_gate']:.4f}"
)


if checkpoint_best_accuracy is not None:

    difference = (
        full_results[
            "top1_accuracy"
        ]
        -
        float(
            checkpoint_best_accuracy
        )
    )


    print(
        f"Difference vs checkpoint record: "
        f"{difference:+.4f} points"
    )


    if abs(
        difference
    ) <= 0.34:

        print(
            "Checkpoint reproduction: PASS ✅"
        )

    else:

        print(
            "WARNING: Validation accuracy differs "
            "from the checkpoint record."
        )


# ============================================================
# SAVE FULL PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame(
    prediction_rows
)


if len(
    predictions_df
) != EXPECTED_VAL:

    raise RuntimeError(
        f"Expected {EXPECTED_VAL} prediction rows, "
        f"found {len(predictions_df)}."
    )


predictions_df.to_csv(

    PREDICTIONS_PATH,

    index=False
)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

class_rows = []


for class_id, group in predictions_df.groupby(
    "true_class_id"
):

    samples = len(
        group
    )


    correct = int(
        group[
            "correct"
        ].sum()
    )


    top5_correct = int(
        group[
            "top5_contains_true"
        ].sum()
    )


    class_rows.append(
        {
            "class_id":
                int(
                    class_id
                ),

            "gloss":
                gloss_of(
                    class_id
                ),

            "samples":
                samples,

            "correct":
                correct,

            "top1_accuracy":
                100.0
                *
                correct
                /
                samples,

            "top5_correct":
                top5_correct,

            "top5_accuracy":
                100.0
                *
                top5_correct
                /
                samples
        }
    )


class_accuracy_df = pd.DataFrame(
    class_rows
).sort_values(

    [
        "top1_accuracy",
        "samples",
        "class_id"
    ],

    ascending=[
        True,
        False,
        True
    ]
)


class_accuracy_df.to_csv(

    CLASS_ACCURACY_PATH,

    index=False
)


macro_top1 = float(
    class_accuracy_df[
        "top1_accuracy"
    ].mean()
)


macro_top5 = float(
    class_accuracy_df[
        "top5_accuracy"
    ].mean()
)


zero_accuracy_classes = int(
    (
        class_accuracy_df[
            "top1_accuracy"
        ]
        ==
        0.0
    ).sum()
)


high_accuracy_classes = int(
    (
        class_accuracy_df[
            "top1_accuracy"
        ]
        >=
        75.0
    ).sum()
)


perfect_classes = int(
    (
        class_accuracy_df[
            "top1_accuracy"
        ]
        ==
        100.0
    ).sum()
)


# ============================================================
# CONFUSIONS
# ============================================================

wrong_df = predictions_df[
    predictions_df[
        "correct"
    ]
    ==
    0
].copy()


confusion_df = (

    wrong_df

    .groupby(
        [
            "true_class_id",
            "true_gloss",
            "pred_class_id",
            "pred_gloss"
        ]
    )

    .size()

    .reset_index(
        name="count"
    )
)


true_sample_counts = (
    predictions_df
    .groupby(
        "true_class_id"
    )
    .size()
    .to_dict()
)


if len(
    confusion_df
) > 0:

    confusion_df[
        "true_class_samples"
    ] = (

        confusion_df[
            "true_class_id"
        ]
        .map(
            true_sample_counts
        )
        .astype(int)
    )


    confusion_df[
        "confusion_rate_percent"
    ] = (

        100.0
        *
        confusion_df[
            "count"
        ]
        /
        confusion_df[
            "true_class_samples"
        ]
    )


    confusion_df = confusion_df.sort_values(

        [
            "count",
            "confusion_rate_percent"
        ],

        ascending=[
            False,
            False
        ]
    )


confusion_df.to_csv(

    CONFUSIONS_PATH,

    index=False
)


# ============================================================
# HIGH-CONFIDENCE ERRORS
# ============================================================

high_confidence_errors_df = (

    wrong_df[
        wrong_df[
            "confidence"
        ]
        >=
        HIGH_CONFIDENCE_THRESHOLD
    ]

    .copy()

    .sort_values(
        "confidence",
        ascending=False
    )
)


high_confidence_errors_df.to_csv(

    HIGH_CONFIDENCE_ERRORS_PATH,

    index=False
)


# ============================================================
# HAND COVERAGE: CORRECT VS INCORRECT
# ============================================================

coverage_correct = (
    predictions_df[
        predictions_df[
            "correct"
        ]
        ==
        1
    ]
)


coverage_incorrect = (
    predictions_df[
        predictions_df[
            "correct"
        ]
        ==
        0
    ]
)


coverage_stats = {}


for column in [
    "left_hand_coverage",
    "right_hand_coverage",
    "both_hands_coverage"
]:

    correct_mean = float(
        coverage_correct[
            column
        ].mean()
    )


    incorrect_mean = float(
        coverage_incorrect[
            column
        ].mean()
    )


    coverage_stats[
        column
    ] = (
        correct_mean,
        incorrect_mean,
        correct_mean
        -
        incorrect_mean
    )


# ============================================================
# BOTH-HAND COVERAGE BUCKETS
# ============================================================

bucket_df = predictions_df.copy()


bucket_df[
    "both_hand_bucket"
] = pd.cut(

    bucket_df[
        "both_hands_coverage"
    ],

    bins=[
        -0.001,
        0.25,
        0.50,
        0.75,
        1.001
    ],

    labels=[
        "0-25%",
        "25-50%",
        "50-75%",
        "75-100%"
    ],

    include_lowest=True,

    right=False
)


bucket_rows = []


for bucket, group in bucket_df.groupby(
    "both_hand_bucket",
    observed=False
):

    if len(
        group
    ) == 0:

        continue


    bucket_rows.append(
        {
            "bucket":
                str(
                    bucket
                ),

            "samples":
                len(
                    group
                ),

            "top1_accuracy":
                100.0
                *
                group[
                    "correct"
                ].mean()
        }
    )


# ============================================================
# RUN ABLATIONS
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "HAND FEATURE ABLATION"
)

print(
    "=" * 78
)


ablation_results = [
    full_results
]


for mode in ABLATION_MODES[1:]:

    print(
        f"\nEvaluating: {mode}"
    )


    result, _ = evaluate_mode(

        mode,

        collect_predictions=False
    )


    ablation_results.append(
        result
    )


    print(
        f"Top-1: "
        f"{result['top1_accuracy']:.2f}%"
        f" | Top-5: "
        f"{result['top5_accuracy']:.2f}%"
    )


ablation_df = pd.DataFrame(
    ablation_results
)


full_top1 = float(
    full_results[
        "top1_accuracy"
    ]
)


full_top5 = float(
    full_results[
        "top5_accuracy"
    ]
)


ablation_df[
    "top1_drop_vs_full"
] = (

    full_top1

    -

    ablation_df[
        "top1_accuracy"
    ]
)


ablation_df[
    "top5_drop_vs_full"
] = (

    full_top5

    -

    ablation_df[
        "top5_accuracy"
    ]
)


ablation_df.to_csv(

    ABLATION_PATH,

    index=False
)


# ============================================================
# SUMMARY
# ============================================================

elapsed = (
    time.time()
    -
    start_time
)


summary_lines = [

    "=" * 78,

    "STEP 55 - HAND-CENTRIC V3 VALIDATION ANALYSIS",

    "=" * 78,

    "",

    f"Validation samples evaluated: "
    f"{len(predictions_df)}",

    "Training samples evaluated: 0",

    "Test samples evaluated: 0 ✅",

    "",

    "FULL V3:",

    f"Top-1 accuracy: "
    f"{full_results['top1_accuracy']:.2f}%",

    f"Top-5 accuracy: "
    f"{full_results['top5_accuracy']:.2f}%",

    f"Combined validation loss: "
    f"{full_results['combined_val_loss']:.4f}",

    f"Macro Top-1 accuracy: "
    f"{macro_top1:.2f}%",

    f"Macro Top-5 accuracy: "
    f"{macro_top5:.2f}%",

    "",

    f"Classes represented in validation: "
    f"{len(class_accuracy_df)}",

    f"Classes with 0% Top-1: "
    f"{zero_accuracy_classes}",

    f"Classes with >=75% Top-1: "
    f"{high_accuracy_classes}",

    f"Classes with 100% Top-1: "
    f"{perfect_classes}",

    "",

    "FULL-V3 FUSION GATES:",

    f"Base raw gate: "
    f"{full_results['base_raw_gate']:.4f}",

    f"Base velocity gate: "
    f"{full_results['base_velocity_gate']:.4f}",

    f"Base acceleration gate: "
    f"{full_results['base_acceleration_gate']:.4f}",

    f"Hand velocity gate: "
    f"{full_results['hand_velocity_gate']:.4f}",

    f"Hand acceleration gate: "
    f"{full_results['hand_acceleration_gate']:.4f}",

    f"Hand injection gate: "
    f"{full_results['hand_injection_gate']:.4f}",

    "",

    "HAND COVERAGE - CORRECT VS INCORRECT:"
]


for column, (
    correct_mean,
    incorrect_mean,
    difference
) in coverage_stats.items():

    pretty_name = (
        column
        .replace(
            "_coverage",
            ""
        )
        .replace(
            "_",
            " "
        )
        .title()
    )


    summary_lines.append(

        f"{pretty_name}: "
        f"correct={100.0 * correct_mean:.2f}%"
        f" | incorrect={100.0 * incorrect_mean:.2f}%"
        f" | difference={100.0 * difference:+.2f} points"
    )


summary_lines.extend(
    [
        "",
        "ACCURACY BY BOTH-HAND COVERAGE:"
    ]
)


for row in bucket_rows:

    summary_lines.append(

        f"{row['bucket']}: "
        f"{row['samples']} samples"
        f" | Top-1="
        f"{row['top1_accuracy']:.2f}%"
    )


summary_lines.extend(
    [
        "",
        "ABLATION:"
    ]
)


for _, row in ablation_df.iterrows():

    summary_lines.append(

        f"{row['mode']}: "
        f"Top-1={row['top1_accuracy']:.2f}%"
        f" | Top-5={row['top5_accuracy']:.2f}%"
        f" | Top-1 drop="
        f"{row['top1_drop_vs_full']:+.2f}"
    )


summary_lines.extend(
    [
        "",
        f"High-confidence errors "
        f"(confidence >= "
        f"{HIGH_CONFIDENCE_THRESHOLD:.2f}): "
        f"{len(high_confidence_errors_df)}",

        "",
        "TOP CONFUSIONS:"
    ]
)


for _, row in confusion_df.head(
    20
).iterrows():

    summary_lines.append(

        f"{row['true_gloss']} -> "
        f"{row['pred_gloss']}"
        f" | count={int(row['count'])}"
        f"/{int(row['true_class_samples'])}"
        f" | rate="
        f"{row['confusion_rate_percent']:.2f}%"
    )


summary_lines.extend(
    [
        "",
        "OUTPUT FILES:",

        str(
            PREDICTIONS_PATH
        ),

        str(
            CLASS_ACCURACY_PATH
        ),

        str(
            CONFUSIONS_PATH
        ),

        str(
            HIGH_CONFIDENCE_ERRORS_PATH
        ),

        str(
            ABLATION_PATH
        ),

        str(
            SUMMARY_PATH
        ),

        "",

        f"Analysis time: "
        f"{elapsed / 60:.2f} minutes",

        "",

        "IMPORTANT ABLATION NOTE:",

        "The 'no_hand_branch' result uses the already fine-tuned V3 "
        "backbone with the hand input removed. It is therefore not "
        "identical to the old Step-49 47.00% checkpoint. It measures "
        "how much the trained V3 currently depends on its hand branch.",

        "",

        "STATUS: STEP 55 VALIDATION ANALYSIS COMPLETED ✅",

        "TEST DATA USED: 0 ✅"
    ]
)


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
    "\n" + "\n".join(
        summary_lines
    )
)