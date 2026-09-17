import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

FEATURE_DIR = Path(
    "processed_data/WLASL300/landmarks_512_image"
)

MANIFEST_PATH = Path(
    "metadata/wlasl300_landmark_manifest_final.csv"
)

EXPECTED_COUNT = 2660
EXPECTED_SHAPE = (32, 382)


# =========================================================
# START
# =========================================================

print("=" * 75)
print("VERIFYING WLASL300 LANDMARK FEATURES")
print("=" * 75)


# =========================================================
# CHECK FOLDER
# =========================================================

if not FEATURE_DIR.exists():

    print(
        f"ERROR: Feature folder not found:\n"
        f"{FEATURE_DIR}"
    )

    raise SystemExit


# =========================================================
# GET ALL NPY FILES
# =========================================================

files = sorted(
    FEATURE_DIR.glob("*.npy")
)

print(
    f"\nLandmark files found: {len(files)}"
)

print(
    f"Expected files: {EXPECTED_COUNT}"
)


# =========================================================
# COUNTERS
# =========================================================

valid = 0
invalid_shape = 0
nan_files = 0
inf_files = 0
corrupt_files = 0

invalid_records = []


# =========================================================
# VERIFY EACH FILE
# =========================================================

for number, file_path in enumerate(
    files,
    start=1
):

    try:

        features = np.load(
            file_path
        )

        problems = []

        # ---------------------------------------------
        # Shape
        # ---------------------------------------------

        if features.shape != EXPECTED_SHAPE:

            invalid_shape += 1

            problems.append(
                f"shape={features.shape}"
            )

        # ---------------------------------------------
        # NaN
        # ---------------------------------------------

        nan_count = int(
            np.isnan(features).sum()
        )

        if nan_count > 0:

            nan_files += 1

            problems.append(
                f"NaN={nan_count}"
            )

        # ---------------------------------------------
        # Infinite
        # ---------------------------------------------

        inf_count = int(
            np.isinf(features).sum()
        )

        if inf_count > 0:

            inf_files += 1

            problems.append(
                f"Inf={inf_count}"
            )

        # ---------------------------------------------
        # Valid
        # ---------------------------------------------

        if len(problems) == 0:

            valid += 1

        else:

            invalid_records.append({
                "file": str(file_path),
                "problem": "; ".join(
                    problems
                )
            })

    except Exception as error:

        corrupt_files += 1

        invalid_records.append({
            "file": str(file_path),
            "problem": (
                f"Could not load: {error}"
            )
        })

    # ---------------------------------------------
    # Progress
    # ---------------------------------------------

    if (
        number % 250 == 0
        or
        number == len(files)
    ):

        print(
            f"Checked "
            f"{number}/{len(files)}"
        )


# =========================================================
# CHECK MANIFEST
# =========================================================

print("\n" + "=" * 75)
print("MANIFEST CHECK")
print("=" * 75)

if MANIFEST_PATH.exists():

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"Manifest rows: {len(manifest)}"
    )

    if "status" in manifest.columns:

        print(
            "\nStatus distribution:"
        )

        print(
            manifest[
                "status"
            ].value_counts(
                dropna=False
            )
        )

    if "split" in manifest.columns:

        print(
            "\nSplit distribution:"
        )

        print(
            manifest[
                "split"
            ].value_counts(
                dropna=False
            )
        )

    if "class_id" in manifest.columns:

        print(
            "\nUnique classes:"
        )

        print(
            manifest[
                "class_id"
            ].nunique()
        )

else:

    print(
        "Manifest file not found."
    )


# =========================================================
# INVALID FILE REPORT
# =========================================================

if invalid_records:

    invalid_df = pd.DataFrame(
        invalid_records
    )

    invalid_df.to_csv(
        "metadata/"
        "wlasl300_invalid_landmark_files.csv",
        index=False
    )

    print(
        "\nInvalid file report saved to:"
    )

    print(
        "metadata/"
        "wlasl300_invalid_landmark_files.csv"
    )


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 75)
print("WLASL300 LANDMARK VERIFICATION SUMMARY")
print("=" * 75)

print(
    f"Expected files: {EXPECTED_COUNT}"
)

print(
    f"Files found: {len(files)}"
)

print(
    f"Valid arrays: {valid}"
)

print(
    f"Invalid shape: {invalid_shape}"
)

print(
    f"Files containing NaN: {nan_files}"
)

print(
    f"Files containing Inf: {inf_files}"
)

print(
    f"Corrupt/unreadable files: {corrupt_files}"
)


# =========================================================
# FINAL DECISION
# =========================================================

if (
    len(files) == EXPECTED_COUNT
    and
    valid == EXPECTED_COUNT
    and
    invalid_shape == 0
    and
    nan_files == 0
    and
    inf_files == 0
    and
    corrupt_files == 0
):

    print(
        "\nSTATUS: WLASL300 LANDMARK FEATURES "
        "ARE FULLY VALID ✅"
    )

else:

    print(
        "\nSTATUS: SOME LANDMARK FILES "
        "NEED ATTENTION ❌"
    )

print("=" * 75)