from pathlib import Path
import re
import zipfile

ROOT = Path.cwd()
DATASET = ROOT / "datasets" / "INCLUDE"
DESKTOP = Path.home() / "Desktop"

MISSING = [
    "clothing",
    "pant",
    "pocket",
    "secondnumber",
    "shoes",
    "tshirt",
]

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}


def norm(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def main():
    print("=" * 78)
    print("INCLUDE STEP 4 - LOCATE MISSING 6 CLASSES LOCALLY")
    print("=" * 78)

    print("\nMissing classes:")
    for x in MISSING:
        print(f"  - {x}")

    # ------------------------------------------------------------
    # 1) Inspect extracted dataset folders/files
    # ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("A. SEARCH EXTRACTED INCLUDE DATASET")
    print("-" * 78)

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET}")

    all_paths = list(DATASET.rglob("*"))

    extracted_hits = {label: [] for label in MISSING}

    for p in all_paths:
        rel = str(p.relative_to(DATASET))
        n = norm(rel)

        for label in MISSING:
            if norm(label) in n:
                extracted_hits[label].append(rel)

    for label in MISSING:
        hits = extracted_hits[label]
        print(f"{label:15s}: {len(hits)} extracted-path matches")
        for h in hits[:10]:
            print(f"    {h}")

    # ------------------------------------------------------------
    # 2) Inspect all ZIP archives under project + Desktop
    # ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("B. SEARCH ZIP ARCHIVES WITHOUT EXTRACTING")
    print("-" * 78)

    roots = []
    for candidate in [ROOT, DESKTOP]:
        if candidate.exists():
            roots.append(candidate)

    zip_paths = []
    seen = set()

    for base in roots:
        for z in base.rglob("*.zip"):
            try:
                key = str(z.resolve()).lower()
            except Exception:
                key = str(z).lower()

            if key not in seen:
                seen.add(key)
                zip_paths.append(z)

    print(f"\nZIP archives found: {len(zip_paths)}")

    archive_hits = {label: [] for label in MISSING}

    for i, zpath in enumerate(zip_paths, start=1):
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                names = zf.namelist()

                for member in names:
                    nm = norm(member)

                    for label in MISSING:
                        if norm(label) in nm:
                            archive_hits[label].append(
                                (str(zpath), member)
                            )

        except zipfile.BadZipFile:
            print(f"WARNING: bad ZIP skipped: {zpath}")
        except Exception as e:
            print(f"WARNING: could not inspect {zpath}: {e}")

        if i % 10 == 0 or i == len(zip_paths):
            print(f"Inspected ZIPs: {i}/{len(zip_paths)}")

    print("\n" + "-" * 78)
    print("ZIP RESULTS")
    print("-" * 78)

    any_zip_hit = False

    for label in MISSING:
        hits = archive_hits[label]

        print(f"\n{label}: {len(hits)} ZIP member matches")

        for zpath, member in hits[:20]:
            any_zip_hit = True
            print(f"  ZIP   : {zpath}")
            print(f"  MEMBER: {member}")

    # ------------------------------------------------------------
    # 3) Final decision
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("DECISION")
    print("=" * 78)

    missing_extracted = [
        label
        for label in MISSING
        if len(extracted_hits[label]) == 0
    ]

    missing_everywhere = [
        label
        for label in MISSING
        if (
            len(extracted_hits[label]) == 0
            and len(archive_hits[label]) == 0
        )
    ]

    print(
        f"Missing from extracted dataset: "
        f"{len(missing_extracted)}/{len(MISSING)}"
    )
    print(
        f"Missing from extracted dataset AND all local ZIPs: "
        f"{len(missing_everywhere)}/{len(MISSING)}"
    )

    if missing_everywhere:
        print("\nNeed re-download/recovery from official INCLUDE source:")
        for label in missing_everywhere:
            print(f"  - {label}")

    if any_zip_hit:
        print(
            "\nAt least one missing class exists inside a local ZIP. "
            "Do NOT re-download the whole dataset yet."
        )
    else:
        print(
            "\nNone of the six missing classes were found in local ZIP archives. "
            "Recovery from the official dataset source is required."
        )


if __name__ == "__main__":
    main()