from pathlib import Path
import cv2
import pandas as pd
import time

ROOT = Path.cwd()
META = ROOT / "metadata"
RESULTS = ROOT / "results"

INPUT_CSV = META / "include262_official_split_final.csv"
OUTPUT_CSV = META / "include262_video_integrity_resumable.csv"
FAILED_CSV = META / "include262_video_integrity_failed.csv"
SUMMARY_TXT = RESULTS / "include262_video_integrity_resumable_summary.txt"

SAVE_EVERY = 25
OPEN_TIMEOUT_MS = 10000
READ_TIMEOUT_MS = 10000


def open_video(path):
    """
    Prefer FFmpeg backend with explicit open/read timeouts.
    Fall back to normal VideoCapture if timeout properties are unsupported.
    """
    try:
        cap = cv2.VideoCapture(
            str(path),
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS,
            ],
        )
        if cap.isOpened():
            return cap
        cap.release()
    except Exception:
        pass

    return cv2.VideoCapture(str(path))


def check_video_sequential(path):
    """
    Decode sequentially instead of using CAP_PROP_POS_FRAMES seeking.
    This avoids the random-seek stalls seen in some INCLUDE MOV files.
    """
    p = Path(path)

    result = {
        "video_path": str(p),
        "exists": p.exists(),
        "open_ok": False,
        "decoded_frames": 0,
        "reported_frame_count": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "integrity_ok": False,
        "reason": "",
        "elapsed_sec": 0.0,
    }

    start = time.time()

    if not p.exists():
        result["reason"] = "file_missing"
        result["elapsed_sec"] = time.time() - start
        return result

    cap = open_video(p)

    if not cap.isOpened():
        result["reason"] = "cannot_open"
        result["elapsed_sec"] = time.time() - start
        cap.release()
        return result

    result["open_ok"] = True
    result["reported_frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
    result["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    result["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    decoded = 0
    bad_frame = False

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame is None or frame.size == 0:
            bad_frame = True
            break

        decoded += 1

    cap.release()

    result["decoded_frames"] = decoded
    result["elapsed_sec"] = time.time() - start

    if decoded <= 0:
        result["reason"] = "no_decodable_frames"
        return result

    if result["width"] <= 0 or result["height"] <= 0:
        result["reason"] = "invalid_resolution"
        return result

    if bad_frame:
        result["reason"] = "empty_decoded_frame"
        return result

    # Some containers report an inaccurate CAP_PROP_FRAME_COUNT, so we do not
    # fail merely because decoded_frames != reported_frame_count.
    result["integrity_ok"] = True
    result["reason"] = "ok"
    return result


def save_checkpoint(rows):
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)

    if not out.empty and "integrity_ok" in out.columns:
        failed = out[~out["integrity_ok"].astype(bool)].copy()
        failed.to_csv(FAILED_CSV, index=False)


def main():
    print("=" * 84)
    print("INCLUDE STEP 13 - RESUMABLE VIDEO INTEGRITY CHECK")
    print("SEQUENTIAL DECODING + OPENCV/FFMPEG TIMEOUTS")
    print("=" * 84)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Final metadata not found:\n{INPUT_CSV}")

    RESULTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    required = {"video_path", "video_name", "label", "class_id", "split"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise RuntimeError(f"Missing required columns: {sorted(missing_cols)}")

    # Resume support
    completed = {}
    rows = []

    if OUTPUT_CSV.exists():
        old = pd.read_csv(OUTPUT_CSV)
        if not old.empty and "video_path" in old.columns:
            for _, r in old.iterrows():
                completed[str(r["video_path"])] = r.to_dict()
                rows.append(r.to_dict())

        print(f"Existing checkpoint found: {len(completed)} videos already checked")
    else:
        print("No previous resumable checkpoint found. Starting clean.")

    total = len(df)
    started = time.time()
    newly_checked = 0

    print(f"Total videos: {total}")
    print(f"Classes     : {df['class_id'].nunique()}")
    print(f"Splits      : {df['split'].value_counts().to_dict()}")
    print()

    try:
        for idx, meta_row in df.iterrows():
            video_path = str(meta_row["video_path"])

            if video_path in completed:
                continue

            result = check_video_sequential(video_path)

            result.update({
                "video_name": meta_row["video_name"],
                "label": meta_row["label"],
                "class_id": int(meta_row["class_id"]),
                "split": meta_row["split"],
                "metadata_index": int(idx),
            })

            rows.append(result)
            completed[video_path] = result
            newly_checked += 1

            if (newly_checked % SAVE_EVERY == 0) or (len(completed) == total):
                save_checkpoint(rows)

                good = sum(bool(r.get("integrity_ok", False)) for r in rows)
                bad = len(rows) - good
                elapsed = (time.time() - started) / 60

                print(
                    f"Checked {len(completed)}/{total} | "
                    f"OK={good} | Failed={bad} | "
                    f"New this run={newly_checked} | "
                    f"Time={elapsed:.1f} min"
                )

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Saving checkpoint before exit...")
        save_checkpoint(rows)
        print(f"Saved {len(rows)} checked videos to:\n{OUTPUT_CSV}")
        print("Run the same command again to resume.")
        return

    save_checkpoint(rows)

    out = pd.read_csv(OUTPUT_CSV)

    good = int(out["integrity_ok"].astype(bool).sum())
    bad = len(out) - good

    failed = out[~out["integrity_ok"].astype(bool)].copy()

    split_failures = (
        failed["split"].value_counts().to_dict()
        if not failed.empty
        else {}
    )

    reasons = (
        failed["reason"].value_counts().to_dict()
        if not failed.empty
        else {}
    )

    slowest = (
        out.sort_values("elapsed_sec", ascending=False)
        [["video_name", "label", "split", "elapsed_sec", "video_path"]]
        .head(10)
    )

    print("\n" + "=" * 84)
    print("FINAL RESUMABLE INTEGRITY RESULT")
    print("=" * 84)
    print(f"Total checked : {len(out)} / {total}")
    print(f"Valid videos  : {good}")
    print(f"Failed videos : {bad}")
    print(f"Failures/split: {split_failures}")
    print(f"Reasons       : {reasons}")

    print("\n10 slowest videos:")
    print(slowest.to_string(index=False))

    summary_lines = [
        "=" * 84,
        "INCLUDE 262 RESUMABLE VIDEO INTEGRITY SUMMARY",
        "=" * 84,
        "",
        "Method: sequential OpenCV decoding; no random frame seeking.",
        f"Open timeout: {OPEN_TIMEOUT_MS} ms",
        f"Read timeout: {READ_TIMEOUT_MS} ms",
        "",
        f"Total checked: {len(out)} / {total}",
        f"Valid videos: {good}",
        f"Failed videos: {bad}",
        f"Failures by split: {split_failures}",
        f"Failure reasons: {reasons}",
        "",
        f"Full report: {OUTPUT_CSV}",
        f"Failed-only report: {FAILED_CSV}",
        "",
        "STATUS:",
    ]

    if len(out) == total and bad == 0:
        summary_lines.append("ALL 4284 VIDEOS PASSED SEQUENTIAL DECODING ✅")
        summary_lines.append(
            "NEXT: 64-frame MediaPipe feature extraction with the updated architecture."
        )
    elif len(out) == total:
        summary_lines.append("INTEGRITY CHECK FINISHED WITH FAILURES ⚠️")
        summary_lines.append(
            "Review failed-only CSV before MediaPipe extraction."
        )
    else:
        summary_lines.append("CHECK INCOMPLETE - RUN SAME COMMAND AGAIN TO RESUME ⚠️")

    SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "\n".join(summary_lines))
    print("=" * 84)


if __name__ == "__main__":
    main()