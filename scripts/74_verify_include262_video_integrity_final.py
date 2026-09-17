from pathlib import Path
import cv2
import pandas as pd
import time

ROOT = Path.cwd()
META = ROOT / "metadata"
RESULTS = ROOT / "results"

INPUT_CSV = META / "include262_official_split_final.csv"
OUTPUT_CSV = META / "include262_video_integrity_final.csv"
SUMMARY_TXT = RESULTS / "include262_video_integrity_final_summary.txt"


def read_frame_at(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, frame = cap.read()
    if not ok or frame is None:
        return False
    if frame.size == 0:
        return False
    return True


def check_video(path):
    p = Path(path)

    result = {
        "video_path": str(p),
        "exists": p.exists(),
        "open_ok": False,
        "frame_count": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "first_frame_ok": False,
        "middle_frame_ok": False,
        "last_frame_ok": False,
        "integrity_ok": False,
        "reason": "",
    }

    if not p.exists():
        result["reason"] = "file_missing"
        return result

    cap = cv2.VideoCapture(str(p))

    if not cap.isOpened():
        result["reason"] = "cannot_open"
        cap.release()
        return result

    result["open_ok"] = True

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    result["frame_count"] = frame_count
    result["fps"] = fps
    result["width"] = width
    result["height"] = height

    if frame_count <= 0:
        result["reason"] = "invalid_frame_count"
        cap.release()
        return result

    if width <= 0 or height <= 0:
        result["reason"] = "invalid_resolution"
        cap.release()
        return result

    first_idx = 0
    middle_idx = frame_count // 2
    last_idx = max(0, frame_count - 1)

    result["first_frame_ok"] = read_frame_at(cap, first_idx)
    result["middle_frame_ok"] = read_frame_at(cap, middle_idx)
    result["last_frame_ok"] = read_frame_at(cap, last_idx)

    cap.release()

    if not result["first_frame_ok"]:
        result["reason"] = "first_frame_decode_failed"
        return result

    if not result["middle_frame_ok"]:
        result["reason"] = "middle_frame_decode_failed"
        return result

    if not result["last_frame_ok"]:
        result["reason"] = "last_frame_decode_failed"
        return result

    result["integrity_ok"] = True
    result["reason"] = "ok"
    return result


def main():
    print("=" * 82)
    print("INCLUDE STEP 12 - FINAL VIDEO INTEGRITY VERIFICATION")
    print("=" * 82)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Final metadata not found:\n{INPUT_CSV}"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    required = {"video_path", "split", "label", "class_id"}
    missing_cols = required - set(df.columns)

    if missing_cols:
        raise RuntimeError(
            f"Missing required metadata columns: {sorted(missing_cols)}"
        )

    print(f"Videos to verify: {len(df)}")
    print(f"Classes          : {df['class_id'].nunique()}")
    print(f"Split counts     : {df['split'].value_counts().to_dict()}")
    print()

    started = time.time()
    rows = []

    total = len(df)

    for i, row in df.iterrows():
        result = check_video(row["video_path"])

        result.update({
            "video_name": row.get("video_name", Path(row["video_path"]).name),
            "label": row["label"],
            "class_id": int(row["class_id"]),
            "split": row["split"],
        })

        rows.append(result)

        done = i + 1

        if done % 100 == 0 or done == total:
            elapsed = time.time() - started
            good = sum(1 for r in rows if r["integrity_ok"])
            bad = done - good

            print(
                f"Checked {done}/{total} | "
                f"OK={good} | Failed={bad} | "
                f"Time={elapsed/60:.1f} min"
            )

    out = pd.DataFrame(rows)

    out = out[
        [
            "video_name",
            "video_path",
            "label",
            "class_id",
            "split",
            "exists",
            "open_ok",
            "frame_count",
            "fps",
            "width",
            "height",
            "first_frame_ok",
            "middle_frame_ok",
            "last_frame_ok",
            "integrity_ok",
            "reason",
        ]
    ]

    out.to_csv(OUTPUT_CSV, index=False)

    total_ok = int(out["integrity_ok"].sum())
    total_bad = int((~out["integrity_ok"]).sum())

    bad_df = out[~out["integrity_ok"]].copy()

    train_bad = int(
        ((out["split"] == "train") & (~out["integrity_ok"])).sum()
    )
    val_bad = int(
        ((out["split"] == "val") & (~out["integrity_ok"])).sum()
    )
    test_bad = int(
        ((out["split"] == "test") & (~out["integrity_ok"])).sum()
    )

    reason_counts = (
        bad_df["reason"].value_counts().to_dict()
        if not bad_df.empty
        else {}
    )

    print("\n" + "=" * 82)
    print("FINAL INTEGRITY RESULT")
    print("=" * 82)
    print(f"Total videos : {len(out)}")
    print(f"Valid videos : {total_ok}")
    print(f"Failed videos: {total_bad}")
    print(f"Train failed : {train_bad}")
    print(f"Val failed   : {val_bad}")
    print(f"Test failed  : {test_bad}")

    if reason_counts:
        print(f"Failure reasons: {reason_counts}")

    summary_lines = [
        "=" * 82,
        "INCLUDE 262 FINAL VIDEO INTEGRITY SUMMARY",
        "=" * 82,
        "",
        f"Total videos: {len(out)}",
        f"Valid videos: {total_ok}",
        f"Failed videos: {total_bad}",
        "",
        f"Train failed: {train_bad}",
        f"Val failed: {val_bad}",
        f"Test failed: {test_bad}",
        "",
        f"Failure reasons: {reason_counts}",
        "",
        f"Detailed report: {OUTPUT_CSV}",
        "",
        "STATUS:",
    ]

    if total_bad == 0 and len(out) == 4284:
        summary_lines.append("ALL 4284 VIDEOS PASSED INTEGRITY CHECK ✅")
        summary_lines.append(
            "NEXT: extract 64-frame MediaPipe landmark features."
        )
    else:
        summary_lines.append(
            "ONE OR MORE VIDEOS FAILED INTEGRITY CHECK ⚠️"
        )
        summary_lines.append(
            "Review failed rows before feature extraction."
        )

    SUMMARY_TXT.write_text(
        "\n".join(summary_lines),
        encoding="utf-8"
    )

    print("\n" + "\n".join(summary_lines))
    print("=" * 82)


if __name__ == "__main__":
    main()