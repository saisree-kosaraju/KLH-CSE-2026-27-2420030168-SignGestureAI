"""
WebM support helpers for POST /predict.

Nothing here touches the model, normalization, class mapping, or the 64-frame
preprocessing. It only makes a browser-recorded WebM clip readable by the
existing OpenCV-based pipeline, transcoding to MP4 only when OpenCV cannot use
the file directly.
"""

import os
import shutil
import subprocess
import tempfile

import cv2


def opencv_pipeline_compatible(path: str) -> bool:
    """
    True only when OpenCV can open the clip AND reports a positive frame count.

    scripts/76 `sample_64_frames()` rejects any clip whose CAP_PROP_FRAME_COUNT
    is <= 0. MediaRecorder WebM is decodable frame-by-frame but usually reports
    no (or negative) frame count, so it returns False here and gets transcoded.
    Well-formed MP4/MOV/WebM with proper metadata returns True and is used as-is.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            return False
        ok, frame = cap.read()
        return bool(ok) and frame is not None and getattr(frame, "size", 0) > 0
    finally:
        cap.release()


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def transcode_to_mp4(src_path: str) -> str:
    """
    Decode any ffmpeg-readable clip into an OpenCV-friendly H.264 MP4 and return
    its path. The caller is responsible for deleting the returned file.
    """
    fd, dst = tempfile.mkstemp(suffix=".mp4", prefix="include262_webm_")
    os.close(fd)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src_path,
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        dst,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (subprocess.TimeoutExpired, OSError) as exc:
        _safe_remove(dst)
        raise RuntimeError(f"ffmpeg transcode failed: {exc}") from exc

    if proc.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        _safe_remove(dst)
        raise RuntimeError(
            f"ffmpeg transcode failed: {(proc.stderr or '').strip()[:400]}"
        )

    return dst


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
