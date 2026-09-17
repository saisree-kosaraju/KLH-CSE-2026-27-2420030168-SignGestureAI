"""
INCLUDE262 sign-recognition FastAPI backend.

Serving layer over app/backend:
    config / model_loader / preprocessing / inference.predict()      -> POST /predict   (frozen, upload tab)
    webcam_extract / trim / quality / rejection / live_inference     -> POST /predict/live (webcam tab)
    webcam.vocab                                                     -> GET  /vocab , GET /model-info
    collector_store (dev-only, INCLUDE262_COLLECTOR=1)               -> POST /collector/sample , GET /collector/counts

The model + class map are loaded once on startup and reused. Nothing outside
app/backend/ and webcam_data/ is written.
"""

import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config          # noqa: E402
import inference       # noqa: E402
import live_inference  # noqa: E402
import media           # noqa: E402
import schemas         # noqa: E402

sys.path.insert(0, str(config.ROOT))
from webcam import vocab as vocab_mod  # noqa: E402

logger = logging.getLogger("include262.backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}
ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "application/octet-stream",
}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB safety cap
FRONTEND_ORIGIN = "http://localhost:5173"
INCLUDE_TEST_TOP1 = 0.9804  # frozen locked-test result (script 84) — reference only


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.check_paths()
    inference._load_model("original")   # frozen checkpoint, used by /predict
    inference._load_class_map()
    live_variant = config.active_live_variant()
    if live_variant != "original":
        inference._load_model(live_variant)  # warm the adapted checkpoint for /predict/live
    app.state.model_loaded = True
    logger.info("INCLUDE262 model + class map loaded (%d classes). Live variant: %s.",
                config.NUM_CLASSES, live_variant)
    if config.COLLECTOR_ENABLED:
        logger.info("Webcam data collector endpoints ENABLED (INCLUDE262_COLLECTOR).")
    yield
    app.state.model_loaded = False


app = FastAPI(title="INCLUDE262 Sign Recognition API", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# shared upload handling
# ---------------------------------------------------------------------------
def _validate_type(filename: str, content_type: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ext not in ALLOWED_EXTENSIONS or (ctype and ctype not in ALLOWED_CONTENT_TYPES):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a single .mp4, .mov or .webm video.",
        )
    return ext


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file is too large.")
    return data


@contextmanager
def _video_source(data: bytes, ext: str):
    """Write the upload to a temp file OpenCV can read (transcoding WebM if needed). Cleans up."""
    fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="include262_upload_")
    converted = None
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        source = tmp_path
        if ext == ".webm" and not media.opencv_pipeline_compatible(tmp_path):
            if not media.ffmpeg_available():
                raise HTTPException(
                    status_code=422,
                    detail="This recording is in a WebM format the server cannot decode, and "
                           "ffmpeg is not available for conversion. Upload an MP4 or MOV instead.",
                )
            try:
                converted = media.transcode_to_mp4(tmp_path)
            except RuntimeError:
                logger.exception("WebM transcode failed")
                raise HTTPException(
                    status_code=422,
                    detail="The recorded WebM clip could not be converted for recognition. "
                           "Try recording again.",
                )
            source = converted
        yield source
    finally:
        for path in (tmp_path, converted):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    logger.warning("Could not delete temporary file: %s", path)


# ---------------------------------------------------------------------------
# health / info
# ---------------------------------------------------------------------------
@app.get("/", response_model=schemas.RootResponse)
def root():
    return schemas.RootResponse(
        service="INCLUDE262 Sign Recognition API",
        status="backend ready",
        docs="/docs",
    )


@app.get("/health", response_model=schemas.HealthResponse)
def health():
    return schemas.HealthResponse(
        status="ok",
        model_loaded=bool(getattr(app.state, "model_loaded", False)),
        num_classes=config.NUM_CLASSES,
    )


@app.get("/model-info", response_model=schemas.ModelInfoResponse)
def model_info():
    live_variant = config.active_live_variant()
    _, checkpoint = inference._load_model(live_variant)
    return schemas.ModelInfoResponse(
        checkpoint=os.path.basename(checkpoint.get("_path", config.CHECKPOINT.name)),
        weights="ema_state",
        num_classes=config.NUM_CLASSES,
        num_frames=config.NUM_FRAMES,
        landmark_dim=config.LANDMARK_DIM,
        geometry_dim=config.GEOMETRY_DIM,
        availability_dim=config.AVAILABILITY_DIM,
        include_test_top1=INCLUDE_TEST_TOP1,
        variant=live_variant,
        epoch=int(checkpoint.get("epoch", -1)),
    )


@app.get("/vocab", response_model=schemas.VocabResponse)
def vocab(q: str | None = Query(default=None), limit: int = Query(default=15, ge=1, le=262)):
    total = config.NUM_CLASSES
    if not q:
        rows = vocab_mod.all_classes()
        return schemas.VocabResponse(
            query=None, count=len(rows), total_classes=total,
            results=[schemas.VocabItem(**r) for r in rows],
        )
    results = vocab_mod.search(q, limit=limit)
    exact = next((r for r in results if r.get("exact")), None)
    return schemas.VocabResponse(
        query=q,
        count=len(results),
        total_classes=total,
        exact_match=schemas.VocabItem(**exact) if exact else None,
        results=[schemas.VocabItem(**r) for r in results],
        message=None if results else vocab_mod.OUT_OF_VOCAB_MESSAGE,
    )


# ---------------------------------------------------------------------------
# recognition
# ---------------------------------------------------------------------------
def _preprocess_error(exc: Exception) -> HTTPException:
    msg = str(exc)
    if any(tok in msg for tok in ("extraction failed", "shape", "Non-finite", "decodable",
                                  "cannot_open", "frame", "invalid_frame_count")):
        logger.warning("Preprocessing failure: %s", msg)
        return HTTPException(
            status_code=422,
            detail="Could not process the video. It may be corrupt, too short, or contain no visible signer.",
        )
    logger.exception("Inference failure")
    return HTTPException(status_code=500, detail="Inference failed.")


@app.post("/predict", response_model=schemas.PredictionResponse)
async def predict(file: UploadFile = File(...)):
    ext = _validate_type(file.filename, file.content_type)
    data = await _read_upload(file)
    with _video_source(data, ext) as source:
        try:
            result = inference.predict(source, topk=5)
        except (FileNotFoundError, RuntimeError) as exc:
            raise _preprocess_error(exc)
        except HTTPException:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected prediction error")
            raise HTTPException(status_code=500, detail="Internal server error.")

    return schemas.PredictionResponse(
        predicted_class_id=result["predicted_class_id"],
        predicted_sign=result["predicted_sign"],
        confidence=round(float(result["confidence"]), 4),
        top5=[
            schemas.Top5Item(
                class_id=item["class_id"],
                sign=item["sign"],
                probability=round(float(item["probability"]), 4),
            )
            for item in result["top5"]
        ],
        modality_coverage=result["modality_coverage"],
        preprocessing_time_ms=round(result["preprocess_seconds"] * 1000.0, 1),
        inference_time_ms=round(result["inference_seconds"] * 1000.0, 1),
        extraction_status=result["extraction_status"],
    )


@app.post("/predict/live", response_model=schemas.LivePredictionResponse)
async def predict_live(
    file: UploadFile = File(...),
    disable_trim: bool = Query(default=False, description="skip active-sign trimming (debug)"),
):
    """
    Live webcam single-sign recognition:
    active-sign trim -> quality gate -> exact training preprocessing -> model ->
    confidence/margin rejection. The frozen /predict path is unaffected.
    """
    ext = _validate_type(file.filename, file.content_type)
    data = await _read_upload(file)
    with _video_source(data, ext) as source:
        try:
            result = live_inference.predict_live(source, disable_trim=disable_trim)
        except (FileNotFoundError, RuntimeError) as exc:
            raise _preprocess_error(exc)
        except HTTPException:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected live prediction error")
            raise HTTPException(status_code=500, detail="Internal server error.")

    return schemas.LivePredictionResponse(**result)


# ---------------------------------------------------------------------------
# dev-only webcam data collector
# ---------------------------------------------------------------------------
if config.COLLECTOR_ENABLED:
    import collector_store  # noqa: E402

    @app.get("/collector/counts", response_model=schemas.CollectorCountsResponse)
    def collector_counts():
        c = collector_store.counts()
        return schemas.CollectorCountsResponse(
            enabled=True, total=c["total"], by_split=c["by_split"], by_class=c["by_class"],
        )

    @app.post("/collector/sample", response_model=schemas.CollectorSaveResponse)
    async def collector_sample(
        file: UploadFile = File(...),
        class_id: int = Form(...),
        label: str = Form(""),
        split: str = Form(...),
        session_id: str = Form(""),
        background_id: str = Form(""),
        lighting: str = Form(""),
        position: str = Form(""),
        clothing: str = Form(""),
        camera_distance: str = Form(""),
        duration_s: str = Form(""),
        width: str = Form(""),
        height: str = Form(""),
        fps: str = Form(""),
        left_coverage: str = Form(""),
        right_coverage: str = Form(""),
        pose_coverage: str = Form(""),
        face_coverage: str = Form(""),
        notes: str = Form(""),
        app_version: str = Form(""),
    ):
        ext = _validate_type(file.filename, file.content_type)
        data = await _read_upload(file)
        meta = dict(
            class_id=class_id, label=label, split=split, session_id=session_id,
            background_id=background_id, lighting=lighting, position=position,
            clothing=clothing, camera_distance=camera_distance, duration_s=duration_s,
            width=width, height=height, fps=fps, left_coverage=left_coverage,
            right_coverage=right_coverage, pose_coverage=pose_coverage,
            face_coverage=face_coverage, notes=notes, app_version=app_version,
        )
        try:
            return schemas.CollectorSaveResponse(**collector_store.save_sample(data, ext, meta))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
