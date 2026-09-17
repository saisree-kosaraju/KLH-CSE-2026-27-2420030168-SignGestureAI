"""Pydantic response models for the INCLUDE262 FastAPI backend."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RootResponse(BaseModel):
    service: str
    status: str
    docs: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int


class Top5Item(BaseModel):
    class_id: int
    sign: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_class_id: int
    predicted_sign: str
    confidence: float
    top5: List[Top5Item]
    modality_coverage: Dict[str, Any]
    preprocessing_time_ms: float
    inference_time_ms: float
    extraction_status: str


# ---------------------------------------------------------------------------
# Live / webcam single-sign recognition  (POST /predict/live)
# ---------------------------------------------------------------------------
class LivePredictionResponse(BaseModel):
    # verdict — always present
    accepted: bool
    rejection_reason: Optional[str] = None
    message: Optional[str] = None          # user-facing text when not accepted
    stage: str                             # "extraction" | "quality" | "complete"

    # prediction — present once the model has run (stage == "complete")
    predicted_class_id: Optional[int] = None
    predicted_sign: Optional[str] = None
    confidence: Optional[float] = None
    top1_sign: Optional[str] = None
    top1_confidence: Optional[float] = None
    top2_sign: Optional[str] = None
    top2_confidence: Optional[float] = None
    top2_margin: Optional[float] = None
    thresholds: Optional[Dict[str, float]] = None
    top5: List[Top5Item] = []

    # diagnostics
    modality_coverage: Optional[Dict[str, Any]] = None
    quality: Optional[Dict[str, Any]] = None
    trim: Optional[Dict[str, Any]] = None
    tensor_shapes: Optional[Dict[str, Any]] = None
    timings_ms: Optional[Dict[str, float]] = None
    extraction_status: Optional[str] = None
    checkpoint: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Vocabulary  (GET /vocab)
# ---------------------------------------------------------------------------
class VocabItem(BaseModel):
    class_id: int
    label: str
    score: Optional[float] = None
    exact: Optional[bool] = None


class VocabResponse(BaseModel):
    query: Optional[str] = None
    count: int
    total_classes: int
    exact_match: Optional[VocabItem] = None
    results: List[VocabItem]
    message: Optional[str] = None          # OUT_OF_VOCAB_MESSAGE when nothing matched


class ModelInfoResponse(BaseModel):
    checkpoint: str
    weights: str
    num_classes: int
    num_frames: int
    landmark_dim: int
    geometry_dim: int
    availability_dim: int
    include_test_top1: float
    variant: str
    epoch: int


# ---------------------------------------------------------------------------
# Dev-only webcam data collector  (POST /collector/sample, GET /collector/counts)
# ---------------------------------------------------------------------------
class CollectorSaveResponse(BaseModel):
    saved: bool
    filename: str
    split: str
    class_id: int
    label: str
    coverage: Dict[str, float]
    counts: Dict[str, Any]
    message: Optional[str] = None


class CollectorCountsResponse(BaseModel):
    enabled: bool
    total: int
    by_split: Dict[str, int]
    by_class: Dict[str, Any]
