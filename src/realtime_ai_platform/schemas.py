from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    user_id: str = Field(min_length=1)
    features: dict[str, float]
    context: dict[str, str] = Field(default_factory=dict)


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest] = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    user_id: str
    score: float = Field(ge=0.0, le=1.0)
    label: str
    model_version: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str


class MetricsResponse(BaseModel):
    request_count: int
    average_latency_ms: float
    p95_latency_ms: float
