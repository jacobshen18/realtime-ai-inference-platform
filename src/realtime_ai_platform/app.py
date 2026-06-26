from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI

from realtime_ai_platform.metrics import LatencyMetrics
from realtime_ai_platform.model import LifecycleRiskModel
from realtime_ai_platform.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
)
from realtime_ai_platform.service import InferenceService


def create_app() -> FastAPI:
    app = FastAPI(
        title="Real-Time AI Inference Platform",
        version="0.1.0",
        description="Reference FastAPI platform for real-time AI inference and monitoring.",
    )

    model = LifecycleRiskModel()
    metrics = LatencyMetrics()
    service = InferenceService(model=model, metrics=metrics)

    @app.on_event("startup")
    def startup() -> None:
        model.warmup()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_version=model.version)

    @app.get("/ready", response_model=HealthResponse)
    def ready() -> HealthResponse:
        status = "ready" if model.is_warmed else "warming"
        return HealthResponse(status=status, model_version=model.version)

    @app.post("/v1/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        return service.predict(request)

    @app.post("/v1/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        started = perf_counter()
        predictions = [service.predict(item) for item in request.items]
        total_latency_ms = (perf_counter() - started) * 1000
        return BatchPredictionResponse(
            predictions=predictions,
            total_latency_ms=round(total_latency_ms, 3),
        )

    @app.get("/metrics", response_model=MetricsResponse)
    def get_metrics() -> MetricsResponse:
        return MetricsResponse(
            request_count=metrics.request_count,
            average_latency_ms=metrics.average_latency_ms,
            p95_latency_ms=metrics.p95_latency_ms,
        )

    return app
