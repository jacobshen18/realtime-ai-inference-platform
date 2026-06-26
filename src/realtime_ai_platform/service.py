from __future__ import annotations

from time import perf_counter

from realtime_ai_platform.metrics import LatencyMetrics
from realtime_ai_platform.model_loader import PredictiveModel
from realtime_ai_platform.schemas import PredictionRequest, PredictionResponse


class InferenceService:
    def __init__(self, model: PredictiveModel, metrics: LatencyMetrics) -> None:
        self.model = model
        self.metrics = metrics

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        started = perf_counter()
        prediction = self.model.predict(request.features)
        latency_ms = (perf_counter() - started) * 1000
        self.metrics.observe(latency_ms)

        return PredictionResponse(
            user_id=request.user_id,
            score=prediction.score,
            label=prediction.label,
            model_version=self.model.version,
            latency_ms=round(latency_ms, 3),
        )
