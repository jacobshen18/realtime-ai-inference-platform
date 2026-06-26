from __future__ import annotations

import os
from typing import Protocol

from realtime_ai_platform.model import LifecycleRiskModel, ModelPrediction


class PredictiveModel(Protocol):
    version: str

    @property
    def is_warmed(self) -> bool: ...

    def warmup(self) -> None: ...

    def predict(self, features: dict[str, float]) -> ModelPrediction: ...


def load_model_from_environment() -> PredictiveModel:
    model_uri = os.getenv("MLFLOW_MODEL_URI")
    if model_uri:
        return MlflowPyfuncModelAdapter(model_uri)
    return LifecycleRiskModel()


class MlflowPyfuncModelAdapter:
    def __init__(self, model_uri: str) -> None:
        try:
            import mlflow.pyfunc
            import pandas as pd
        except ImportError as error:
            raise RuntimeError(
                "MLFLOW_MODEL_URI is set, but MLflow extras are not installed. "
                "Install with: pip install -e '.[mlflow]'"
            ) from error

        self.model_uri = model_uri
        self.version = model_uri
        self._pd = pd
        self._model = mlflow.pyfunc.load_model(model_uri)
        self._is_warmed = False

    @property
    def is_warmed(self) -> bool:
        return self._is_warmed

    def warmup(self) -> None:
        self.predict(
            {
                "engagement_score": 0.5,
                "days_since_last_visit": 7.0,
                "support_contacts_30d": 0.0,
                "plan_tenure_days": 180.0,
            }
        )
        self._is_warmed = True

    def predict(self, features: dict[str, float]) -> ModelPrediction:
        frame = self._pd.DataFrame([features])
        raw_prediction = self._model.predict(frame)
        return _coerce_prediction(raw_prediction)


def _coerce_prediction(raw_prediction) -> ModelPrediction:
    if hasattr(raw_prediction, "iloc"):
        first = raw_prediction.iloc[0]
        if hasattr(first, "to_dict"):
            payload = first.to_dict()
            return ModelPrediction(score=round(float(payload["score"]), 6), label=str(payload["label"]))
        return ModelPrediction(score=round(float(first), 6), label=_label(float(first)))

    if isinstance(raw_prediction, list):
        first = raw_prediction[0]
        if isinstance(first, dict):
            return ModelPrediction(score=round(float(first["score"]), 6), label=str(first["label"]))
        return ModelPrediction(score=round(float(first), 6), label=_label(float(first)))

    if isinstance(raw_prediction, dict):
        return ModelPrediction(
            score=round(float(raw_prediction["score"]), 6),
            label=str(raw_prediction["label"]),
        )

    score = float(raw_prediction)
    return ModelPrediction(score=round(score, 6), label=_label(score))


def _label(score: float) -> str:
    if score >= 0.7:
        return "high_risk"
    if score >= 0.35:
        return "medium_risk"
    return "low_risk"
