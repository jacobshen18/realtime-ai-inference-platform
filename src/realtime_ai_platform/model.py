from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrediction:
    score: float
    label: str


class LifecycleRiskModel:
    """Deterministic toy model used to exercise the inference platform."""

    version = "lifecycle-risk-toy-v1"

    def __init__(self) -> None:
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
        engagement = _get(features, "engagement_score", 0.5)
        days_since_visit = _get(features, "days_since_last_visit", 0.0)
        support_contacts = _get(features, "support_contacts_30d", 0.0)
        tenure_days = _get(features, "plan_tenure_days", 0.0)

        logit = (
            -1.6
            - 2.2 * engagement
            + 0.085 * days_since_visit
            + 0.42 * support_contacts
            - 0.0012 * tenure_days
        )
        score = 1.0 / (1.0 + math.exp(-logit))
        return ModelPrediction(score=round(score, 6), label=_label(score))


def _get(features: dict[str, float], name: str, default: float) -> float:
    value = features.get(name, default)
    return float(value)


def _label(score: float) -> str:
    if score >= 0.7:
        return "high_risk"
    if score >= 0.35:
        return "medium_risk"
    return "low_risk"
