from realtime_ai_platform.model import LifecycleRiskModel


def test_model_warmup_sets_ready_state() -> None:
    model = LifecycleRiskModel()

    model.warmup()

    assert model.is_warmed


def test_model_prediction_is_bounded() -> None:
    model = LifecycleRiskModel()
    prediction = model.predict(
        {
            "engagement_score": 0.2,
            "days_since_last_visit": 20,
            "support_contacts_30d": 2,
            "plan_tenure_days": 40,
        }
    )

    assert 0 <= prediction.score <= 1
    assert prediction.label in {"low_risk", "medium_risk", "high_risk"}
