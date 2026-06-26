from realtime_ai_platform.model import LifecycleRiskModel
from realtime_ai_platform.model_loader import load_model_from_environment


def test_load_model_from_environment_uses_builtin_model_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_MODEL_URI", raising=False)

    model = load_model_from_environment()

    assert isinstance(model, LifecycleRiskModel)
