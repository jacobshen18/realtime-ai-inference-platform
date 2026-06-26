from fastapi.testclient import TestClient

from realtime_ai_platform.app import create_app


def test_health_endpoint() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/predict",
        json={
            "user_id": "user-123",
            "features": {
                "engagement_score": 0.82,
                "days_since_last_visit": 2,
                "support_contacts_30d": 0,
                "plan_tenure_days": 420,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "user-123"
    assert 0 <= payload["score"] <= 1
    assert payload["model_version"] == "lifecycle-risk-toy-v1"


def test_batch_predict_endpoint() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/predict/batch",
        json={
            "items": [
                {
                    "user_id": "user-123",
                    "features": {"engagement_score": 0.82},
                },
                {
                    "user_id": "user-456",
                    "features": {"engagement_score": 0.12, "days_since_last_visit": 30},
                },
            ]
        },
    )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
