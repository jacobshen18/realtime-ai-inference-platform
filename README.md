# Real-Time AI Inference Platform

A compact reference project for productionizing real-time AI systems.

This repository demonstrates the core pieces of an inference service: MLflow-backed model lifecycle
management, real-time and batch prediction APIs, request validation, latency metrics, health checks,
Docker packaging, and tests.

The model is intentionally lightweight and deterministic so the platform code is easy to run and
review without GPU access or external model downloads.

## Why This Project

Production AI work is not only model training. A useful model needs reliable inference paths,
observable behavior, safe rollouts, and clear interfaces for product systems. This project focuses
on those engineering layers.

## Features

- FastAPI inference service
- Optional MLflow model loading with `MLFLOW_MODEL_URI`
- Real-time single prediction endpoint
- Batch prediction endpoint
- Model warmup on application startup
- Pydantic request and response schemas
- In-memory latency and throughput metrics
- Health and readiness endpoints
- Dockerfile and local run commands
- Unit tests for model, service, and metrics behavior
- GitHub Actions CI

## API

### Health

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### Real-Time Prediction

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d @examples/predict.json
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d @examples/batch_predict.json
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn realtime_ai_platform.app:create_app --factory --reload
```

Run tests:

```bash
pytest
```

## MLflow Model Path

The service can run with the built-in deterministic model, or load an MLflow pyfunc model at startup:

```bash
pip install -e ".[dev,mlflow]"
python scripts/log_mlflow_model.py --output-uri ./mlruns
export MLFLOW_MODEL_URI="runs:/<run_id>/lifecycle-risk-model"
uvicorn realtime_ai_platform.app:create_app --factory --reload
```

In production, `MLFLOW_MODEL_URI` can point to a model registry alias or version, such as:

```bash
export MLFLOW_MODEL_URI="models:/LifecycleRiskModel@champion"
```

## Docker

```bash
docker build -t realtime-ai-inference-platform .
docker run --rm -p 8000:8000 realtime-ai-inference-platform
```

## Architecture

```text
client
  |
  v
FastAPI routes
  |
  v
request schema -> model service -> prediction response
                  |
                  v
             latency metrics
```

## Production Notes

In a real deployment, this skeleton would usually connect to:

- A model artifact registry
- MLflow model registry aliases for champion/challenger deployments
- Feature stores or online feature services
- Canary and shadow deployments
- Distributed tracing and structured logs
- Autoscaling and load shedding
- Model drift and data quality monitors

## License

MIT
