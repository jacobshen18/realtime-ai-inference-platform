from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Log the demo lifecycle model as an MLflow pyfunc.")
    parser.add_argument("--output-uri", default="./mlruns", help="MLflow tracking URI.")
    parser.add_argument("--experiment-name", default="realtime-ai-inference-platform")
    parser.add_argument("--artifact-path", default="lifecycle-risk-model")
    parser.add_argument("--registered-model-name")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        import mlflow
        import mlflow.pyfunc
        import pandas as pd
    except ImportError as error:
        raise SystemExit("Install MLflow extras first: pip install -e '.[mlflow]'") from error

    from realtime_ai_platform.model import LifecycleRiskModel

    class LifecycleRiskPyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context) -> None:
            self.model = LifecycleRiskModel()
            self.model.warmup()

        def predict(self, context, model_input):
            rows = []
            for features in model_input.to_dict(orient="records"):
                prediction = self.model.predict(features)
                rows.append({"score": prediction.score, "label": prediction.label})
            return pd.DataFrame(rows)

    tracking_path = Path(args.output_uri)
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(tracking_path))
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path=args.artifact_path,
            python_model=LifecycleRiskPyfunc(),
            registered_model_name=args.registered_model_name,
        )
        model_uri = f"runs:/{run.info.run_id}/{args.artifact_path}"
        print(model_uri)


if __name__ == "__main__":
    main()
