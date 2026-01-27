import json
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.automl.data_profiler import profile_dataset
from src.automl.data_loader import load_dataset
from src.automl.preprocessor import preprocess
from src.automl.automl_engine import run_automl
from src.llm.mock_llm import mock_llm_decision

MODEL_NAME = "llm_automl_tabular_model"


def train_from_config(config_path: str) -> dict:
    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Load dataset
    df = load_dataset(
        config["dataset_path"],
        config["target_column"]
    )

    # Profile dataset
    dataset_profile = profile_dataset(df, config["target_column"])

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess(
        df,
        config["target_column"],
        test_size=config["test_size"],
        random_state=config["random_state"]
    )

    automl_cfg = config["automl"]

    with mlflow.start_run(run_name="AutoML_Run") as run:

        # 1. Log dataset profile
        mlflow.log_text(
            json.dumps(dataset_profile, indent=2),
            artifact_file="dataset_profile.json"
        )

        # 2. LLM decides which models to run
        llm_output = mock_llm_decision(
            dataset_profile,
            automl_cfg["models"]
        )

        mlflow.log_text(
            json.dumps(llm_output, indent=2),
            artifact_file="llm_decision.json"
        )

        models_to_run = llm_output["selected_models"]

        # 3. Run AutoML (training only)
        best_name, best_score, best_model = run_automl(
            models_to_run,
            X_train,
            X_test,
            y_train,
            y_test
        )

        # 4. Log metrics
        mlflow.log_metric("best_r2", best_score)
        mlflow.log_param("best_model", best_name)

        # 5. Log model artifact
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model"
        )

        run_id = run.info.run_id

    # 6. Register model OUTSIDE the run
    client = MlflowClient()
    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=MODEL_NAME
    )

    model_version = registered.version

    return {
        "best_model": best_name,
        "best_score": best_score,
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "run_id": run_id
    }
