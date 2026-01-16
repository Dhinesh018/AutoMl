import json
import mlflow

from src.automl.data_loader import load_dataset
from src.automl.preprocessor import preprocess
from src.automl.automl_engine import run_automl


def train_from_config(config_path: str):
    with open(config_path) as f:
        config = json.load(f)

    df = load_dataset(
        config["dataset_path"],
        config["target_column"]
    )

    X_train, X_test, y_train, y_test = preprocess(
        df,
        config["target_column"],
        test_size=config["test_size"],
        random_state=config["random_state"]
    )

    automl_cfg = config["automl"]

    with mlflow.start_run(run_name="AutoML_Run"):
        best_name, best_score, _ = run_automl(
            automl_cfg["models"],
            X_train, X_test,
            y_train, y_test
        )

        mlflow.log_metric("best_r2", best_score)
        mlflow.log_param("best_model", best_name)

    return {
        "best_model": best_name,
        "best_score": best_score
    }
