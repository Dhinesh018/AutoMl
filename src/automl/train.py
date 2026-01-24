import json
import mlflow
from src.automl.data_profiler import profile_dataset
from src.automl.data_loader import load_dataset
from src.automl.preprocessor import preprocess
from src.automl.automl_engine import run_automl
from src.llm.prompt_builder import build_model_selection_prompt
from src.llm.mock_llm import mock_llm_decision




def train_from_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = json.load(f)

    df = load_dataset(
        config["dataset_path"],
        config["target_column"]
    )

    dataset_profile = profile_dataset(df, config["target_column"])

    X_train, X_test, y_train, y_test = preprocess(
        df,
        config["target_column"],
        test_size=config["test_size"],
        random_state=config["random_state"]
    )

    automl_cfg = config["automl"]

    with mlflow.start_run(run_name="AutoML_Run"):

        # 1. Log dataset profile
        mlflow.log_text(
            json.dumps(dataset_profile, indent=2),
            artifact_file="dataset_profile.json"
        )

        # 2. LLM controls model selection
        llm_output = mock_llm_decision(
            dataset_profile,
            automl_cfg["models"]
        )

        mlflow.log_text(
            json.dumps(llm_output, indent=2),
            artifact_file="llm_decision.json"
        )

        models_to_run = llm_output["selected_models"]

        # 3. Run AutoML ONLY on LLM-approved models
        best_name, best_score, _ = run_automl(
            models_to_run,
            X_train, X_test,
            y_train, y_test
        )

        # 4. Log final decision
        mlflow.log_metric("best_r2", best_score)
        mlflow.log_param("best_model", best_name)

    # outside MLflow run, but inside function
    return {
        "best_model": best_name,
        "best_score": best_score
    }
