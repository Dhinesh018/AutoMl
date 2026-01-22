import json
import mlflow
from src.automl.data_profiler import profile_dataset
from src.automl.data_loader import load_dataset
from src.automl.preprocessor import preprocess
from src.automl.automl_engine import run_automl
from src.llm.reasoning import reason_about_models



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

    # 2. LLM-style reasoning (NEW)
        llm_decision = reason_about_models(
            dataset_profile,
            automl_cfg["models"]
    )

        mlflow.log_text(
            "\n".join(llm_decision["reasoning"]),
            artifact_file="llm_reasoning.txt"
    )

    # 3. Run AutoML ONLY on selected models
        best_name, best_score, _ = run_automl(
            llm_decision["selected_models"],
            X_train, X_test,
            y_train, y_test
    )

    # 4. Log decision
        mlflow.log_metric("best_r2", best_score)
        mlflow.log_param("best_model", best_name)

    return {
"best_model": best_name,
"best_score": best_score
}
