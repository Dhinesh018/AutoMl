import json
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.automl.data_profiler import profile_dataset
from src.automl.data_loader import load_dataset
from src.automl.preprocessor import preprocess
from src.automl.automl_engine import run_automl
from src.llm.mock_llm import mock_llm_decision
from src.config import MLFLOW_TRACKING_URI, MODEL_NAME

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


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
            artifact_path="model",
            input_example=X_train.iloc[:1]
        )

        run_id = run.info.run_id

    # 6. Register model OUTSIDE the run
    client = MlflowClient()
    
    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=MODEL_NAME
    )

    model_version = registered.version

    # 7. ✅ AUTO-PROMOTE TO PRODUCTION (NEW!)
    print(f"🚀 Auto-promoting model version {model_version} to Production...")
    
    # Archive old Production model (if exists)
    try:
        current_prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if current_prod:
            old_version = current_prod[0].version
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=old_version,
                stage="Archived"
            )
            print(f"📦 Archived old Production model (version {old_version})")
    except Exception as e:
        print(f"⚠️  No previous Production model to archive")

    # Promote new model to Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version,
        stage="Production"
    )
    print(f"✅ Model version {model_version} promoted to Production!")

    return {
        "best_model": best_name,
        "best_score": best_score,
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "run_id": run_id,
        "stage": "Production"  # ✅ Added this
    }