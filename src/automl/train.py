import json
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.automl.data_profiler import profile_dataset
from src.automl.data_loader import load_dataset
from src.automl.preprocessor import preprocess
from src.automl.automl_engine import run_automl
from src.llm.real_llm import get_llm_decision
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

    # 🔥 EXTRACT FEATURE NAMES (NEW!)
    target_column = config["target_column"]
    feature_columns = [col for col in df.columns if col != target_column]
    
    # Get feature types
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target from feature lists
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

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

        # 🔥 2. LOG FEATURE METADATA (NEW!)
        feature_metadata = {
            "features": feature_columns,
            "target": target_column,
            "num_features": len(feature_columns),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_dtypes": {col: str(df[col].dtype) for col in feature_columns}
        }
        
        mlflow.log_dict(
            feature_metadata,
            artifact_file="feature_metadata.json"
        )
        
        print(f"📊 Logged {len(feature_columns)} features for model")

        # 3. LLM decides which models to run
        llm_output = get_llm_decision(
            dataset_profile,
            automl_cfg["models"]
        )

        mlflow.log_text(
            json.dumps(llm_output, indent=2),
            artifact_file="llm_decision.json"
        )

        models_to_run = llm_output["selected_models"]

        # 4. Run AutoML (training only)
        best_name, best_score, best_model = run_automl(
            models_to_run,
            X_train,
            X_test,
            y_train,
            y_test
        )

        # Calculate additional metrics from best model
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np
        
        y_pred = best_model.predict(X_test)
        best_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        best_mae = mean_absolute_error(y_test, y_pred)

        # 5. Log all metrics
        mlflow.log_metric("best_r2", best_score)    
        mlflow.log_metric("best_rmse", best_rmse)
        mlflow.log_metric("best_mae", best_mae)
        mlflow.log_param("best_model", best_name)

        # 6. Log model artifact
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_train.iloc[:1]
        )

        run_id = run.info.run_id

    # 7. Register model OUTSIDE the run
    client = MlflowClient()
    
    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=MODEL_NAME
    )

    model_version = registered.version

    # 8. ✅ AUTO-PROMOTE TO PRODUCTION
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
        "best_rmse": best_rmse,
        "best_mae": best_mae,
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "run_id": run_id,
        "stage": "Production",
        "num_features": len(feature_columns)  # 🔥 NEW!
    }