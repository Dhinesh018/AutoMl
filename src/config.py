import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# MLflow configuration
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
MLFLOW_ARTIFACT_ROOT = str(PROJECT_ROOT / "mlflow_artifacts")

# Model configuration
MODEL_NAME = "llm_automl_tabular_model"
MODEL_STAGE = "Production"