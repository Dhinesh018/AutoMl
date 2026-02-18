import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# MLflow configuration
# Uses environment variable if set (Docker), otherwise SQLite locally
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)

MLFLOW_ARTIFACT_ROOT = str(PROJECT_ROOT / "mlflow_artifacts")

# Model configuration
MODEL_NAME = "llm_automl_tabular_model"
MODEL_STAGE = "Production"

# Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY not found!")