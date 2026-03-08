from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException

from src.automl.train import train_from_config
from src.config import MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_STAGE
from fastapi import UploadFile, Form
from src.api.upload import upload_dataset

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------------- APP ----------------
app = FastAPI(
    title="LLM-Augmented AutoML Assistant",
    version="0.1.0"
)

# ---------------- CONSTANTS ----------------
EXPECTED_FEATURES = [
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "GrLivArea",
    "FullBath",
    "GarageCars"
]


# ---------------- LOAD MODEL ON STARTUP ----------------
def load_model():
    """Load model from Production stage. Returns None if not found."""
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        model_versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        version = int(model_versions[0].version) if model_versions else 0
        
        return loaded_model, version
    except MlflowException:
        # Don't crash - return None instead
        return None, None

try:
    model, model_version = load_model()
    print(f"✅ Loaded model version {model_version} from Production")
except Exception as e:
    print(f"⚠️  No Production model found: {e}")
    model, model_version = None, None

# ---------------- SCHEMAS ----------------
class TrainRequest(BaseModel):
    config_path: str


class TrainResponse(BaseModel):
    best_model: str
    best_score: float
    run_id: str


class PredictRequest(BaseModel):
    features: dict


class PredictResponse(BaseModel):
    prediction: float
    model_name: str
    model_stage: str
    model_version: int


# ---------------- ENDPOINTS ----------------
@app.post("/train", response_model=TrainResponse)
def train_model(req: TrainRequest):
    try:
        return train_from_config(req.config_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/datasets/upload")
async def upload_dataset_endpoint(
    file: UploadFile,
    target_column: str = Form(...)
):
    """
    Upload a dataset for training
    
    Parameters:
    - file: CSV or Excel file
    - target_column: Name of the target column
    
    Returns:
    - dataset_id: Unique identifier for the dataset
    - Dataset statistics and profile
    """
    return await upload_dataset(file, target_column)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Check if model exists
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No model in Production stage. Train a model first using /train endpoint."
        )
    
    try:
        incoming_features = req.features

        # Check missing fields
        missing = set(EXPECTED_FEATURES) - set(incoming_features.keys())
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {missing}"
            )

        # Check unexpected fields
        extra = set(incoming_features.keys()) - set(EXPECTED_FEATURES)
        if extra:
            raise HTTPException(
                status_code=400,
                detail=f"Unexpected features: {extra}"
            )

        df = pd.DataFrame([incoming_features])
        prediction = model.predict(df)[0]

        # Log before returning
        with mlflow.start_run(run_name="prediction_log"):
            mlflow.log_params(incoming_features)
            mlflow.log_metric("prediction", float(prediction))

        # Return prediction with version
        return {
            "prediction": float(prediction),
            "model_name": MODEL_NAME,
            "model_stage": MODEL_STAGE,
            "model_version": model_version
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE
    }