from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException

from src.automl.train import train_from_config

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ---------------- APP ----------------
app = FastAPI(
    title="LLM-Augmented AutoML Assistant",
    version="0.1.0"
)

# ---------------- CONSTANTS ----------------
MODEL_NAME = "llm_automl_tabular_model"
MODEL_STAGE = "Production"

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
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        return mlflow.pyfunc.load_model(model_uri)
    except MlflowException:
        raise RuntimeError(
            f"No model found in stage '{MODEL_STAGE}' for {MODEL_NAME}"
        )

model = load_model()

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


# ---------------- ENDPOINTS ----------------
@app.post("/train", response_model=TrainResponse)
def train_model(req: TrainRequest):
    try:
        return train_from_config(req.config_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
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

        return {
            "prediction": float(prediction),
            "model_name": MODEL_NAME,
            "model_stage": MODEL_STAGE
        }
        with mlflow.start_run(run_name="prediction_log"):
            mlflow.log_params(incoming_features)
            mlflow.log_metric("prediction", float(prediction))


    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE
    }
