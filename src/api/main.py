from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


from src.automl.train import train_from_config

app = FastAPI(
    title="LLM-Augmented AutoML Assistant",
    version="0.1.0"
)


class TrainRequest(BaseModel):
    config_path: str


class TrainResponse(BaseModel):
    best_model: str
    best_score: float
    run_id: str


@app.post("/train", response_model=TrainResponse)
def train_model(req: TrainRequest):
    try:
        result = train_from_config(req.config_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PredictRequest(BaseModel):
    features: dict


class PredictResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: int

MODEL_NAME = "llm_automl_tabular_model"
@app.post("/predict", response_model=PredictResponse)
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model_uri = f"models:/{MODEL_NAME}/Latest"
        model = mlflow.pyfunc.load_model(model_uri)

        input_df = pd.DataFrame([req.features])
        prediction = model.predict(input_df)[0]

        client = MlflowClient()
        latest_versions = client.get_latest_versions(
            name=MODEL_NAME,
            stages=["None", "Production", "Staging"]
        )

        model_version = latest_versions[0].version

        return {
            "prediction": float(prediction),
            "model_name": MODEL_NAME,
            "model_version": model_version
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
