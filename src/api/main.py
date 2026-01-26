from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow

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
