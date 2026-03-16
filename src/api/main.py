from fastapi import BackgroundTasks
from src.jobs.training_jobs import run_training_job
from src.jobs.job_store import job_store
import uuid
import psutil
import platform
from fastapi import FastAPI, HTTPException
from fastapi import Request
from src.utils.logger import logger
import uuid
import time
from pydantic import BaseModel ,Field
import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from src.utils.exceptions import (
    DatasetNotFoundError,
    ModelNotFoundError,
    JobNotFoundError,
    NoProductionModelError
)
from fastapi.responses import JSONResponse
import traceback
from datetime import datetime

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
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Catch-all exception handler with logging
    """
    error_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    # Log the full error
    logger.error(
        f"[{request_id}] ❌ ERROR [{error_id}]: {str(exc)}",
        exc_info=True,
        extra={"request_id": request_id, "error_id": error_id}
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "error_id": error_id,
            "request_id": request_id,
            "detail": str(exc),
            "suggestion": "Contact support with error_id if issue persists"
        }
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming requests with detailed info
    """
    # Generate request ID
    request_id = str(uuid.uuid4())[:8]
    
    # Store in request state
    request.state.request_id = request_id
    
    start_time = time.time()
    
    # Log incoming request
    logger.info(
        f"[{request_id}] ➡️  {request.method} {request.url.path}",
        extra={"request_id": request_id}
    )
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"[{request_id}] ⬅️  {request.method} {request.url.path} - {response.status_code} ({duration:.2f}s)",
            extra={"request_id": request_id}
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"[{request_id}] ❌ {request.method} {request.url.path} - ERROR ({duration:.2f}s): {str(e)}",
            exc_info=True,
            extra={"request_id": request_id}
        )
        raise
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
@app.post("/train")
async def trigger_training(
    dataset_id: str,
    target_column: str,
    background_tasks: BackgroundTasks
):
    """
    Trigger training job with uploaded dataset
    """
    
    logger.info(f"Training request - dataset_id: {dataset_id}, target: {target_column}")
    
    # ... existing validation code ...
    
    # Create job
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job = job_store.create_job(job_id, dataset_id, target_column)
    
    logger.info(f"Created training job: {job_id}")
    
    # Start background task
    background_tasks.add_task(
        run_training_job,
        job_id,
        dataset_id,
        target_column
    )
    
    logger.info(f"Started background training job: {job_id}")
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Training job started. Use /train/status/{job_id} to check progress.",
        "dataset_id": dataset_id,
        "target_column": target_column
    }

# Initialize MLflow client
mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


@app.get("/models/versions")
async def list_model_versions():
    """
    List all registered model versions
    
    Returns version history with stages and metadata
    """
    try:
        versions = mlflow_client.search_model_versions(f'name="{MODEL_NAME}"')
        
        if not versions:
            return {
                "model_name": MODEL_NAME,
                "total_versions": 0,
                "versions": [],
                "message": "No models registered yet. Train a model first."
            }
        
        version_list = []
        for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
            version_list.append({
                "version": int(v.version),
                "stage": v.current_stage,
                "created_at": v.creation_timestamp,
                "updated_at": v.last_updated_timestamp,
                "run_id": v.run_id,
                "description": v.description or ""
            })
        
        return {
            "model_name": MODEL_NAME,
            "total_versions": len(version_list),
            "versions": version_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/promote/{version}")
async def promote_model(
    version: int,
    stage: str = "Production"
):
    """
    Promote a model version to a specific stage
    
    Parameters:
    - version: Model version number
    - stage: Target stage (Production, Staging, or Archived)
    
    Automatically archives current Production model
    """
    valid_stages = ["Production", "Staging", "Archived"]
    
    if not versions:
        raise ModelNotFoundError(version)
    
    try:
        # Check if version exists
        # Check if version exists
        all_versions = mlflow_client.search_model_versions(f'name="{MODEL_NAME}"')
        versions = [v for v in all_versions if int(v.version) == version]
        if not versions:
            raise HTTPException(
                status_code=404,
                detail=f"Model version {version} not found"
            )
        
        # If promoting to Production, archive current Production
        if stage == "Production":
            current_prod = mlflow_client.get_latest_versions(
                MODEL_NAME,
                stages=["Production"]
            )
            
            for model in current_prod:
                if int(model.version) != version:
                    mlflow_client.transition_model_version_stage(
                        name=MODEL_NAME,
                        version=model.version,
                        stage="Archived"
                    )
        
        # Promote the new version
        mlflow_client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage=stage
        )
        
        return {
            "message": f"Model version {version} promoted to {stage}",
            "model_name": MODEL_NAME,
            "version": version,
            "stage": stage,
            "note": "Restart API to load the new Production model" if stage == "Production" else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/rollback")
async def rollback_model():
    """
    Rollback to the previous Production model
    
    Finds the most recently archived model and promotes it
    """
    try:
        # Get all versions
        all_versions = mlflow_client.search_model_versions(f'name="{MODEL_NAME}"')
        
        if not all_versions:
            raise HTTPException(
                status_code=404,
                detail="No models found to rollback to"
            )
        
        # Find archived versions (previously in Production)
        archived = [v for v in all_versions if v.current_stage == "Archived"]
        
        if not archived:
            raise HTTPException(
                status_code=404,
                detail="No archived models to rollback to"
            )
        
        # Get most recently updated archived model
        latest_archived = max(archived, key=lambda v: v.last_updated_timestamp)
        
        # Archive current Production
        current_prod = mlflow_client.get_latest_versions(
            MODEL_NAME,
            stages=["Production"]
        )
        
        for model in current_prod:
            mlflow_client.transition_model_version_stage(
                name=MODEL_NAME,
                version=model.version,
                stage="Archived"
            )
        
        # Promote archived to Production
        mlflow_client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_archived.version,
            stage="Production"
        )
        
        return {
            "message": "Rollback successful",
            "model_name": MODEL_NAME,
            "version": int(latest_archived.version),
            "stage": "Production",
            "note": "Restart API to load the rolled-back model"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/compare")
async def compare_models(version1: int, version2: int):
    """
    Compare two model versions
    
    Shows metrics and parameters side-by-side
    """
    try:
        def get_model_details(version: int):
            # Get model version info
             # Get model version info
            all_versions = mlflow_client.search_model_versions(f'name="{MODEL_NAME}"')
            versions = [v for v in all_versions if int(v.version) == version]
            if not versions:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model version {version} not found"
                )
            
            model_version = versions[0]
            
            # Get run details
            run = mlflow_client.get_run(model_version.run_id)
            
            return {
                "version": version,
                "stage": model_version.current_stage,
                "created_at": model_version.creation_timestamp,
                "run_id": model_version.run_id,
                "metrics": run.data.metrics,
                "params": run.data.params
            }
        
        model1 = get_model_details(version1)
        model2 = get_model_details(version2)
        
        # Determine winner based on best_r2 or r2 metric
        r2_key = "best_r2" if "best_r2" in model1["metrics"] else "r2"
        
        if r2_key in model1["metrics"] and r2_key in model2["metrics"]:
            winner = version1 if model1["metrics"][r2_key] > model2["metrics"][r2_key] else version2
            r2_diff = abs(model1["metrics"][r2_key] - model2["metrics"][r2_key])
        else:
            winner = None
            r2_diff = None
        
        return {
            "model_name": MODEL_NAME,
            "version1": model1,
            "version2": model2,
            "comparison": {
                "winner": winner,
                "metric_used": r2_key,
                "r2_difference": round(r2_diff, 4) if r2_diff else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/train/status/{job_id}")
async def get_training_status(job_id: str):
    """
    Get training job status
    
    Returns current status, progress, and results
    """
    job = job_store.get_job(job_id)
    
    if not job:
        raise JobNotFoundError(job_id)
    
    return job


@app.get("/train/jobs")
async def list_training_jobs():
    """
    List all training jobs
    """
    jobs = job_store.list_jobs()
    return {
        "total": len(jobs),
        "jobs": jobs
    }
    
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
        raise NoProductionModelError()
    
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
async def health_check():
    """
    Comprehensive health check with system metrics
    """
    try:
        # Check if model is loaded
        model_status = "loaded" if model is not None else "not_loaded"
        
        # System metrics
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Count training jobs
        all_jobs = job_store.list_jobs()
        job_stats = {
            "total": len(all_jobs),
            "pending": len([j for j in all_jobs if j["status"] == "pending"]),
            "running": len([j for j in all_jobs if j["status"] == "running"]),
            "completed": len([j for j in all_jobs if j["status"] == "completed"]),
            "failed": len([j for j in all_jobs if j["status"] == "failed"])
        }
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model": {
                "status": model_status,
                "version": model_version if model_version else None,
                "name": MODEL_NAME
            },
            "system": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory": {
                    "total_mb": round(memory.total / 1024 / 1024, 2),
                    "used_mb": round(memory.used / 1024 / 1024, 2),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
                    "used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
                    "percent": disk.percent
                }
            },
            "training_jobs": job_stats,
            "mlflow": {
                "tracking_uri": MLFLOW_TRACKING_URI
            }
        }
        
        logger.info("Health check performed")
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }