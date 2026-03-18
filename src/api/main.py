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
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, Form, Query
from typing import Optional

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------------- APP ----------------
app = FastAPI(
    title="🤖 LLM-Augmented AutoML System",
    description="""
## Intelligent AutoML with LLM-Powered Model Selection

An advanced AutoML pipeline that uses a **Large Language Model** to analyze your dataset 
and intelligently select the most promising ML algorithms — instead of blindly training all models.

### 🎯 Why This is Different

| Traditional AutoML | This System |
|-------------------|-------------|
| Trains ALL models every time | LLM selects 1-3 best candidates |
| Wastes compute resources | ~70% faster training |
| No explainability | LLM reasoning logged to MLflow |

### ⚡ How It Works
```
Upload Dataset → LLM Analyzes & Selects Models → Train Subset 
    → Register Best Model → Auto-Promote to Production → Make Predictions
```

### 🚀 Key Features

- **LLM Model Selection** - Groq API (Llama 3.3 70B) reads your dataset profile and decides which algorithms will perform best
- **Full MLOps Pipeline** - MLflow experiment tracking, model registry, versioning, promotion, and rollback
- **Background Training** - Non-blocking async training with real-time status updates
- **Production Safety** - Auto-promotion, one-click rollback, version comparison

### 📊 Available Models

`RandomForest` • `XGBoost` • `LightGBM` • `ElasticNet` • `LinearRegression`

---

---

**Tech Stack:** FastAPI • MLflow • Groq API • Docker • Python 3.11
    """,
    version="1.0.0",
    contact={
        "name": "AutoML Assistant Project",
        "url": "https://github.com/yourusername/automl-assistant",
    },
    license_info={
        "name": "MIT License",
    },
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
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API docs"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@app.post(
    "/train",
    tags=["🚀 Training & Jobs"],
    summary="Start an AutoML training job",
    status_code=202,
    response_description="Training job started successfully",
)
async def trigger_training(
    dataset_id: str = Query(..., description="Dataset ID from /datasets/upload", example="dataset_20260314_123456_abc123"),
    target_column: str = Query(..., description="Target column to predict", example="SalePrice"),
    background_tasks: BackgroundTasks = None
):
    """
    Trigger an AutoML training run for an uploaded dataset.
    
    **What happens:**
    1. LLM analyzes the dataset profile
    2. LLM selects 1-3 most promising models
    3. Selected models are trained and evaluated
    4. Best model is registered in MLflow
    5. Winner is **auto-promoted to Production**
    
    Training runs in the **background**. This endpoint returns immediately with a `job_id`.
    
    **Poll** `GET /train/status/{job_id}` to track progress.
    
    **Typical training time:** 30 seconds - 5 minutes
    """
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


@app.get(
    "/models/versions",
    tags=["📦 Model Management"],
    summary="List all registered model versions",
    response_description="List of all model versions with metrics",
)
async def list_model_versions():
    """
    List every model version in the MLflow registry.
    
    Shows:
    - Which version is in **Production**
    - Which are in **Staging** or **Archived**
    - Performance metrics for each version
    - Creation timestamps
    
    Use this before promoting or rolling back models.
    """
    # ... your existing implementation ...
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


@app.post(
    "/models/promote/{version}",
    tags=["📦 Model Management"],
    summary="Promote a model version to a stage",
    response_description="Model promoted successfully",
)
async def promote_model(
    version: int,
    stage: str = Query("Production", description="Target stage: Production, Staging, or Archived")
):
    """
    Manually promote a model version to **Production**, **Staging**, or **Archived**.
    
    When promoting to Production:
    - Previous Production model → Archived
    - New model → Production
    - There is always exactly **one** Production model
    
    **Use cases:**
    - Validating a Staging model before going live
    - Restoring a known-good archived model
    - Demoting a broken Production model
    
    ⚠️ **Remember to restart the API** after promoting to Production!
    """
    # ... your existing implementation ...
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


@app.post(
    "/models/rollback",
    tags=["📦 Model Management"],
    summary="Rollback to previous Production model",
    response_description="Rollback completed successfully",
)
async def rollback_model():
    """
    Instantly roll back to the **most recent Archived** model.
    
    This is your **emergency brake**. If the current Production model starts 
    making bad predictions, call this endpoint to immediately restore the 
    previous version.
    
    **What happens:**
    1. Current Production model → Archived
    2. Most recent Archived model → Production
    
    ⚠️ **Restart the API** after rollback to load the restored model.
    """
    # ... your existing implementation ...
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


@app.get(
    "/models/compare",
    tags=["📦 Model Management"],
    summary="Compare two model versions",
    response_description="Side-by-side comparison of two versions",
)
async def compare_models(
    version1: int = Query(..., description="First model version", example=5),
    version2: int = Query(..., description="Second model version", example=4)
):
    """
    Compare two model versions by their training metrics.
    
    Returns:
    - Side-by-side R², RMSE, MAE
    - Which version performed better
    - Metric differences
    
    Use this before promoting a new model to confirm it actually improves 
    on the current Production model.
    """
    # ... your existing implementation ...
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

@app.get(
    "/train/status/{job_id}",
    tags=["🚀 Training & Jobs"],
    summary="Check training job progress",
    response_description="Current job status and progress",
)
async def get_training_status(job_id: str):
    """
    Poll this endpoint to track a background training job.
    
    **Status values:**
    - `pending` - Job is queued
    - `running` - Actively training models
    - `completed` - Training finished, best model in Production
    - `failed` - Training encountered an error
    
    **Recommended polling interval:** Every 3-5 seconds
    
    **Example response (running):**
```json
    {
      "job_id": "job_abc123",
      "status": "running",
      "progress": 60,
      "current_step": "Training XGBoost (model 2/3)"
    }
```
    
    **Example response (completed):**
```json
    {
      "job_id": "job_abc123",
      "status": "completed",
      "progress": 100,
      "result": {
        "best_model": "XGBoost",
        "best_score": 0.92,
        "model_version": 5
      }
    }
```
    """
    # ... your existing implementation ...
    job = job_store.get_job(job_id)
    
    if not job:
        raise JobNotFoundError(job_id)
    
    return job


@app.get(
    "/train/jobs",
    tags=["🚀 Training & Jobs"],
    summary="List all training jobs",
    response_description="List of all training jobs",
)
async def list_training_jobs():
    """
    Retrieve a list of all training jobs, newest first.
    
    Useful for:
    - Building a training history dashboard
    - Debugging past runs
    - Monitoring training activity
    
    Returns all jobs with their current status, progress, and results.
    """
    # ... your existing implementation ...
    jobs = job_store.list_jobs()
    return {
        "total": len(jobs),
        "jobs": jobs
    }
    
@app.post(
    "/datasets/upload",
    tags=["📁 Dataset Management"],
    summary="Upload a training dataset",
    response_description="Dataset uploaded and profiled successfully",
)
async def upload_dataset_endpoint(
    file: UploadFile,
    target_column: str = Form(..., description="Name of the column to predict (e.g., 'SalePrice', 'Price', 'Churn')")
):
    """
    Upload a CSV or Excel file to use for AutoML training.
    
    **The system will:**
    1. Validate the file format (CSV, XLSX, XLS)
    2. Check that the target column exists
    3. Profile the dataset (types, distributions, missing values)
    4. Return a `dataset_id` for use in training
    
    **Supported formats:** .csv, .xlsx, .xls  
    **Max file size:** 100 MB  
    **Returns:** `dataset_id` - pass this to POST /train to start training
    """
    return await upload_dataset(file, target_column)

@app.post(
    "/predict",
    tags=["🎯 Predictions"],
    summary="Get prediction from Production model",
    response_description="Prediction generated successfully",
)
def predict(req: PredictRequest):
    """
    Run inference using the currently active **Production** model.
    
    Pass your feature values as a JSON object. The system automatically applies 
    the same preprocessing used during training.
    
    **Example request:**
```json
    {
      "features": {
        "LotArea": 11950,
        "OverallQual": 7,
        "YearBuilt": 2003,
        "GrLivArea": 1710,
        "FullBath": 2,
        "GarageCars": 2
      }
    }
```
    
    **Example response:**
```json
    {
      "prediction": 215000.0,
      "model_name": "llm_automl_tabular_model",
      "model_version": 5,
      "model_stage": "Production"
    }
```
    
    ⚠️ **Note:** If you get a 503 error, train a model first and restart the API.
    """
    # ... your existing implementation ...
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


@app.get(
    "/health",
    tags=["💚 System Health"],
    summary="System health and resource metrics",
    response_description="System status and live metrics",
)
async def health_check():
    """
    Full system health check with live resource monitoring.
    
    **Checks:**
    - API status
    - MLflow connectivity
    - Model registry status
    - CPU usage
    - Memory usage
    - Disk usage
    - Production model details
    
    **Use for:**
    - Docker health checks
    - Uptime monitoring
    - Pre-demo sanity check
    
    Returns **200** if all systems healthy, **503** if any component is down.
    """
    # ... your existing implementation ...
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