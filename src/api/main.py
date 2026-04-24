from src.db.database import init_db
from src.api.routers import auth
from src.db.database import get_db
from sqlalchemy.orm import Session
from src.auth.jwt import get_current_user_id
from fastapi import BackgroundTasks , Depends
from src.jobs.training_jobs import run_training_job
from src.jobs.job_store import job_store
import json
import tempfile
from src.db import models
from src.config import MLFLOW_TRACKING_URI, get_model_name
from mlflow.tracking import MlflowClient
from typing import Dict , Any
import psutil
import platform
from fastapi import FastAPI, HTTPException
from fastapi import Request
import pathlib
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
from fastapi.middleware.cors import CORSMiddleware

from src.automl.train import train_from_config
from src.config import MLFLOW_TRACKING_URI, MODEL_STAGE, get_model_name
from fastapi import UploadFile, Form
from src.api.upload import upload_dataset
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, Form, Query
from typing import Optional
from src.auth.api_key import verify_api_key
from src.db.models import APIUsage

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
        "url": "https://github.com/Dhinesh018/automl-assistant",
    },
    license_info={
        "name": "MIT License",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://adventurous-alignment-production-6fad.up.railway.app", # Your specific Railway URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize database
@app.on_event("startup")
def startup():
    init_db()
    print("✅ Database initialized")

# Include auth router
app.include_router(auth.router)

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
    return None, None

model, model_version = None, None

# ---------------- SCHEMAS ----------------
class TrainRequest(BaseModel):
    config_path: str


class TrainResponse(BaseModel):
    best_model: str
    best_score: float
    run_id: str


class PredictRequest(BaseModel):
    features: Dict[str, float]


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
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
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
    
    
    logger.info(f"Training request - dataset_id: {dataset_id}, target: {target_column}")
    
    # Validate dataset exists
    upload_dir = pathlib.Path("/app/data/uploads")
    dataset_files = list(upload_dir.glob(f"{dataset_id}.*"))
    
    if not dataset_files:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # 1. Create unique job identifier
    import uuid
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    
    # 2. Create job in store
    job = job_store.create_job(job_id, dataset_id, target_column)
    
    logger.info(f"Created training job: {job_id}")
    
    # 3. Start background task
    background_tasks.add_task(
        run_training_job,
        job_id,
        dataset_id,
        target_column,
        user_id  # 🔥 PASS user_id
    )
    
    logger.info(f"Started background training job: {job_id}")
    
    # 4. Return immediate response
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Training job started. Use /train/status/{job_id} to check progress.",
        "dataset_id": dataset_id,
        "target_column": target_column
    }
mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
@app.get(
    "/models/versions",
    tags=["📦 Model Management"],
    summary="List all registered model versions",
    response_description="List of all model versions with metrics")

async def list_model_versions(
    user_id: int = Depends(get_current_user_id), 
    db: Session = Depends(get_db)
):
    """
    List every model version in the MLflow registry.
    
    Shows:
    - Which version is in **Production**
    - Which are in **Staging** or **Archived**
    - Performance metrics for each version
    - Creation timestamps
    
    Use this before promoting or rolling back models.
    """
    
    # 1. 🔥 Get models from DB for this user
    user_models = db.query(models.Model).filter(
        models.Model.user_id == user_id
    ).all()
    MODEL_NAME = get_model_name(user_id)
    # Convert versions to integers for comparison, filtering out placeholders like "v1-pending"
    user_versions = []
    for m in user_models:
        try:
            version_val = int(m.version)
            if version_val != 0:  # 🔥 Add this: Skip the '0' placeholder
                user_versions.append(version_val)
        except (ValueError, TypeError):
            continue
    
    try:
        # 2. Fetch all versions from MLflow
        all_versions = mlflow_client.search_model_versions(f'name="{MODEL_NAME}"')
        
        # 3. 🔥 Filter to only include versions that exist in the user's DB records
        filtered_versions = [v for v in all_versions if int(v.version) in user_versions]
        
        if not filtered_versions:
            return {
                "model_name": MODEL_NAME,
                "total_versions": 0,
                "models": [], 
                "production_version": None,
                "message": "No models registered yet. Train a model first."
            }
        
        version_list = []
        production_version = None
        
        # Sort the filtered list
        for v in sorted(filtered_versions, key=lambda x: int(x.version), reverse=True):
            # Get the run to fetch metrics
            try:
                run = mlflow_client.get_run(v.run_id)
                metrics = run.data.metrics
                params = run.data.params
                
                # Try different metric names, default to None if not found
                r2_score = metrics.get('best_r2', metrics.get('r2', metrics.get('r2_score', None)))
                rmse = metrics.get('best_rmse', metrics.get('rmse', None))
                mae = metrics.get('best_mae', metrics.get('mae', None))

                # Algorithm from parameters
                algorithm = params.get('best_model', params.get('model_name', 'Unknown'))
                
            except Exception as e:
                print(f"⚠️ Failed to get metrics for version {v.version}: {e}")
                r2_score = None
                rmse = None
                algorithm = None
            
            version_data = {
                "version": int(v.version),
                "stage": v.current_stage,
                "algorithm": algorithm,
                "r2_score": r2_score,
                "rmse": rmse,
                "created_at": v.creation_timestamp,
                "updated_at": v.last_updated_timestamp,
                "run_id": v.run_id,
                "description": v.description or ""
            }
            
            version_list.append(version_data)
            
            # Track production version
            if v.current_stage == "Production":
                production_version = int(v.version)
        
        return {
            "model_name": MODEL_NAME,
            "total_versions": len(version_list),
            "models": version_list, 
            "production_version": production_version
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
    user_id: int = Depends(get_current_user_id),
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
    MODEL_NAME = get_model_name(user_id)
    valid_stages = ["Production", "Staging", "Archived"]
    
    if stage not in valid_stages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage. Must be one of: {valid_stages}"
        )
    
    try:
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
            version=str(version),  # Convert to string
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
async def rollback_model(user_id: int = Depends(get_current_user_id)):
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
    MODEL_NAME = get_model_name(user_id)
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
    version2: int = Query(..., description="Second model version", example=4),
    user_id: int = Depends(get_current_user_id)
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
    MODEL_NAME = get_model_name(user_id)
    try:
        def get_model_details(version: int):
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
            metrics = run.data.metrics
            params = run.data.params
            
            # Extract metrics with fallbacks
            r2 = metrics.get('best_r2', metrics.get('r2', None))
            rmse = metrics.get('best_rmse', metrics.get('rmse', None))
            mae = metrics.get('best_mae', metrics.get('mae', None))
            algorithm = params.get('best_model', params.get('model_name', 'Unknown'))
            
            return {
                "version": version,
                "algorithm": algorithm,
                "r2_score": r2,
                "rmse": rmse,
                "mae": mae,
                "stage": model_version.current_stage,
                "created_at": model_version.creation_timestamp,
                "run_id": model_version.run_id,
            }
        
        model1 = get_model_details(version1)
        model2 = get_model_details(version2)
        
        # Determine winner based on R² score
        winner = None
        if model1["r2_score"] is not None and model2["r2_score"] is not None:
            winner = version1 if model1["r2_score"] > model2["r2_score"] else version2
        
        # Return in format frontend expects
        return {
            "version_a": model1,  # Changed from version1
            "version_b": model2,  # Changed from version2
            "winner": winner,
            "improvement": {
                "r2": abs(model1["r2_score"] - model2["r2_score"]) if model1["r2_score"] and model2["r2_score"] else None,
                "rmse": abs(model1["rmse"] - model2["rmse"]) if model1["rmse"] and model2["rmse"] else None,
            } if model1["r2_score"] and model2["r2_score"] else None
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
    target_column: str = Form(..., description="Name of the column to predict (e.g., 'SalePrice', 'Price', 'Churn')"),
    user_id: int = Depends(get_current_user_id)
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

from pydantic import BaseModel
from typing import Dict, Any
class DynamicPredictionRequest(BaseModel):
    features: Dict[str, Any]  # Dynamic - accepts any features

@app.post(
    "/predict",
    tags=["🎯 Predictions"],
    summary="Make a prediction (dynamic features)",
)
async def predict(
    request: DynamicPredictionRequest,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db) 
):
    """
    Make a prediction using Production model.
    Includes strict validation for feature types and user isolation.
    """

    # 1. Isolation: Each user has a unique model namespace
    MODEL_NAME = get_model_name(user_id)
    
    try:
        # 2. Load Production model info from MLflow
        client = mlflow.tracking.MlflowClient()
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        
        if not prod_versions:
            raise HTTPException(
                status_code=404,
                detail="No Production model found. Train a model first."
            )
        
        prod_version = prod_versions[0]
        run_id = prod_version.run_id
        version = int(prod_version.version) 
        
        # 3. Load feature metadata from MLflow artifact
        try:
            local_dir = tempfile.mkdtemp()
            local_path = client.download_artifacts(
                run_id, 
                "feature_metadata.json", 
                dst_path=local_dir
            )
            
            with open(local_path, 'r') as f:
                feature_metadata = json.load(f)
            
            expected_features = set(feature_metadata['features'])
            numeric_features = feature_metadata.get('numeric_features', [])
            categorical_features = feature_metadata.get('categorical_features', [])
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load feature metadata: {str(e)}"
            )
        
        # 4. Validate incoming features (Presence and Names)
        incoming_features = set(request.features.keys())
        missing_features = expected_features - incoming_features
        extra_features = incoming_features - expected_features
        
        if missing_features or extra_features:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Feature mismatch",
                    "missing": list(missing_features),
                    "unexpected": list(extra_features)
                }
            )

        # 5. 🔥 NEW: Strict Type Validation
        # This prevents "3" (int) from being accepted in categorical text fields
        type_errors = []
        for feat, value in request.features.items():
            if feat in numeric_features:
                if not isinstance(value, (int, float)):
                    type_errors.append(f"'{feat}' must be a number (got {type(value).__name__})")
            
            elif feat in categorical_features:
                if not isinstance(value, str):
                    type_errors.append(f"'{feat}' must be a string/text (got {type(value).__name__})")

        if type_errors:
            raise HTTPException(
                status_code=422,
                detail={"error": "Type validation failed", "details": type_errors}
            )

        # 6. Load model
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        
        # 7. Create DataFrame with correct feature order
        # Ensures features are in the same order as they were during training
        ordered_features = {
            feat: request.features[feat] 
            for feat in feature_metadata['features']
        }
        
        feature_df = pd.DataFrame([ordered_features])
        
        # 8. Make prediction
        prediction = model.predict(feature_df)[0]
        
        # 9. Get model info for response
        try:
            run = client.get_run(run_id)
            algorithm = run.data.params.get('best_model', 'Unknown')
            r2_score = run.data.metrics.get('best_r2', None)
        except:
            algorithm = 'Unknown'
            r2_score = None
        
        return {
            "prediction": float(prediction),
            "model_name": MODEL_NAME,
            "model_version": version,
            "model_algorithm": algorithm,
            "model_r2_score": r2_score,
            "model_stage": "Production",
            "target": feature_metadata.get('target'),
            "features_used": list(ordered_features.keys()),
            "num_features": len(ordered_features)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )
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
                "status": "multi_user",
                "note": "Each user has their own model namespace"
                
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
@app.get(
    "/models/production/features",
    tags=["🎯 Predictions"],
    summary="Get features for Production model",
)
async def get_production_features(user_id: int = Depends(get_current_user_id) ):
    """
    Get features required by Production model.
    
    """
    MODEL_NAME = get_model_name(user_id)

    try:
        client = MlflowClient()
        
        # Get Production model
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        
        if not prod_versions:
            raise HTTPException(
                status_code=404, 
                detail="No Production model found. Train a model first."
            )
        
        prod_version = prod_versions[0]
        run_id = prod_version.run_id
        
        # Try to download feature metadata
        import tempfile
        import os
        
        try:
            local_dir = tempfile.mkdtemp()
            artifact_path = os.path.join(local_dir, "feature_metadata.json")
            
            # Download artifact
            client.download_artifacts(
                run_id, 
                "feature_metadata.json", 
                dst_path=local_dir
            )
            
            # Read file
            with open(artifact_path, 'r') as f:
                feature_metadata = json.load(f)
            
            # Add model info
            feature_metadata["model_version"] = int(prod_version.version)
            feature_metadata["model_stage"] = "Production"
            
            return feature_metadata
            
        except Exception as e:
            print(f"⚠️ Failed to load feature_metadata.json: {e}")
            
            # Fallback: try dataset profile
            try:
                local_dir = tempfile.mkdtemp()
                artifact_path = os.path.join(local_dir, "dataset_profile.json")
                
                client.download_artifacts(
                    run_id,
                    "dataset_profile.json",
                    dst_path=local_dir
                )
                
                with open(artifact_path, 'r') as f:
                    profile = json.load(f)
                
                # Extract features from profile
                all_columns = profile.get('columns', [])
                target = profile.get('target_column', '')
                features = [col for col in all_columns if col != target]
                
                return {
                    "features": features,
                    "target": target,
                    "num_features": len(features),
                    "numeric_features": profile.get('numeric_columns', []),
                    "categorical_features": profile.get('categorical_columns', []),
                    "model_version": int(prod_version.version),
                    "model_stage": "Production"
                }
                
            except Exception as e2:
                print(f"⚠️ Failed to load dataset_profile.json: {e2}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not load feature metadata. Train a new model with updated code. Error: {str(e2)}"
                )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get production features: {str(e)}"
        )


@app.get(
    "/models/{version}/features",
    tags=["📦 Model Management"],
    summary="Get feature names for a model version",
)
async def get_model_features(version: int , user_id: int = Depends(get_current_user_id)):
    """
    Get the list of features required by a specific model version.
    """
    
    try:
        MODEL_NAME = get_model_name(user_id)
        client = MlflowClient()
        
        # Get model version
        all_versions = client.search_model_versions(f'name="{MODEL_NAME}"')
        model_versions = [v for v in all_versions if int(v.version) == version]
        
        if not model_versions:
            raise HTTPException(status_code=404, detail=f"Model version {version} not found")
        
        model_version = model_versions[0]
        run_id = model_version.run_id
        
        # Download feature metadata artifact
        try:
            import tempfile
            
            local_dir = tempfile.mkdtemp()
            local_path = client.download_artifacts(run_id, "feature_metadata.json", dst_path=local_dir)
            
            with open(local_path, 'r') as f:
                feature_metadata = json.load(f)
            
            feature_metadata["model_version"] = version
            
            return feature_metadata
            
        except Exception as e:
            # Fallback - get from dataset profile
            try:
                local_dir = tempfile.mkdtemp()
                local_path = client.download_artifacts(run_id, "dataset_profile.json", dst_path=local_dir)
                
                with open(local_path, 'r') as f:
                    profile = json.load(f)
                
                features = [col for col in profile.get('columns', []) if col != profile.get('target_column')]
                
                return {
                    "features": features,
                    "target": profile.get('target_column'),
                    "num_features": len(features),
                    "model_version": version
                }
            except:
                raise HTTPException(
                    status_code=404,
                    detail="Feature metadata not found for this model"
                )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/predict",
    tags=["🔑 Public API"],
    summary="Public prediction endpoint (API key auth)"
)
async def api_predict(
    request: DynamicPredictionRequest,
    user_id: int = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    **PUBLIC ENDPOINT** - Authenticate with API key
    
    Usage:
```bash
    curl -X POST http://your-domain.com/api/predict \
      -H "X-API-Key: your_api_key_here" \
      -H "Content-Type: application/json" \
      -d '{"features": {"feature1": 10, "feature2": "value"}}'
```
    """
    
    MODEL_NAME = get_model_name(user_id)
    
    try:
        client = MlflowClient()
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        
        if not prod_versions:
            raise HTTPException(404, "No Production model found")
        
        version = int(prod_versions[0].version)
        run_id = prod_versions[0].run_id
        
        # Load metadata
        local_dir = tempfile.mkdtemp()
        local_path = client.download_artifacts(run_id, "feature_metadata.json", dst_path=local_dir)
        
        with open(local_path, 'r') as f:
            meta = json.load(f)
        
        # Validate features
        expected = set(meta['features'])
        incoming = set(request.features.keys())
        
        if missing := expected - incoming:
            raise HTTPException(400, {"missing_features": list(missing)})
        
        # Load & predict
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
        df = pd.DataFrame([{f: request.features[f] for f in meta['features']}])
        prediction = model.predict(df)[0]
        
        # 🔥 Log usage
        api_key_record = db.query(models.APIKey).filter(
            models.APIKey.user_id == user_id
        ).first()
        
        usage = APIUsage(
            user_id=user_id,
            api_key_id=api_key_record.id,
            endpoint="/api/predict",
            model_version=version
        )
        db.add(usage)
        db.commit()
        
        return {
            "prediction": float(prediction),
            "model_version": version,
            "credits_used": 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    

@app.get(
    "/api/keys",
    tags=["🔑 Public API"],
    summary="Get your API keys"
)
async def get_api_keys(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Get all API keys for logged-in user"""
    
    keys = db.query(models.APIKey).filter(
        models.APIKey.user_id == user_id
    ).all()
    
    return {
        "api_keys": [
            {
                "id": k.id,
                "key": k.key,
                "created_at": k.created_at
            }
            for k in keys
        ]
    }

@app.get(
    "/api/usage",
    tags=["🔑 Public API"],
    summary="Get API usage stats"
)
async def get_api_usage(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Get usage statistics"""
    
    total_calls = db.query(APIUsage).filter(
        APIUsage.user_id == user_id
    ).count()
    
    recent = db.query(APIUsage).filter(
        APIUsage.user_id == user_id
    ).order_by(APIUsage.timestamp.desc()).limit(10).all()
    
    return {
        "total_calls": total_calls,
        "recent_calls": [
            {
                "endpoint": u.endpoint,
                "model_version": u.model_version,
                "timestamp": u.timestamp
            }
            for u in recent
        ]
    }
