from pathlib import Path
import json
from datetime import datetime
from src.automl.train import train_from_config
from src.jobs.job_store import job_store, JobStatus
from src.utils.logger import logger


async def run_training_job(
    job_id: str,
    dataset_id: str,
    target_column: str,
    user_id: int
):
    """
    Execute training in background
    Updates job status as it progresses
    """
    
    logger.info(f"[{job_id}] Training job started - dataset: {dataset_id}")
    
    try:
        # Mark as running
        job_store.update_job(
            job_id,
            status=JobStatus.RUNNING,
            started_at=datetime.now().isoformat(),
            progress=5,
            current_step="Validating dataset..."
        )
        
        logger.info(f"[{job_id}] Validating dataset...")
        
        # Find dataset file
        upload_dir = Path("/app/data/uploads")
        dataset_files = list(upload_dir.glob(f"{dataset_id}.*"))
        
        if not dataset_files:
            logger.error(f"[{job_id}] Dataset not found: {dataset_id}")
            raise FileNotFoundError(f"Dataset {dataset_id} not found")
        
        dataset_path = str(dataset_files[0])
        logger.info(f"[{job_id}] Dataset found: {dataset_path}")
        
        job_store.update_job(
            job_id,
            progress=10,
            current_step="Creating training configuration..."
        )
        
        # Create temporary config
        config = {
            "dataset_path": dataset_path,
            "target_column": target_column,
            "test_size": 0.2,
            "random_state": 42,
            "automl": {
                "models": [
                    {"name": "RandomForest", "params": {}},
                    {"name": "XGBoost", "params": {}},
                    {"name": "LightGBM", "params": {}},
                    {"name": "ElasticNet", "params": {"alpha": 0.1, "l1_ratio": 0.5}},
                    {"name": "LinearRegression", "params": {}}
                ]
            }
        }
        
        # Save temp config
        config_path = f"/tmp/train_config_{job_id}.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        job_store.update_job(
            job_id,
            progress=20,
            current_step="Starting AutoML training..."
        )
        
        logger.info(f"[{job_id}] Starting AutoML training...")
        
        # Run training
        result = train_from_config(config_path ,user_id)
        
        logger.info(f"[{job_id}] Training completed - Best model: {result['best_model']}, R²: {result['best_score']:.4f}")
        
        job_store.update_job(
            job_id,
            progress=95,
            current_step="Finalizing..."
        )
        
        # Cleanup
        Path(config_path).unlink(missing_ok=True)
        
        # Mark as completed
        job_store.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=datetime.now().isoformat(),
            progress=100,
            current_step="Training completed successfully!",
            result={
                "best_model": result["best_model"],
                "best_score": result["best_score"],
                "model_name": result["model_name"],
                "model_version": result["model_version"],
                "model_stage": result.get("stage", "Production")
            }
        )
        
        logger.info(f"[{job_id}] ✅ Job completed successfully!")
        
        
    except Exception as e:
        logger.error(f"[{job_id}] ❌ Training failed: {str(e)}", exc_info=True)
        
        # Mark as failed
        job_store.update_job(
            job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.now().isoformat(),
            current_step="Training failed",
            error=str(e)
        )