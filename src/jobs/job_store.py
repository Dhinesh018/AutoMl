from datetime import datetime
from enum import Enum
from typing import Dict, Optional
import threading


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStore:
    """
    In-memory job storage for tracking training jobs
    For production: use Redis or database
    """
    
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()
    
    def create_job(
        self,
        job_id: str,
        dataset_id: str,
        target_column: str
    ) -> dict:
        """Create new training job"""
        with self._lock:
            job = {
                "job_id": job_id,
                "dataset_id": dataset_id,
                "target_column": target_column,
                "status": JobStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "progress": 0,
                "current_step": "Initializing...",
                "result": None,
                "error": None
            }
            self.jobs[job_id] = job
            return job
    
    def update_job(self, job_id: str, **kwargs):
        """Update job fields"""
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(kwargs)
                self.jobs[job_id]["updated_at"] = datetime.now().isoformat()
    
    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job by ID"""
        with self._lock:
            return self.jobs.get(job_id)
    
    def list_jobs(self) -> list:
        """List all jobs"""
        with self._lock:
            return list(self.jobs.values())


# Global instance
job_store = JobStore()