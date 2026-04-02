from pydantic import BaseModel, Field, validator
from typing import Dict, Optional ,Any


class DatasetUploadResponse(BaseModel):
    """Response for dataset upload"""
    dataset_id: str
    filename: str
    num_rows: int = Field(..., gt=0, description="Must have at least 1 row")
    num_columns: int = Field(..., gt=0, description="Must have at least 1 column")
    target_column: str
    missing_values: int = Field(..., ge=0)
    file_size_mb: float
    uploaded_at: str


class TrainRequest(BaseModel):
    """Request for training"""
    dataset_id: str = Field(..., min_length=10, description="Dataset ID from upload")
    target_column: str = Field(..., min_length=1, description="Target column name")


class TrainResponse(BaseModel):
    """Response for training job"""
    job_id: str
    status: str
    message: str
    dataset_id: str
    target_column: str


class PredictRequest(BaseModel):
    """Request for prediction"""
    features: Dict[str, Any]
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        return v


class PredictResponse(BaseModel):
    """Response for prediction"""
    prediction: float
    model_name: str
    model_stage: str
    model_version: Optional[int]