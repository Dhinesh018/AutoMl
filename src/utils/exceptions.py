from fastapi import HTTPException, status


class DatasetNotFoundError(HTTPException):
    """Raised when dataset doesn't exist"""
    def __init__(self, dataset_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "DatasetNotFound",
                "message": f"Dataset '{dataset_id}' not found",
                "suggestion": "Upload a dataset first using POST /datasets/upload"
            }
        )


class ModelNotFoundError(HTTPException):
    """Raised when model version doesn't exist"""
    def __init__(self, version: int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "ModelNotFound",
                "message": f"Model version {version} not found",
                "suggestion": "Use GET /models/versions to see available versions"
            }
        )


class JobNotFoundError(HTTPException):
    """Raised when training job doesn't exist"""
    def __init__(self, job_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "JobNotFound",
                "message": f"Training job '{job_id}' not found",
                "suggestion": "Use GET /train/jobs to see all jobs"
            }
        )


class NoProductionModelError(HTTPException):
    """Raised when no model in Production stage"""
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "NoProductionModel",
                "message": "No model in Production stage",
                "suggestion": "Train a model using POST /train, then restart the API"
            }
        )


class TrainingFailedError(HTTPException):
    """Raised when training job fails"""
    def __init__(self, reason: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "TrainingFailed",
                "message": f"Training failed: {reason}",
                "suggestion": "Check dataset quality and try again"
            }
        )


class InvalidFileTypeError(HTTPException):
    """Raised when file type is not supported"""
    def __init__(self, filename: str, allowed_types: list):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "InvalidFileType",
                "message": f"File '{filename}' has unsupported type",
                "allowed_types": allowed_types,
                "suggestion": f"Upload one of: {', '.join(allowed_types)}"
            }
        )


class FileTooLargeError(HTTPException):
    """Raised when file exceeds size limit"""
    def __init__(self, size_mb: float, max_size_mb: int):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "FileTooLarge",
                "message": f"File size {size_mb:.2f}MB exceeds limit",
                "max_size_mb": max_size_mb,
                "suggestion": f"Upload a file smaller than {max_size_mb}MB"
            }
        )


class InvalidTargetColumnError(HTTPException):
    """Raised when target column doesn't exist"""
    def __init__(self, target_column: str, available_columns: list):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "InvalidTargetColumn",
                "message": f"Target column '{target_column}' not found in dataset",
                "available_columns": available_columns,
                "suggestion": "Choose one of the available columns"
            }
        )