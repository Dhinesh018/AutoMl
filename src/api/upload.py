from fastapi import UploadFile, HTTPException, Form
from pathlib import Path
import pandas as pd
import hashlib
from datetime import datetime
from typing import Optional

UPLOAD_DIR = Path("/app/data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


async def upload_dataset(
    file: UploadFile,
    target_column: str
) -> dict:
    """
    Upload and validate dataset
    Returns dataset_id and basic stats
    """
    
    # 1. Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"
        )
    
    # 2. Read file contents
    contents = await file.read()
    
    # 3. Check file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {MAX_FILE_SIZE / 1024 / 1024}MB"
        )
    
    # 4. Generate unique dataset_id
    file_hash = hashlib.md5(contents).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_id = f"dataset_{timestamp}_{file_hash}"
    
    # 5. Save file
    file_path = UPLOAD_DIR / f"{dataset_id}{file_ext}"
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # 6. Load and validate DataFrame
    try:
        if file_ext == ".csv":
            df = pd.read_csv(file_path)
        else:  # Excel
            df = pd.read_excel(file_path)
    except Exception as e:
        file_path.unlink()  # Delete invalid file
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read file: {str(e)}"
        )
    
    # 7. Validate target column exists
    if target_column not in df.columns:
        file_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found. Available: {list(df.columns)}"
        )
    
    # 8. Check for empty dataset
    if len(df) == 0:
        file_path.unlink()
        raise HTTPException(
            status_code=400,
            detail="Dataset is empty!"
        )
    
    # 9. Generate profile
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    profile = {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "file_path": str(file_path),
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "target_column": target_column,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": int(df.isnull().sum().sum()),
        "file_size_mb": round(len(contents) / 1024 / 1024, 2),
        "uploaded_at": datetime.now().isoformat()
    }
    
    return profile