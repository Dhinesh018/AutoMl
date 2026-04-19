from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session
from src.db.database import get_db
from src.db import models

async def verify_api_key(
    x_api_key: str = Header(..., description="API Key"),
    db: Session = Depends(get_db)
) -> int:
    """Validate API key and return user_id"""
    
    api_key_record = db.query(models.APIKey).filter(
        models.APIKey.key == x_api_key
    ).first()
    
    if not api_key_record:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return api_key_record.user_id