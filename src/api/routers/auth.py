from fastapi import APIRouter, HTTPException
from src.db import schemas
import secrets
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["🔐 Authentication"])

# Temporary in-memory storage
USERS_DB = {}
TOKENS_DB = {}

@router.post("/signup", response_model=schemas.Token)
def signup(user: schemas.UserCreate):
    """Signup - temporary in-memory storage"""
    try:
        logger.info(f"🔐 Signup: {user.email}")
        
        # Check if user exists
        if user.email in USERS_DB:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create user
        user_id = len(USERS_DB) + 1
        USERS_DB[user.email] = {
            "id": user_id,
            "email": user.email,
            "password": user.password
        }
        
        # Create token
        token = secrets.token_urlsafe(32)
        TOKENS_DB[token] = user_id
        
        logger.info(f"✅ User created: {user.email}")
        
        return {"access_token": token, "token_type": "bearer"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Signup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/login", response_model=schemas.Token)
def login(user: schemas.UserCreate):
    """Login - temporary in-memory storage"""
    try:
        logger.info(f"🔓 Login: {user.email}")
        
        # Find user
        if user.email not in USERS_DB:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        stored_user = USERS_DB[user.email]
        if stored_user["password"] != user.password:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create token
        token = secrets.token_urlsafe(32)
        TOKENS_DB[token] = stored_user["id"]
        
        logger.info(f"✅ Login successful: {user.email}")
        
        return {"access_token": token, "token_type": "bearer"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))