import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

def get_url():
    url = os.getenv("DATABASE_URL")
    
    # 1. Strict Check: If it's actually missing at runtime, kill the process
    if not url:
        print("❌ FATAL: DATABASE_URL is not set!")
        # We only exit if we aren't in a "build" or "test" environment
        if "pytest" not in sys.modules: 
            return "postgresql://placeholder:placeholder@localhost:5432/placeholder"
        return None

    # 2. Fix Railway's 'postgres://' prefix for SQLAlchemy compatibility
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url

DATABASE_URL = get_url()

# Create engine with a pool size suitable for production
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True  # Detects stale connections and reconnects
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Create all tables - strictly Postgres"""
    if "placeholder" in DATABASE_URL:
        raise ConnectionError("❌ Cannot initialize database: DATABASE_URL is missing or invalid.")
    Base.metadata.create_all(bind=engine)
    print("✅ Postgres tables created successfully")