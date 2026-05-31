import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import os


class JSONFormatter(logging.Formatter):
    """
    Custom formatter for JSON logs
    """
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        
        return json.dumps(log_data)


def setup_logger(name: str = "automl_api") -> logging.Logger:
    """
    Setup structured logging
    
    Logs to both console and file
    Uses /tmp on Render (read-only /app) and ./logs locally
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create logs directory - use /tmp on Render, ./logs locally
    if os.path.exists("/app"):  # We're on Render
        log_dir = Path("/tmp/logs")
    else:  # Local development
        log_dir = Path("./logs")
    
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Logs directory: {log_dir}")
    except Exception as e:
        print(f"⚠️ Could not create logs directory: {e}")
        log_dir = None
    
    # Console handler (human-readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # Add console handler
    logger.addHandler(console_handler)
    
    # File handler (JSON format) - only if directory was created
    if log_dir:
        try:
            log_file = log_dir / f"automl_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(JSONFormatter())
            logger.addHandler(file_handler)
            print(f"✅ File logging to: {log_file}")
        except Exception as e:
            print(f"⚠️ Could not setup file logging: {e}")
    else:
        print("⚠️ File logging disabled (no writable logs directory)")
    
    return logger


# Global logger instance
logger = setup_logger()