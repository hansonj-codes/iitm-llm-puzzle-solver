import logging
import logging.handlers
import os
import json
import datetime
import traceback

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        
        if record.exc_info:
            log_record["exception"] = traceback.format_exception(*record.exc_info)
            
        # Add any extra attributes
        if hasattr(record, "extra"):
            log_record.update(record.extra)
            
        return json.dumps(log_record)

def setup_logging(log_dir="logs", log_file="app.jsonl", max_bytes=500*1024, backup_count=10):
    """
    Sets up logging to a rotating JSONL file.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_path = os.path.join(log_dir, log_file)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates if called multiple times
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # Rotating File Handler
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count
    )
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    # Console Handler (optional, but good for debugging)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger
