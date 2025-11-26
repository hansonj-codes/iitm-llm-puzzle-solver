import asyncio
import os
import glob
import time
import shutil
import uuid
import logging
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Configure logger for this module to avoid recursion/rotation issues
logger = logging.getLogger(__name__)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

async def start_periodic_upload(log_dir="logs", log_file="app.jsonl", interval=60):
    """
    Periodically checks for rotated log files and uploads them to Hugging Face.
    """
    hf_token = os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_LOG_DATASET")
    
    # Use absolute path to avoid CWD issues
    log_dir = os.path.abspath(log_dir)
    
    if not hf_token or not repo_id:
        logger.warning("HF_TOKEN or HF_LOG_DATASET not set. Background upload disabled.")
        return

    api = HfApi(token=hf_token)
    
    logger.info(f"Starting background log uploader for {repo_id} every {interval} seconds.")

    while True:
        try:
            await asyncio.sleep(interval)
            
            # 1. Force Rotation Check
            # Check if active log file exists and is stale (not modified in > 10 mins) and has content
            active_log_path = os.path.join(log_dir, log_file)
            if os.path.exists(active_log_path):
                last_modified = os.path.getmtime(active_log_path)
                file_size = os.path.getsize(active_log_path)
                
                # If file has content of at least 5KB and hasn't been touched in 15 minutes (900 seconds)
                if file_size > 5*1024 and (time.time() - last_modified) > 900:
                    logger.info("Active log file is stale. Forcing rotation.")
                    # Find the rotating handler and force rollover
                    root_logger = logging.getLogger()
                    for handler in root_logger.handlers:
                        if isinstance(handler, logging.handlers.RotatingFileHandler):
                            if handler.baseFilename == os.path.abspath(active_log_path):
                                handler.doRollover()
                                break

            # 2. Identify Rotated Files
            # RotatingFileHandler appends .1, .2, etc.
            # We look for files that match the pattern but are NOT the active file
            pattern = os.path.join(log_dir, f"{log_file}.*")
            all_files = glob.glob(pattern)
            
            for file_path in all_files:
                # Double check it's not the active file (though glob pattern usually excludes it if it's just .jsonl)
                if os.path.abspath(file_path) == os.path.abspath(active_log_path):
                    continue
                
                # 3. Prepare for Upload
                # Rename to unique filename to prevent overwrite
                timestamp = int(time.time())
                unique_id = uuid.uuid4().hex[:8]
                filename = f"log_{timestamp}_{unique_id}.jsonl"
                
                logger.info(f"Uploading {file_path} as {filename}...")
                
                try:
                    # 4. Upload to HF
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=filename,
                        repo_id=repo_id,
                        repo_type="dataset"
                    )
                    
                    logger.info(f"Successfully uploaded {filename}. Deleting local copy.")
                    
                    # 5. Cleanup
                    os.remove(file_path)
                    
                except Exception as e:
                    logger.error(f"Failed to upload {file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in background upload loop: {e}")
            # Don't crash the loop, just wait for next interval
