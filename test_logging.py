import unittest
import os
import shutil
import logging
import time
import asyncio
from unittest.mock import MagicMock, patch
from logger_config import setup_logging
from background_logger import start_periodic_upload

class TestLoggingAndUpload(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.log_dir = "test_logs"
        self.log_file = "app.jsonl"
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir)
        
        # Setup logger with small maxBytes for easy rotation
        self.logger = setup_logging(log_dir=self.log_dir, log_file=self.log_file, max_bytes=100, backup_count=5)

    def tearDown(self):
        # Close handlers to release file locks
        for handler in self.logger.handlers:
            handler.close()
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

    async def test_rotation_and_upload(self):
        # 1. Write enough logs to trigger rotation
        # Each log is roughly 80-100 bytes with JSON formatting
        for i in range(5):
            self.logger.info(f"Log message {i} " * 5)
            
        # Check if rotation happened (app.jsonl and app.jsonl.1 should exist)
        files = os.listdir(self.log_dir)
        print(f"Files after logging: {files}")
        self.assertTrue(any(f.endswith('.1') for f in files), "Log rotation should have occurred")

        # 2. Mock HfApi and env vars
        with patch("background_logger.HfApi") as MockApi, \
             patch.dict(os.environ, {"HF_TOKEN": "fake_token", "HF_LOG_DATASET": "fake/dataset"}):
            
            mock_api_instance = MockApi.return_value
            mock_api_instance.upload_file = MagicMock()
            
            # Run background upload for a short time
            # We'll run it for 2 seconds with 1 second interval
            task = asyncio.create_task(start_periodic_upload(log_dir=self.log_dir, log_file=self.log_file, interval=1))
            
            await asyncio.sleep(2.5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Verify upload was called
            self.assertTrue(mock_api_instance.upload_file.called, "upload_file should have been called")
            
            # Verify file was deleted (renamed and uploaded files are deleted)
            # app.jsonl should still exist (active), but rotated files should be gone
            files_after = os.listdir(self.log_dir)
            print(f"Files after upload: {files_after}")
            # We expect only app.jsonl (active) to remain, or maybe a new one if rotation happened again
            # The rotated file (.1) should be gone
            self.assertFalse(any(f.endswith('.1') for f in files_after), "Rotated file should have been deleted after upload")

if __name__ == "__main__":
    unittest.main()
