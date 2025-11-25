import time
import requests
import multiprocessing
import uvicorn
import os
import sys
from main import app as main_app
from mock_server import app as mock_app

def run_mock_server():
    uvicorn.run(mock_app, host="127.0.0.1", port=8001, log_level="error")

def run_main_app():
    uvicorn.run(main_app, host="127.0.0.1", port=8000, log_level="error")

def test_flow():
    print("Starting servers...")
    mock_proc = multiprocessing.Process(target=run_mock_server)
    main_proc = multiprocessing.Process(target=run_main_app)
    
    mock_proc.start()
    main_proc.start()
    
    # Wait for servers to start
    time.sleep(5)
    
    try:
        print("Sending request to main app...")
        payload = {
            "email": "student@example.com",
            "secret": "default_secret",
            "url": "http://127.0.0.1:8001/quiz-start"
        }
        
        response = requests.post("http://127.0.0.1:8000/run", json=payload)
        print(f"Response: {response.status_code} - {response.text}")
        
        if response.status_code == 200:
            print("Request accepted. Waiting for background processing (check logs)...")
            # We can't easily check the background task result here without shared state or logs
            # But we can wait a bit and see if the mock server received the submission
            # For now, we just wait to let the agent run
            time.sleep(20) 
        else:
            print("Failed to start quiz.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Stopping servers...")
        mock_proc.terminate()
        main_proc.terminate()
        mock_proc.join()
        main_proc.join()

if __name__ == "__main__":
    # Ensure we're in the right directory for imports
    sys.path.append(os.getcwd())
    test_flow()
