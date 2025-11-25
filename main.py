import os
import logging
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from solver import solve_quiz

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Expected secret from environment variables
EXPECTED_SECRET = os.getenv("STUDENT_SECRET", "default_secret")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL", "student@example.com")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: HttpUrl
    # Allow extra fields
    class Config:
        extra = "allow"

@app.post("/run")
async def run_quiz(request: QuizRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to receive quiz tasks.
    """
    logger.info(f"Received request: {request}")

    # Verify secret
    if request.secret != EXPECTED_SECRET:
        logger.warning(f"Invalid secret received: {request.secret}")
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Dispatch background task to solve the quiz
    # We pass the URL string to the solver
    background_tasks.add_task(solve_quiz, str(request.url), request.email, request.secret)

    return {"message": "Quiz processing started", "status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
