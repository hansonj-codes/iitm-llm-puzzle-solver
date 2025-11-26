import os
import logging
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_400_BAD_REQUEST
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
from solver import solve_quiz

from contextlib import asynccontextmanager
from logger_config import setup_logging
from background_logger import start_periodic_upload
import asyncio

# Load .env only if running locally
if os.getenv("SPACE_ID") is None:  # SPACE_ID is set by HF automatically
    from dotenv import load_dotenv
    load_dotenv()

# Configure logging
logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background log uploader
    task = asyncio.create_task(start_periodic_upload())
    yield
    # Cancel task on shutdown (optional, but good practice)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def invalid_json_handler(request, exc):
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": "Invalid JSON payload"},
    )

# Allow CORS for all origins (modify as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Expected secret from environment variables
EXPECTED_SECRET = os.getenv("STUDENT_SECRET")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL")

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

@app.get("/")
def health_check2():
    return {"status": "healthy"}
