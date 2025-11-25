# Quiz Solver API Walkthrough

I have successfully implemented the Quiz Solver API using FastAPI, LangChain (LangGraph), and Playwright. The system is designed to receive a quiz URL, solve the task using an AI agent, and recursively handle subsequent tasks.

## Changes Implemented

### Core API
- **`main.py`**: FastAPI application with a `POST /run` endpoint. It verifies the secret and dispatches the `solve_quiz` background task.
- **`solver.py`**: Implements the main loop. It initializes the agent, manages the URL recursion, and handles the flow from visiting the page to submitting the answer.

### AI Agent
- **`agent.py`**: Configured a LangGraph agent using `gpt-4o`.
- **`tools.py`**: Created custom tools for the agent:
    - `read_page_content`: Uses Playwright to render and extract text from the quiz page.
    - `download_file`: Downloads files referenced in the quiz.
    - `submit_answer`: Submits the JSON payload to the verification endpoint.
    - `PythonREPLTool`: Allows the agent to write and execute Python code for data analysis.

### Verification
- **`mock_server.py`**: A local FastAPI server that simulates the quiz environment. It serves a sample question ("Sum of 10 + 20") and validates the answer.
- **`verify.py`**: A script that spins up both the main app and the mock server, triggers the flow, and verifies that the agent can solve the quiz.

## Verification Results

I ran the `verify.py` script which performed the following steps:
1. Started the Main API and the Mock Server.
2. Sent a request to `POST /run` with the mock quiz URL.
3. The Agent visited the mock URL, read the question "Calculate the sum of 10 + 20".
4. The Agent calculated the answer (30) and submitted it.
5. The Mock Server responded with `correct: true` and a next URL (`/quiz-2`).
6. The Agent proceeded to the next URL.

**Log Output:**
```
INFO:solver:Processing URL: http://127.0.0.1:8001/quiz-start
Response: 200 - {"message":"Quiz processing started","status":"ok"}
...
INFO:solver:Agent output: http://localhost:8001/quiz-2
INFO:solver:Processing URL: http://localhost:8001/quiz-2
```

## How to Run

1. **Start the API:**
   ```bash
   uvicorn main:app --reload
   ```

2. **Trigger a Quiz:**
   Send a POST request to `http://localhost:8000/run` with the JSON payload:
   ```json
   {
     "email": "your_email",
     "secret": "your_secret",
     "url": "https://example.com/quiz-start"
   }
   ```

3. **Monitor Logs:**
   The application logs will show the agent's progress as it solves the quiz.
