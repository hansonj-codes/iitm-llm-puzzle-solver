# Quiz Solver API Implementation Plan

## Goal Description
Build a FastAPI-based API endpoint that receives a quiz task URL, solves the quiz using an AI agent (LangChain + Playwright), and submits the answer. The system must handle recursive quiz tasks (one leading to another) and adhere to strict timing and format constraints.

## User Review Required
> [!IMPORTANT]
> **OpenAI API Key**: The system requires a valid `OPENAI_API_KEY` to function.
> **Student Credentials**: The system requires `STUDENT_EMAIL` and `STUDENT_SECRET` to be configured for submissions.

## Proposed Changes

### Project Structure
#### [NEW] [requirements.txt](file:///c:/Users/hanso/OneDrive/Desktop/New folder/New folder/requirements.txt)
- `fastapi`, `uvicorn`, `playwright`, `langchain`, `langchain-openai`, `pandas`, `requests`, `beautifulsoup4`, `pdfplumber`, `matplotlib`, `python-multipart`, `python-dotenv`.

#### [NEW] [main.py](file:///c:/Users/hanso/OneDrive/Desktop/New folder/New folder/main.py)
- FastAPI app entry point.
- `POST /run` (or `/webhook`) endpoint.
- Validates `secret` header/body.
- Triggers background task `solve_quiz`.

#### [NEW] [solver.py](file:///c:/Users/hanso/OneDrive/Desktop/New folder/New folder/solver.py)
- Contains the `solve_quiz` async function.
- Manages the loop: Visit URL -> Extract Task -> Solve -> Submit -> Handle Response.
- Handles the recursion if a new URL is provided.

#### [NEW] [agent.py](file:///c:/Users/hanso/OneDrive/Desktop/New folder/New folder/agent.py)
- Defines the LangChain agent.
- Configures the LLM (GPT-4o).
- Defines the tools available to the agent.

#### [NEW] [tools.py](file:///c:/Users/hanso/OneDrive/Desktop/New folder/New folder/tools.py)
- `read_page_content(url)`: Uses Playwright to render and extract text.
- `download_file(url)`: Downloads files for processing.
- `analyze_data(query, filepath)`: Pandas/Python REPL for data analysis.
- `visualize_data(query, data)`: Generates charts.

## Verification Plan

### Automated Tests
- Create a `mock_server.py` that simulates the quiz flow:
    - Serves a page with a JS-rendered question.
    - Accepts the POST submission.
    - Returns success/failure and potentially a next URL.
- Run `pytest` to verify the solver against the mock server.

### Manual Verification
- Run the mock server.
- Trigger the main API with a curl command.
- Observe the logs to see the agent solving the mock puzzle.
