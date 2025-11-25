from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import base64

app = FastAPI()

@app.get("/quiz-start", response_class=HTMLResponse)
async def quiz_start():
    html_content = """
    <html>
        <body>
            <h1>Quiz Task 1</h1>
            <div id="question">
                Calculate the sum of 10 + 20.
            </div>
            <p>Submit your answer to http://localhost:8001/submit</p>
            <pre>
            {
                "email": "student@example.com",
                "secret": "default_secret",
                "url": "http://localhost:8001/quiz-start",
                "answer": 30
            }
            </pre>
        </body>
    </html>
    """
    return html_content

@app.post("/submit")
async def submit(request: Request):
    data = await request.json()
    if data.get("answer") == 30:
        return JSONResponse(content={
            "correct": True,
            "url": "http://localhost:8001/quiz-2",
            "reason": "Correct!"
        })
    return JSONResponse(content={"correct": False, "reason": "Wrong answer"})

@app.get("/quiz-2", response_class=HTMLResponse)
async def quiz_2():
    html_content = """
    <html>
        <body>
            <h1>Quiz Task 2</h1>
            <div id="question">
                What is the capital of France?
            </div>
            <p>Submit your answer to http://localhost:8001/submit-2</p>
        </body>
    </html>
    """
    return html_content

@app.post("/submit-2")
async def submit_2(request: Request):
    data = await request.json()
    if str(data.get("answer")).lower() == "paris":
        return JSONResponse(content={
            "correct": True,
            # No new URL means done
            "reason": "Correct!"
        })
    return JSONResponse(content={"correct": False, "reason": "Wrong answer"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
