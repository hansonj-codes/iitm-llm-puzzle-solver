import asyncio
import logging
import json
import os
from langchain_core.globals import set_debug
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from tools import extract_page_data, download_file, transcribe_audio, submit_answer, python_repl
from langchain_experimental.tools import PythonREPLTool

logger = logging.getLogger(__name__)
set_debug(True)

async def solve_quiz(start_url: str, email: str, secret: str):
    """
    Main loop to solve the quiz using the 5-step procedural flow.
    """
    current_url = start_url
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    
    # Tools for the solving agent
    solve_tools = [PythonREPLTool()]

    while current_url:
        logger.info("\n" + "="*50 + f" PROCESSING URL: {current_url} " + "="*50)
        
        # --- Step 1: Deterministic Page Visit & Download ---
        logger.info("[Step 1] Visiting page and extracting data...")
        page_data = await extract_page_data(current_url)
        page_text = page_data["text"]
        
        downloaded_files = []
        
        # Download Media
        for media_link in page_data["media_links"]:
            path = download_file(media_link)
            if "Error" not in path:
                downloaded_files.append({"type": "media", "path": path, "url": media_link})
        
        # Download Other Files
        for file_link in page_data["file_links"]:
            # Simple heuristic: if it looks like a data file
            if file_link.lower().endswith(('.csv', '.json', '.pdf', '.xlsx', '.txt')):
                path = download_file(file_link)
                if "Error" not in path:
                    downloaded_files.append({"type": "file", "path": path, "url": file_link})

        # --- Step 2: Audio Transcription ---
        transcriptions = []
        for f in downloaded_files:
            if f["type"] == "media":
                logger.info(f"[Step 2] Transcribing audio: {f['path']}")
                text = transcribe_audio(f["path"])
                transcriptions.append(f"Transcription of {f['url']}:\n{text}")
        
        # Prepare context for the agent
        context = f"""
        --- PAGE CONTENT ---
        {page_text}
        
        --- DOWNLOADED FILES ---
        {json.dumps(downloaded_files, indent=2)}
        
        --- AUDIO TRANSCRIPTIONS ---
        {chr(10).join(transcriptions)}
        """
        
        logger.info(f"[Context Prepared] Length: {len(context)}")
        logger.info(f"--- CONTEXT --->\n{context}\n --- CONTEXT ---")

        # --- Step 3: Agent Extraction (Submit URL & Format) ---
        logger.info("[Step 3] Extracting submission details...")
        extraction_prompt = f"""
        You are a helper agent. Your ONLY task is to extract the submission URL and the required JSON format from the provided text.
        
        Base URL is: {current_url}

        Text:
        {context}
        
        Return a JSON object with:
        - "submission_url": The URL to submit to - MAKE SURE ITS THE FULL URL AND NOT RELATIVE.
        - "json_format": A sample JSON of what needs to be submitted (keys and value types).
        - "question": The specific question asked in the quiz. Your question will be used to run a solver agent. But the solver agent should not POST the final answer to the submission URL. So, amend the question in such a way that the final submission by POST is not present.
        
        Return ONLY valid JSON.
        """
        
        extraction_response = await llm.ainvoke([HumanMessage(content=extraction_prompt)])
        logger.info(f"--- EXTRACTION RESPONSE --->\n{extraction_response}\n--- EXTRACTION RESPONSE ---")
        try:
            # Clean up markdown code blocks if present
            content = extraction_response.content.replace("```json", "").replace("```", "").strip()
            submission_details = json.loads(content)
            logger.info(f"[Step 3 Result] {submission_details}")
        except Exception as e:
            logger.error(f"Error parsing extraction response: {e}")
            break

        # --- Step 4: Agent Solving ---
        logger.info("[Step 4] Solving the problem...")
        
        solving_prompt = f"""
        You are a generic solver agent.
        
        Task: {submission_details.get('question')}
        
        Context:
        {context}
        
        You have access to a Python REPL to analyze any downloaded files.
        Files are located at the paths specified in "DOWNLOADED FILES".
        
        Calculate the answer.
        
        Return ONLY the answer value. It could be a number, string, or boolean or a json.
        Just return the raw answer.
        """
        
        # We use a simple agent loop here or just the LLM with tools
        from langgraph.prebuilt import create_react_agent
        agent = create_react_agent(llm, solve_tools)
        
        result = await agent.ainvoke({"messages": [HumanMessage(content=solving_prompt)]})
        answer = result["messages"][-1].content.strip()
        logger.info(f"[Step 4 Result] Answer: {answer}")

        # --- Step 5: Submission ---
        logger.info("[Step 5] Submitting answer...")
        
        # Construct payload
        payload = {
            "email": email,
            "secret": secret,
            "url": current_url,
            "answer": answer
        }
        
        # Adjust payload based on extracted format if needed (e.g. if keys are different)
        # But usually it's standard. The user said "answer" field.
        # If the extracted format shows different keys, we might need to be smarter.
        # For now, let's assume standard format or try to merge.
        
        submit_url = submission_details.get("submission_url")
        if not submit_url:
            # Fallback if extraction failed
            submit_url = "https://tds-llm-analysis.s-anand.net/submit" # Guess or fail
            
        response_text = submit_answer(submit_url, payload)
        logger.info(f"--- SUBMISSION RESPONSE --->\n{response_text}\n--- SUBMISSION RESPONSE --- ")
        
        # Check response for next URL
        try:
            response_json = json.loads(response_text)
            if response_json.get("correct"):
                logger.info("✅ Answer Correct!")
                next_url = response_json.get("url")
                if next_url:
                    current_url = next_url
                else:
                    logger.info("🎉 Quiz Finished!")
                    break
            else:
                logger.warning(f"❌ Answer Incorrect: {response_json.get('reason')}")
                # Retry logic? For now, we break to avoid infinite loops of wrong answers
                # Or we could retry Step 4 with feedback.
                break
        except Exception as e:
            logger.error(f"Error parsing submission response: {e}")
            break

        input("Press Enter to continue...")
