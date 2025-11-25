import asyncio
import logging
import json
import os
from langchain_core.globals import set_debug
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from tools import extract_page_data, download_file, transcribe_audio, submit_answer, exec_py, visit_website
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)
set_debug(True)

async def solve_quiz(start_url: str, email: str, secret: str):
    """
    Main loop to solve the quiz using the 5-step procedural flow.
    """
    current_url = start_url
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Tools for the solving agent
    solve_tools = [exec_py, visit_website]
    
    # Tools for the extraction agent
    extraction_tools = [visit_website]

    while current_url:
        logger.info("\n" + "="*50 + f" PROCESSING URL: {current_url} " + "="*50)
        
        # --- Step 1 & 2 & 3: Agentic Visit & Extraction ---
        logger.info("[Step 1-3] Agent visiting page and extracting details...")
        
        extraction_prompt = f"""
        You are a helper agent. 
        1. Use the `visit_website` tool to visit {current_url}.
        2. Analyze the content returned by the tool.
        3. Extract the submission URL and the required JSON format.
        
        Return a JSON object with:
        - "submission_url": The URL to submit to - MAKE SURE ITS THE FULL URL AND NOT RELATIVE.
        - "json_format": A sample JSON of what needs to be submitted (keys and value types).
        - "question": The specific question asked in the quiz. Your question will be used to run a solver agent. But the solver agent should not POST the final answer to the submission URL. So, amend the question in such a way that the final submission by POST is not present.
        
        Return ONLY valid JSON.
        """
        
        agent = create_react_agent(llm, extraction_tools)
        result = await agent.ainvoke({"messages": [HumanMessage(content=extraction_prompt)]})
        
        # Capture Context from Tool Output
        context = ""
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage) and msg.name == "visit_website":
                context = msg.content
                break
        
        if not context:
            logger.error("Could not retrieve context from visit_website tool.")
            break
            
        logger.info(f"[Context Retrieved] Length: {len(context)}")
        
        # Debug: Log all messages
        logger.info("--- AGENT MESSAGE HISTORY ---")
        for i, msg in enumerate(result["messages"]):
            logger.info(f"Msg {i} [{type(msg).__name__}]: {msg.content[:200]}...")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                logger.info(f"  Tool Calls: {msg.tool_calls}")
        logger.info("-----------------------------")

        # Find the last AIMessage that is not a tool call
        extraction_response = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                extraction_response = msg.content
                break
        
        logger.info(f"--- EXTRACTION RESPONSE --->\n{extraction_response}\n--- EXTRACTION RESPONSE ---")
        
        try:
            # Robust JSON extraction using regex
            import re
            json_match = re.search(r'\{.*\}', extraction_response, re.DOTALL)
            if json_match:
                content = json_match.group(0)
                submission_details = json.loads(content)
                logger.info(f"[Step 3 Result] {submission_details}")
                
                # Extract cookies from context for submission
                cookies_match = re.search(r'--- COOKIES ---\s*(\[.*\])', context, re.DOTALL)
                cookies = []
                if cookies_match:
                    try:
                        cookies = json.loads(cookies_match.group(1))
                    except:
                        pass
            else:
                logger.error("No JSON found in extraction response")
                break
                    
        except Exception as e:
            logger.error(f"Error parsing extraction response: {e}")
            break

        # --- Step 4: Agent Solving ---
        logger.info("[Step 4] Solving the problem...")
        
        solving_prompt = f"""
        You are a generic solver agent.
        
        Task: {submission_details.get('question')}
        
        If you get a relative URL, use the base URL to make it absolute. Base URL: {current_url}

        Context:
        {context}
        
        You have access to a python tool `exec_py` run Python code. Use this function for reading downloaded files, doing calculations etc.
        Files are located at the paths specified in "DOWNLOADED FILES".
        To visit a website, you can you Use the `visit_website` tool
        
        When using `exec_py`:
        - You do NOT need to import pandas (pd), numpy (np), json, math, re, datetime, geopy, fitz, pymupdf, folium, or httpx. They are pre-imported.
        - You MUST assign your final answer to a variable named `result`.
        - Example: result = pd.read_csv('file.csv')['value'].mean()
        
        Calculate the answer.
        
        Return ONLY the answer value. It could be a number, string, or boolean or a json.
        Just return the raw answer.
        """
        
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
        
        submit_url = submission_details.get("submission_url")
        if not submit_url:
            # Fallback if extraction failed
            submit_url = "https://tds-llm-analysis.s-anand.net/submit" 
        
        logger.info(f"Submitting with {len(cookies)} cookies and referer {current_url}")
        response_text = submit_answer(submit_url, payload, cookies=cookies, referer=current_url)
        logger.info(f"--- SUBMISSION RESPONSE ---\n{response_text}\n--- SUBMISSION RESPONSE --- ")
        
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
                break
        except Exception as e:
            logger.error(f"Error parsing submission response: {e}")
            break

        input("Press Enter to continue...")
