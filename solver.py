import asyncio
import logging
import json
import os
import pdb
from langchain_core.globals import set_debug
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from tools import extract_page_data, download_file, transcribe_audio, ocr_image, read_text, read_binary, submit_answer, exec_py, visit_website
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
import time

logger = logging.getLogger(__name__)
set_debug(True)

async def solve_quiz(start_url: str, email: str, secret: str):
    """
    Main loop to solve the quiz using the 5-step procedural flow.
    """
    current_url = start_url
    
    # Initialize LLM
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-5-mini"), temperature=0)
    logger.info(f"Using LLM model: {os.getenv('LLM_MODEL', 'gpt-5-mini')}")
    
    # Tools for the solving agent
    solve_tools = [exec_py, visit_website, download_file, transcribe_audio, ocr_image, read_text, read_binary]
    
    # Tools for the extraction agent
    extraction_tools = [visit_website, read_text]

    retry_count = 0
    previously_tried_answers = []
    start_time = time.time()
    while current_url:
        logger.info("\n" + "="*50 + f" PROCESSING URL: {current_url} " + "="*50)
        
        # --- Step 1 & 2 & 3: Agentic Visit & Extraction ---
        logger.info("[Step 1-3] Agent visiting page and extracting details...")
        
        extraction_prompt = f"""
        You are a helper agent. 
        1. Use the `visit_website` tool to visit {current_url}.
        2. Analyze the content returned by the tool.
        3. Extract the submission URL and the required JSON format.
        4. User email is - {email}
        5. User secret is - {secret}
        
        Return a JSON object with:
        - "submission_url": The URL to submit to - MAKE SURE ITS THE FULL URL AND NOT RELATIVE.
        - "json_format": A sample JSON of what needs to be submitted (keys and value types).
        - "question": The specific question asked in the quiz. Your question will be used to run a solver agent.
        Return ONLY valid JSON.
        """
        
        agent = create_react_agent(llm, extraction_tools)
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=extraction_prompt)]},
            config={"recursion_limit": 100}
        )
        
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
        
        if len(previously_tried_answers) > 0:
            prev_tried_str += "Following answers were tried and are wrong:\n        " + "        \n".join(previously_tried_answers)
        else:
            prev_tried_str = ""
        solving_prompt = f"""
        You are a generic solver agent.
        
        Task: {submission_details.get('question')}
        If the question is not clear, you might need to process the downloaded files to understand the question.
        
        If you get a relative URL, use the base URL to make it absolute. Base URL: {current_url}

        IMPORTANT CREDENTIALS:
        - Your email is: {email}
        - Your secret is: {secret}
        - When making HTTP requests or submissions, ALWAYS use these credentials, NOT hardcoded values.

        Context:
        {context}

        {prev_tried_str}
        
        You have access to the following tools:
        - `exec_py`: Run Python code. Use this for calculations, data processing, etc.
        - `visit_website`: Visit a website and get its content
        - `download_file`: Download a file from a URL: download_file(url: str, filename: str = None) -> str (returns local file path)
        - `transcribe_audio`: Transcribe audio file: transcribe_audio(file_path: str) -> str
        - `ocr_image`: Extract text from images: ocr_image(image_path: str) -> str
        - `read_text`: Read text files: read_text(file_path: str) -> str
        - `read_binary`: Read binary files as base64: read_binary(file_path: str) -> str
        
        Files are located at the paths specified in "DOWNLOADED FILES".
        
        
        When using `exec_py`:
        
        Available Pre-imported Modules (50+ modules):
        
        File & Archive Operations:
        - base64, zipfile, tarfile, gzip, shutil, os, glob, tempfile, pathlib, io
        
        Data Formats:
        - csv, json, pickle
        
        Data Structures & Algorithms:
        - collections, itertools, functools, copy, heapq, bisect, array
        
        Math & Numbers:
        - math, random, statistics, decimal, fractions
        
        Text & String Processing:
        - re, string, textwrap
        
        Date & Time:
        - datetime, time, calendar
        
        Network & Web:
        - urllib, hashlib, hmac, secrets
        
        System & OS:
        - sys, platform, subprocess
        
        Utilities:
        - operator, typing, dataclasses, enum, contextlib, warnings, logging
        
        Third-party:
        - pd (pandas), np (numpy), httpx, geopy, fitz/pymupdf, folium
        - scipy, sknetwork, networkx, Image/ImageDraw/ImageFont (PIL)
        
        IMPORTANT:
        - You do NOT need to import any of these modules
        - For scraping, use `visit_website` tool first, but for API requests use `httpx` with `exec_py`
        - You MUST assign your final answer to a variable named `result`
        
        Examples:
        - Basic: result = pd.read_csv('file.csv')['value'].mean()
        - ZIP extraction: 
          with zipfile.ZipFile('file.zip', 'r') as z:
              z.extractall('extracted/')
          result = os.listdir('extracted/')
        - TAR extraction:
          with tarfile.open('file.tar.gz', 'r:gz') as tar:
              tar.extractall('extracted/')
          result = glob.glob('extracted/**/*.txt', recursive=True)
        - File operations:
          files = [f for f in os.listdir('.') if f.endswith('.log')]
          result = len(files)
        - Random sampling:
          result = random.sample(range(100), 10)
        - Statistics:
          data = [1, 2, 3, 4, 5]
          result = statistics.mean(data)
        
        
        Calculate the answer.
        
        Return ONLY the answer value encapsulated in a JSON. It could be a boolean, number, string, base64 URI of a file attachment, or a json object with a combination of these.

        Example: {{"answer": 42}}
        Example: {{"answer": "Paris"}}
        Example: {{"answer": true}}
        Example: {{"answer": {{"space": "Earth"}}}}
        Example: {{"answer": "d2F0ZXI="}}
        """
        
        agent = create_react_agent(llm, solve_tools)
        
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=solving_prompt)]},
            config={"recursion_limit": 100}
        )
        # pdb.set_trace()
        answer_str = result["messages"][-1].content.strip()
        try:
            answer_json = json.loads(answer_str)
            answer = answer_json.get("answer")
        except:
            logger.error(f"Error parsing answer: {answer_str}, using as is")
            answer = answer_str
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
        
        # Always use the correct submission endpoint
        # The extraction agent sometimes incorrectly identifies the page URL as the submission URL
        submit_url = "https://tds-llm-analysis.s-anand.net/submit" 
        
        logger.info(f"Submitting with {len(cookies)} cookies and referer {current_url}")
        response_text = submit_answer(submit_url, payload, cookies=cookies, referer=current_url)
        logger.info(f"--- SUBMISSION RESPONSE ---\n{response_text}\n--- SUBMISSION RESPONSE --- ")
        
        # Check response for next URL
        try:
            response_json = json.loads(response_text)
            if response_json.get("correct"):
                retry_count = 0
                error_retry_count = 0
                previously_tried_answers = []
                logger.info("✅ Answer Correct!")
                next_url = response_json.get("url")
                if next_url:
                    current_url = next_url
                    start_time = time.time()
                else:
                    logger.info("🎉 Quiz Finished!")
                    break
            else:
                logger.warning(f"❌ Answer Incorrect: {response_json.get('reason')}")
                end_time = time.time()
                time_taken = end_time - start_time
                logger.info(f"Time taken for this URL: {time_taken:.2f} seconds")
                avg_time_taken = (time_taken/(retry_count+1))
                approx_time_if_retry = time_taken + avg_time_taken
                if ((time_taken < 150) and (retry_count < 3) and (approx_time_if_retry < 180)) or (next_url == None) or (next_url.strip() == ''):
                    retry_count += 1
                    previously_tried_answers.append(
                        f"Previously tried wrong answer: {answer}. Reason: {response_json.get('reason')}"
                    )
                    logger.info("Retrying the same URL due to quick failure...")
                    error_retry_count = 0
                    continue
                else:
                    logger.info("Moving to next URL or ending due to retries.")
                    next_url = response_json.get("url")
                    previously_tried_answers = []
                    retry_count = 0
                    error_retry_count = 0
                    if next_url:
                        current_url = next_url
                        start_time = time.time()
                    else:
                        logger.info("🎉 Quiz Finished!")
                        break
        except Exception as e:
            logger.error(f"Error parsing submission response: {e}")
            if error_retry_count < 5:
                error_retry_count += 1
                logger.info("Submission error - retrying the same URL")
            else:
                logger.info("Too many submission errors - ending process")
                break
    
    logger.info("Request completed!")
    ## Buffering to ensure that the last log is written to the HF dataset
    for _ in range(10):
        logger.info("0"*104+"\n")

