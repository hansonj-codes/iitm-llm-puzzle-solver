import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from langchain.tools import tool
from playwright.async_api import async_playwright
import logging
from openai import OpenAI
import uuid
import json

logger = logging.getLogger(__name__)

async def extract_page_data(url: str):
    """
    Visits a URL using Playwright and extracts:
    - Visible text content
    - Audio/Video links
    - File links
    Returns a dictionary with these details.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        page = await context.new_page()
        try:
            logger.info(f"Visiting page: {url}")
            await page.goto(url)
            await page.wait_for_timeout(3000) 
            content = await page.content()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # 1. Extract visible text
            text = soup.get_text(separator='\n', strip=True)
            
            # 2. Extract Audio/Video Links
            media_links = []
            # Tags
            for tag in soup.find_all(['audio', 'video']):
                if tag.get('src'):
                    media_links.append(tag['src'])
                for source in tag.find_all('source'):
                    if source.get('src'):
                        media_links.append(source['src'])
            # Links
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac', '.mp4', '.mpeg')):
                    media_links.append(href)
            
            # 3. Extract File Links (generic)
            file_links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                # Exclude media and common non-files
                if not href.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac', '.mp4', '.mpeg', '#')):
                    file_links.append(href)

            # Normalize URLs
            media_links = [requests.compat.urljoin(url, src) for src in set(media_links)]
            file_links = [requests.compat.urljoin(url, src) for src in set(file_links)]
            
            cookies = await context.cookies()
            
            return {
                "text": text,
                "media_links": media_links,
                "file_links": file_links,
                "cookies": cookies
            }

        except Exception as e:
            logger.error(f"Error extracting page data: {e}")
            return {"text": f"Error: {e}", "media_links": [], "file_links": [], "cookies": []}
        finally:
            await browser.close()

def download_file(url: str, filename: str = None) -> str:
    """
    Downloads a file from a URL. 
    Returns the local path.
    """
    try:
        logger.info(f"Downloading file from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        if not filename:
            if "Content-Disposition" in response.headers:
                import re
                fname = re.findall("filename=(.+)", response.headers["Content-Disposition"])
                if fname:
                    filename = fname[0].strip('"')
            
            if not filename:
                filename = url.split("/")[-1].split("?")[0]
        
        # Ensure extension based on Content-Type
        content_type = response.headers.get("Content-Type", "").lower()
        ext_map = {
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/mp4": ".m4a",
            "audio/x-m4a": ".m4a",
            "application/json": ".json",
            "application/pdf": ".pdf",
            "text/csv": ".csv",
            "text/plain": ".txt"
        }
        
        if "." not in filename:
            for ctype, ext in ext_map.items():
                if ctype in content_type:
                    filename += ext
                    break
        
        # Use a unique name to avoid collisions if not provided
        if not filename:
            filename = f"download_{uuid.uuid4()}"
            
        path = os.path.join(os.getcwd(), filename)
        
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"File downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return f"Error: {e}"

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes an audio file using OpenAI's Whisper model.
    """
    try:
        logger.info(f"Transcribing audio file: {file_path}")
        
        # Convert to mp3 if needed (using pydub)
        if not file_path.lower().endswith('.mp3'):
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(file_path)
                mp3_path = file_path + ".mp3"
                audio.export(mp3_path, format="mp3")
                file_path = mp3_path
                logger.info(f"Converted to MP3: {file_path}")
            except Exception as e:
                logger.warning(f"Could not convert to MP3, trying original file: {e}")

        client = OpenAI()
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        text = transcription.text
        logger.info(f"Transcription result: {text}")
        return text
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return f"Error transcribing audio: {e}"

def submit_answer(submission_url: str, payload: dict, cookies: list = None, referer: str = None) -> str:
    """
    Submits the answer to the quiz.
    """
    try:
        logger.info(f"Submitting payload to {submission_url}: {payload}")
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        if referer:
            headers['Referer'] = referer
            
        # Convert Playwright cookies (list of dicts) to Requests cookies (dict)
        req_cookies = {}
        if cookies:
            for c in cookies:
                req_cookies[c['name']] = c['value']
                
        response = requests.post(submission_url, json=payload, headers=headers, cookies=req_cookies)
        logger.info(f"Submission response: {response.text}")
        return response.text
    except Exception as e:
        logger.error(f"Submission error: {e}")
        return f"Error submitting answer: {e}"

# --- Agent Tools ---
# These are wrappers for the agent to use if needed, 
# but the main flow will use the functions above directly.

@tool("exec_py", return_direct=False)
def exec_py(code: str) -> str:
    """
    Execute user-provided Python code in a restricted namespace.
    
    Pre-imported modules available in the namespace:
    - pd (pandas)
    - np (numpy)
    - json
    - math
    - re
    - datetime
    - httpx
    
    You do NOT need to import these.
    
    IMPORTANT: Assign the final answer/result to a variable named `result`.
    Example:
    result = pd.read_csv('file.csv').mean()
    """
    # Do not allow import statements — you can detect that and reject
    if "import " in code:
        return "Rejected: import statements are not allowed. Use pre-imported modules: pd, np, json, math, re, datetime, httpx."

    try:
        import pandas as pd
        import numpy as np
        import json
        import math
        import re
        import datetime
        import httpx
        import geopy
        import fitz # pymupdf
        import folium
        
        # VERY simple sandbox: provide only safe modules
        safe_globals = {
            "__builtins__": {
                "len": len, "range": range, "print": print, "min": min, "max": max,
                "int": int, "float": float, "str": str, "list": list, "dict": dict, "set": set, "open": open
            },
            "pd": pd,
            "np": np,
            "json": json,
            "math": math,
            "re": re,
            "datetime": datetime,
            "httpx": httpx,
            "geopy": geopy,
            "fitz": fitz,
            "pymupdf": fitz, # alias
            "folium": folium
        }
        
        exec(code, safe_globals)
        # convention: user sets variable `result`
        return str(safe_globals.get("result", "Code executed successfully, but no 'result' variable was set."))
    except Exception as e:
        return f"Error: {e}"

@tool("visit_website", return_direct=False)
async def visit_website(url: str) -> str:
    """
    Visits a website, extracts text, downloads files/media, and transcribes audio.
    Returns a comprehensive context string containing page content, file paths, and transcriptions.
    """
    try:
        logger.info(f"Agent visiting: {url}")
        page_data = await extract_page_data(url)
        page_text = page_data["text"]
        cookies = page_data.get("cookies", [])
        
        downloaded_files = []
        
        # Download Media
        for media_link in page_data["media_links"]:
            path = download_file(media_link)
            if "Error" not in path:
                downloaded_files.append({"type": "media", "path": path, "url": media_link})
        
        # Download Other Files
        for file_link in page_data["file_links"]:
            if file_link.lower().endswith(('.csv', '.json', '.pdf', '.xlsx', '.txt')):
                path = download_file(file_link)
                if "Error" not in path:
                    downloaded_files.append({"type": "file", "path": path, "url": file_link})

        # Audio Transcription
        transcriptions = []
        for f in downloaded_files:
            if f["type"] == "media":
                logger.info(f"Transcribing audio: {f['path']}")
                text = transcribe_audio(f["path"])
                transcriptions.append(f"Transcription of {f['url']}:\n{text}")
        
        # Prepare context
        context = f"""
        --- PAGE CONTENT ---
        {page_text}
        
        --- DOWNLOADED FILES ---
        {json.dumps(downloaded_files, indent=2)}
        
        --- AUDIO TRANSCRIPTIONS ---
        {chr(10).join(transcriptions)}
        
        --- COOKIES ---
        {json.dumps(cookies)}
        """
        
        return context
    except Exception as e:
        logger.error(f"Error in visit_website: {e}")
        return f"Error visiting website: {e}"
