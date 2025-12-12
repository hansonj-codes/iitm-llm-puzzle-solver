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
            
            # Save HTML content to file
            source_filename = f"source_{uuid.uuid4()}.html"
            source_path = os.path.join(os.getcwd(), source_filename)
            with open(source_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Saved page source to: {source_path}")

            cookies = await context.cookies()
            
            return {
                "text": text,
                "media_links": media_links,
                "file_links": file_links,
                "cookies": cookies,
                "source_path": source_path
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

def ocr_image(image_path: str) -> str:
    """
    Performs OCR on an image using OpenAI's vision model.
    Returns the extracted text from the image.
    """
    try:
        logger.info(f"Performing OCR on image: {image_path}")
        
        import base64
        
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Determine the image format
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this image. Return only the text content, nothing else."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        text = response.choices[0].message.content
        logger.info(f"OCR result: {text}")
        return text
    except Exception as e:
        logger.error(f"Error performing OCR: {e}")
        return f"Error performing OCR: {e}"

def read_text(file_path: str) -> str:
    """
    Reads a text file and returns its content.
    """
    try:
        logger.info(f"Reading text file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Successfully read {len(content)} characters from {file_path}")
        return content
    except Exception as e:
        logger.error(f"Error reading text file: {e}")
        return f"Error reading text file: {e}"

def read_binary(file_path: str) -> str:
    """
    Reads a binary file, encodes it as base64, and returns the encoded string.
    """
    try:
        import base64
        logger.info(f"Reading binary file: {file_path}")
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        encoded = base64.b64encode(binary_data).decode('utf-8')
        logger.info(f"Successfully encoded {len(binary_data)} bytes from {file_path}")
        return encoded
    except Exception as e:
        logger.error(f"Error reading binary file: {e}")
        return f"Error reading binary file: {e}"

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
    
    Pre-imported modules available (50+ modules):
    
    File & Archive Operations:
    - base64, zipfile, tarfile, gzip, shutil
    - os, glob, tempfile, pathlib, io
    
    Data Formats:
    - csv, json, pickle
    
    Data Structures & Algorithms:
    - collections, itertools, functools, copy
    - heapq, bisect, array
    
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
    - operator, typing, dataclasses, enum
    - contextlib, warnings, logging
    
    Third-party:
    - pd (pandas), np (numpy)
    - httpx, geopy, fitz/pymupdf, folium
    - scipy, sknetwork, networkx
    - Image, ImageDraw, ImageFont (from PIL)
    
    You do NOT need to import any of these modules.
    
    IMPORTANT: Assign the final answer/result to a variable named `result`.
    Example:
    result = pd.read_csv('file.csv').mean()
    """
    # Do not allow import statements — you can detect that and reject
    if "import " in code:
        return "Rejected: import statements are not allowed. Use pre-imported modules. Available: zipfile, tarfile, gzip, shutil, os, glob, tempfile, pathlib, io, csv, json, pickle, collections, itertools, functools, copy, heapq, bisect, array, math, random, statistics, decimal, fractions, re, string, textwrap, datetime, time, calendar, urllib, hashlib, hmac, secrets, sys, platform, subprocess, operator, typing, dataclasses, enum, contextlib, warnings, logging, base64, pd, np, httpx, geopy, fitz, folium, scipy, sknetwork, networkx, Image, etc."

    try:
        # Standard library imports - File & Archive Operations
        import base64
        import zipfile
        import tarfile
        import gzip
        import shutil
        import os
        import glob
        import tempfile
        import pathlib
        import io
        
        # Standard library imports - Data Formats
        import csv
        import json
        import pickle
        
        # Standard library imports - Data Structures & Algorithms
        import collections
        import itertools
        import functools
        import copy
        import heapq
        import bisect
        import array
        
        # Standard library imports - Math & Numbers
        import math
        import random
        import statistics
        import decimal
        import fractions
        
        # Standard library imports - Text & String Processing
        import re
        import string
        import textwrap
        
        # Standard library imports - Date & Time
        import datetime
        import time
        import calendar
        
        # Standard library imports - Network & Web
        import urllib
        import hashlib
        import hmac
        import secrets
        
        # Standard library imports - System & OS
        import sys
        import platform
        import subprocess
        
        # Standard library imports - Utilities
        import operator
        import typing
        import dataclasses
        import enum
        import contextlib
        import warnings
        import logging
        
        # Third-party imports
        import pandas as pd
        import numpy as np
        import httpx
        import geopy
        import fitz # pymupdf
        import folium
        import scipy
        import sknetwork
        import networkx
        from PIL import Image, ImageDraw, ImageFont
        
        # VERY simple sandbox: provide only safe modules
        safe_builtins = {
            # Basic types / constructors
            "int": int,
            "float": float,
            "complex": complex,
            "str": str,
            "bytes": bytes,
            "bytearray": bytearray,
            "bool": bool,
            "list": list,
            "tuple": tuple,
            "set": set,
            "frozenset": frozenset,
            "dict": dict,

            # Basic object helpers (non-reflective)
            "len": len,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "round": round,
            "sorted": sorted,
            "reversed": reversed,
            "enumerate": enumerate,
            "range": range,
            "zip": zip,
            "map": map,
            "filter": filter,

            # predicates / small utilities
            "all": all,
            "any": any,
            "isinstance": isinstance,
            "issubclass": issubclass,

            # conversions (safe)
            "ord": ord,
            "chr": chr,
            "hex": hex,
            "oct": oct,

            # small helpers
            "print": print,           # keep for debugging (stdout capture recommended)
            "repr": repr,
        }
        safe_globals = {
            "__builtins__": safe_builtins,
            
            # File & Archive Operations
            "base64": base64,
            "zipfile": zipfile,
            "tarfile": tarfile,
            "gzip": gzip,
            "shutil": shutil,
            "os": os,
            "glob": glob,
            "tempfile": tempfile,
            "pathlib": pathlib,
            "io": io,
            
            # Data Formats
            "csv": csv,
            "json": json,
            "pickle": pickle,
            
            # Data Structures & Algorithms
            "collections": collections,
            "itertools": itertools,
            "functools": functools,
            "copy": copy,
            "heapq": heapq,
            "bisect": bisect,
            "array": array,
            
            # Math & Numbers
            "math": math,
            "random": random,
            "statistics": statistics,
            "decimal": decimal,
            "fractions": fractions,
            
            # Text & String Processing
            "re": re,
            "string": string,
            "textwrap": textwrap,
            
            # Date & Time
            "datetime": datetime,
            "time": time,
            "calendar": calendar,
            
            # Network & Web
            "urllib": urllib,
            "hashlib": hashlib,
            "hmac": hmac,
            "secrets": secrets,
            
            # System & OS
            "sys": sys,
            "platform": platform,
            "subprocess": subprocess,
            
            # Utilities
            "operator": operator,
            "typing": typing,
            "dataclasses": dataclasses,
            "enum": enum,
            "contextlib": contextlib,
            "warnings": warnings,
            "logging": logging,
            
            # Third-party modules
            "pd": pd,
            "np": np,
            "httpx": httpx,
            "geopy": geopy,
            "fitz": fitz,
            "pymupdf": fitz, # alias
            "folium": folium,
            "scipy": scipy,
            "sknetwork": sknetwork,
            "networkx": networkx,
            "Image": Image,
            "ImageDraw": ImageDraw,
            "ImageFont": ImageFont
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
        source_path = page_data.get("source_path", "")
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
        
        --- PAGE SOURCE ---
        Path: {source_path}
        Instruction: You can read this file using the read_text tool if you need to inspect the HTML source code.

        --- COOKIES ---
        {json.dumps(cookies)}
        """
        
        return context
    except Exception as e:
        logger.error(f"Error in visit_website: {e}")
        return f"Error visiting website: {e}"
