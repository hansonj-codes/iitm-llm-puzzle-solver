import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# Ensure we can import from current directory
sys.path.append(os.getcwd())

from solver import solve_quiz

# Configure logging to show up in stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()

async def main():
    email = "student@example.com"
    secret = "default_secret"
    url = "https://tds-llm-analysis.s-anand.net/demo"
    
    print(f"Starting live test with URL: {url}")
    print(f"Email: {email}")
    
    await solve_quiz(url, email, secret)

if __name__ == "__main__":
    asyncio.run(main())
