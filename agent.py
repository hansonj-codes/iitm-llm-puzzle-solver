import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_experimental.tools import PythonREPLTool
from tools import read_page_content, download_file, submit_answer, transcribe_audio

def get_agent_executor():
    """
    Creates and returns a LangGraph agent with the necessary tools.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    tools = [
        read_page_content,
        download_file,
        submit_answer,
        transcribe_audio,
        PythonREPLTool() # For data analysis and calculations
    ]
    
    # Create the agent using LangGraph
    # This returns a CompiledGraph which can be invoked
    agent = create_react_agent(llm, tools)
    
    return agent
