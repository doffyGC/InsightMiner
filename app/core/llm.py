from os import getenv
from agno.models.groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return Groq(
        id="llama-3.3-70b-versatile",  # Groq Model    
        )