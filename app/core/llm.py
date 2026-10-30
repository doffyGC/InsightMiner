from agno.models.groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return Groq(
        id="openai/gpt-oss-120b",  # Groq Model    
        )