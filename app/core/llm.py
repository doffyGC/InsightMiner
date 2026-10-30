from agno.models.groq import Groq
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"  # Groq model


def get_llm(model_id: str | None = None):
    return Groq(id=model_id or DEFAULT_MODEL)
