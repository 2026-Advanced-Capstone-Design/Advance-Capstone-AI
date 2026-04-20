import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
GPT_MINI_MODEL = "gpt-4o-mini"
GPT_STRONG_MODEL = "gpt-4o"
SPRING_CALLBACK_URL = os.getenv("SPRING_CALLBACK_URL", "http://localhost:8080/api/v1/internal/callback")
