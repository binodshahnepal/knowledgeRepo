import os
from dotenv import load_dotenv

load_dotenv()

APP_TITLE = "Universal Knowledge Assistant"
APP_ICON = "🧠"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
VECTOR_DIR = os.path.join(DATA_DIR, "vectorstore")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt", "csv", "xlsx", "xls", "json", "md"]

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gemini-flash-latest"

TOP_K = 5
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def ensure_directories():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)