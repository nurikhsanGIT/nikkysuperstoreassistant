import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
CHROMA_DB_DIR = os.path.join(DATABASE_DIR, "chroma_db")

# Ensure directories exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Ollama Config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Embeddings Config
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
