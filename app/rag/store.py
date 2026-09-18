from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# ==================================================
# PATHS (anchored to project root, not cwd)
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CHUNKS_FILE = PROJECT_ROOT / "chunks.json"
MANIFEST_FILE = PROJECT_ROOT / "index_manifest.json"


# ==================================================
# EMBEDDINGS
# ==================================================

EMBEDDING_MODEL = "nomic-embed-text"

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# ==================================================
# VECTOR STORE
# ==================================================

vector_store = Chroma(
    collection_name="codebase",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)
