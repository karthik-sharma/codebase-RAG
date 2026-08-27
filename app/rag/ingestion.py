from pathlib import Path
import json

from chunking_strategy.scan_project import scan_project
from chunking_strategy.chunk_documents import chunks_to_documents
from store import vector_store, CHUNKS_FILE


# ==================================================
# PROJECT
# ==================================================

PROJECT_DIR = Path(
    "/Users/karthik/Documents/python/ExpenseTracker"
)


# ==================================================
# SCAN PROJECT
# ==================================================

chunks = scan_project(
    PROJECT_DIR
)

print(
    f"\nFound {len(chunks)} chunks"
)


# ==================================================
# SAVE CHUNKS FOR DEBUGGING
# ==================================================

with open(
    CHUNKS_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        chunks,
        f,
        indent=2,
    )


# ==================================================
# CONVERT TO DOCUMENTS
# ==================================================

documents = chunks_to_documents(
    chunks
)

print(
    f"Created {len(documents)} Documents"
)


# ==================================================
# STORE IN CHROMA
# ==================================================

# Reset so re-running ingestion doesn't duplicate every
# chunk (add_documents has no dedup/upsert without explicit ids).

vector_store.reset_collection()

vector_store.add_documents(
    documents
)


print(
    "\nIngestion complete!"
)