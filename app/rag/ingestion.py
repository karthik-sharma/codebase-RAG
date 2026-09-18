from pathlib import Path
import json

from chunking_strategy.scan_project import scan_project
from chunking_strategy.chunk_documents import chunks_to_documents
from chunking_strategy.manifest import load_manifest, save_manifest
from store import vector_store, CHUNKS_FILE, MANIFEST_FILE


# ==================================================
# PROJECT
# ==================================================

PROJECT_DIR = Path(
    "/Users/karthik/Documents/python/ExpenseTracker"
)


# ==================================================
# LOAD PREVIOUS STATE (for incremental indexing)
# ==================================================

# No manifest yet -> nothing to diff against, so this run must be a full
# scan + full rebuild (also migrates a pre-incremental index, whose chunk
# IDs used a different scheme, to the current one).
is_bootstrap = not MANIFEST_FILE.exists()

previous_manifest = None
previous_chunks_by_file = None

if not is_bootstrap:

    previous_manifest = load_manifest(MANIFEST_FILE)

    previous_chunks_by_file = {}

    if CHUNKS_FILE.exists():

        previous_chunks = json.loads(
            CHUNKS_FILE.read_text(encoding="utf-8")
        )

        for chunk in previous_chunks:

            previous_chunks_by_file.setdefault(
                chunk["file"], []
            ).append(chunk)


# ==================================================
# SCAN PROJECT
# ==================================================

chunks, manifest, deleted_files = scan_project(
    PROJECT_DIR,
    previous_manifest=previous_manifest,
    previous_chunks_by_file=previous_chunks_by_file,
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
# UPDATE CHROMA
# ==================================================

if is_bootstrap:

    print("No previous index found - building from scratch.")

    vector_store.reset_collection()

    documents = chunks_to_documents(chunks)

    vector_store.add_documents(
        documents,
        ids=[doc.metadata["chunk_id"] for doc in documents],
    )

    print(f"Created {len(documents)} Documents")

else:

    changed_or_new_files = {
        file_key for file_key, file_hash in manifest.items()
        if previous_manifest.get(file_key) != file_hash
    }

    files_needing_delete = changed_or_new_files | set(deleted_files)

    ids_to_delete = [
        f"{chunk['file']}:{chunk.get('chunk_index', 0)}"
        for file_key in files_needing_delete
        for chunk in previous_chunks_by_file.get(file_key, [])
    ]

    if ids_to_delete:

        vector_store.delete(ids=ids_to_delete)

        print(f"Removed {len(ids_to_delete)} stale chunks")

    chunks_to_add = [
        chunk for chunk in chunks
        if chunk["file"] in changed_or_new_files
    ]

    documents = chunks_to_documents(chunks_to_add)

    if documents:

        vector_store.add_documents(
            documents,
            ids=[doc.metadata["chunk_id"] for doc in documents],
        )

        print(f"Added/updated {len(documents)} chunks")

    if not ids_to_delete and not documents:

        print("No changes detected - index already up to date.")

    if deleted_files:

        print(f"{len(deleted_files)} file(s) removed from the project")


# ==================================================
# SAVE MANIFEST
# ==================================================

save_manifest(MANIFEST_FILE, manifest)


print(
    "\nIngestion complete!"
)
