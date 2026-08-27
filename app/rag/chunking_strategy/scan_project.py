from .text_chunker import chunk_text_file
from .chunk_python_file import chunk_python_file
from .chunk_js_ts_file import chunk_js_ts_file
from .file_support import get_file_type, IGNORED_DIRS
from pathlib import Path

# ==================================================
# CHECK IGNORED PATH
# ==================================================

def should_ignore(path: Path):

    
    return any(
        part in IGNORED_DIRS
        for part in path.parts
    )

# ==================================================
# SCAN PROJECT
# ==================================================

def scan_project(project_dir: Path):

    all_chunks = []

    for path in project_dir.rglob("*"):

        if not path.is_file():
            continue

        if should_ignore(path):
            continue

        file_type = get_file_type(path)

        if file_type is None:
            continue

        print(f"Processing: {path}")

        if file_type == "python":

            chunks = chunk_python_file(path)

        elif file_type == "javascript":

            chunks = chunk_js_ts_file(path)

        else:

            chunks = chunk_text_file(path)

        all_chunks.extend(chunks)

    return all_chunks