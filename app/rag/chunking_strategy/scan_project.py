from pathlib import Path

import pathspec

from .manifest import hash_file
from .text_chunker import chunk_text_file
from .chunk_python_file import chunk_python_file
from .chunk_js_ts_file import chunk_js_ts_file
from .file_support import get_file_type, IGNORED_DIRS

# ==================================================
# LOAD ALL .gitignore FILES IN THE PROJECT
# ==================================================

def load_gitignore_specs(project_dir: Path):

    specs = {}

    for gitignore_path in project_dir.rglob(".gitignore"):

        if any(
            part in IGNORED_DIRS
            for part in gitignore_path.parts
        ):
            continue

        lines = gitignore_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()

        specs[gitignore_path.parent] = pathspec.PathSpec.from_lines(
            "gitwildmatch",
            lines,
        )

    return specs

# ==================================================
# CHECK IGNORED PATH
# ==================================================

def should_ignore(path: Path, gitignore_specs):

    if any(
        part in IGNORED_DIRS
        for part in path.parts
    ):
        return True

    for gitignore_dir, spec in gitignore_specs.items():

        if gitignore_dir not in path.parents:
            continue

        relative_path = path.relative_to(gitignore_dir)

        if spec.match_file(relative_path.as_posix()):
            return True

    return False

# ==================================================
# DISPATCH TO THE RIGHT CHUNKER
# ==================================================

def chunk_file(path: Path, file_type: str):

    if file_type == "python":
        return chunk_python_file(path)

    if file_type == "javascript":
        return chunk_js_ts_file(path)

    return chunk_text_file(path)

# ==================================================
# SCAN PROJECT (full scan, or incremental scan when
# previous_manifest / previous_chunks_by_file are given)
# ==================================================

def scan_project(
    project_dir: Path,
    previous_manifest=None,
    previous_chunks_by_file=None,
):
    """
    previous_manifest is None -> full scan, every file is (re)chunked.

    previous_manifest is a dict {file_path: content_hash} from a prior run
    -> incremental scan: a file whose current hash matches its previous
    hash is skipped and its chunks are reused from previous_chunks_by_file
    instead of being re-chunked. Everything else is (re)chunked.

    Returns (all_chunks, manifest, deleted_files):
      - all_chunks: chunks for every file currently present in the project
      - manifest: {file_path: content_hash} reflecting the current scan
      - deleted_files: file paths present in previous_manifest but no
        longer found on disk (empty list on a full scan)
    """

    incremental = previous_manifest is not None

    gitignore_specs = load_gitignore_specs(project_dir)

    manifest = {}
    seen_files = set()
    all_chunks = []

    for path in project_dir.rglob("*"):

        if not path.is_file():
            continue

        if should_ignore(path, gitignore_specs):
            continue

        file_type = get_file_type(path)

        if file_type is None:
            continue

        file_key = str(path)
        seen_files.add(file_key)

        file_hash = hash_file(path)
        manifest[file_key] = file_hash

        if incremental and previous_manifest.get(file_key) == file_hash:

            all_chunks.extend(
                previous_chunks_by_file.get(file_key, [])
            )
            continue

        print(f"Processing: {path}")

        chunks = chunk_file(path, file_type)

        for index, chunk in enumerate(chunks):
            chunk["chunk_index"] = index

        all_chunks.extend(chunks)

    deleted_files = []

    if incremental:
        deleted_files = [
            file_key for file_key in previous_manifest
            if file_key not in seen_files
        ]

    return all_chunks, manifest, deleted_files
