from pathlib import Path

import pathspec

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
# SCAN PROJECT
# ==================================================

def scan_project(project_dir: Path):

    gitignore_specs = load_gitignore_specs(project_dir)

    all_chunks = []

    for path in project_dir.rglob("*"):

        if not path.is_file():
            continue

        if should_ignore(path, gitignore_specs):
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
