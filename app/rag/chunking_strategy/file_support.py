from pathlib import Path
import ast


SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".txt",
}

SPECIAL_FILES = {
    "Dockerfile",
}

IGNORED_FILES = {
    "package-lock.json",
    "npm-shrinkwrap.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "Cargo.lock",
    "Gemfile.lock",
    "composer.lock",
}

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "chroma_db",
}

# ==================================================
# FILE TYPE
# ==================================================

def get_file_type(path: Path):

    if path.name in IGNORED_FILES:
        return None

    if path.name in SPECIAL_FILES:
        return "config"

    if path.suffix == ".py":
        return "python"

    if path.suffix in {
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
    }:
        return "javascript"

    if path.suffix in {
        ".html",
        ".css",
        ".scss",
    }:
        return "frontend"

    if path.suffix in {
        ".json",
        ".yaml",
        ".yml",
        ".toml",
    }:
        return "config"

    if path.suffix == ".md":
        return "markdown"

    if path.suffix == ".txt":
        return "text"

    return None