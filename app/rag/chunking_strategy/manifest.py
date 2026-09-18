import hashlib
import json
from pathlib import Path

# ==================================================
# HASH A FILE'S CONTENT
# ==================================================

def hash_file(path: Path) -> str:

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

# ==================================================
# LOAD / SAVE THE INDEX MANIFEST
# ==================================================

def load_manifest(manifest_path: Path) -> dict:

    if not manifest_path.exists():
        return {}

    return json.loads(
        manifest_path.read_text(encoding="utf-8")
    )


def save_manifest(manifest_path: Path, manifest: dict):

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
