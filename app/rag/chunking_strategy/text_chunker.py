from pathlib import Path

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        " ",
        "",
    ],
)


def chunk_text_file(path: Path):

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    chunks = splitter.split_text(text)

    results = []

    for index, chunk in enumerate(chunks):

        results.append({
            "code": chunk,
            "file": str(path),
            "type": "text",
            "name": path.name,
            "class": None,
            "chunk_index": index,
        })

    return results