from langchain_core.documents import Document


def chunks_to_documents(chunks):
    
    documents = []

    for index, chunk in enumerate(chunks):

        metadata = {
            "chunk_id": f"{chunk['file']}:{index}",
            "file": chunk["file"],
            "type": chunk["type"],
            "name": chunk["name"],
            "class": chunk["class"] or "",
            "start_line": chunk.get("start_line") or 0,
            "end_line": chunk.get("end_line") or 0,
            "chunk_index": chunk.get(
                "chunk_index",
                index,
            ),
        }

        # Prefix the file path so the embedding itself is aware
        # of where the chunk lives (directory, language, etc.),
        # not just the metadata attached to it.
        page_content = f"File: {chunk['file']}\n\n{chunk['code']}"

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata,
            )
        )

    return documents