import json

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from store import vector_store, CHUNKS_FILE
from chunking_strategy.chunk_documents import chunks_to_documents


# ==================================================
# LOAD CHUNKS
# ==================================================

with open(
    CHUNKS_FILE,
    encoding="utf-8",
) as f:

    chunks = json.load(f)

documents = chunks_to_documents(chunks)


# ==================================================
# RETRIEVERS
# ==================================================

bm25_retriever = BM25Retriever.from_documents(documents)

vector_retriever = vector_store.as_retriever()


# ==================================================
# HYBRID SEARCH
# ==================================================

def hybrid_search(
    question,
    top_k=3,
):

    bm25_retriever.k = top_k
    vector_retriever.search_kwargs["k"] = top_k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3],
        id_key="chunk_id",
    )

    return ensemble_retriever.invoke(question)[:top_k]
