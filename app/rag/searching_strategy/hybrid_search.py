import json

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from store import vector_store, CHUNKS_FILE
from chunking_strategy.chunk_documents import chunks_to_documents


RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates the hybrid retriever pulls before reranking narrows
# them down to top_k. Reranking only helps if it has more than top_k
# candidates to actually choose between.
CANDIDATE_POOL_SIZE = 10


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

cross_encoder = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)


# ==================================================
# HYBRID SEARCH (retrieve wide, rerank down)
# ==================================================

def hybrid_search(
    question,
    top_k=3,
):

    pool_size = max(top_k, CANDIDATE_POOL_SIZE)

    bm25_retriever.k = pool_size
    vector_retriever.search_kwargs["k"] = pool_size

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3],
        id_key="chunk_id",
    )

    reranker = CrossEncoderReranker(
        model=cross_encoder,
        top_n=top_k,
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever,
    )

    return compression_retriever.invoke(question)
