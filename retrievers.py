"""Pluggable retrievers for the rag_injection suite.

Two implementations, same interface — `retrieve(query, knowledge_base, top_k)
-> list[dict]` — so the eval logic in evals/rag_injection.py never needs to
know which one is in use:

  - keyword: dependency-free term-overlap retriever (see evals/rag_injection.py).
    Good for CI, good for proving the vulnerability exists with zero setup.
  - embedding: sentence-transformers + FAISS cosine-similarity retriever.
    What an actual production RAG pipeline looks like. Same poisoned docs,
    same attacks — this is here to answer "sure, but does this happen with
    *real* semantic retrieval, not just keyword matching?"

Set RAG_RETRIEVER=embedding in .env (or pass retriever="embedding" to
evals.rag_injection.run) to switch. Falls back to keyword automatically if
sentence-transformers/faiss aren't installed, so the base toolkit still
runs with zero heavy dependencies.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "i", "you", "your", "it",
    "its", "my", "me", "to", "of", "in", "on", "for", "with", "and", "or",
    "not", "no", "can", "could", "would", "will", "what", "how", "get",
    "s", "t", "this", "that", "about",
}

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = None
_faiss_index_cache: dict[int, tuple] = {}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def keyword_retrieve(query: str, knowledge_base: list[dict], top_k: int = 3) -> list[dict]:
    """Dependency-free keyword-overlap retriever (see evals/rag_injection.py docstring)."""
    query_tokens = _tokenize(query)
    scored = []
    for doc in knowledge_base:
        doc_tokens = _tokenize(doc["title"] + " " + doc["text"])
        overlap = len(query_tokens & doc_tokens)
        scored.append((overlap, doc))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for score, doc in scored[:top_k] if score > 0] or [scored[0][1]]


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s (first call only)...", _EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_faiss_index(knowledge_base: list[dict]):
    """Build (or reuse) a FAISS cosine-similarity index over the knowledge base."""
    import numpy as np
    import faiss

    cache_key = id(knowledge_base)
    if cache_key in _faiss_index_cache:
        return _faiss_index_cache[cache_key]

    model = _get_embedding_model()
    texts = [doc["title"] + ". " + doc["text"] for doc in knowledge_base]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product == cosine (normalized)
    index.add(embeddings)

    _faiss_index_cache[cache_key] = (index, model)
    return index, model


def embedding_retrieve(query: str, knowledge_base: list[dict], top_k: int = 3) -> list[dict]:
    """Real semantic retriever: sentence-transformer embeddings + FAISS cosine search."""
    import numpy as np

    index, model = _get_faiss_index(knowledge_base)
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.asarray(query_vec, dtype="float32")

    k = min(top_k, len(knowledge_base))
    scores, indices = index.search(query_vec, k)
    return [knowledge_base[i] for i in indices[0] if i != -1]


def get_retriever(name: str | None = None):
    """Return a retrieve(query, knowledge_base, top_k) callable by name.

    name: "keyword", "embedding", or None (reads RAG_RETRIEVER env var,
    defaults to "keyword"). Silently falls back to keyword if the embedding
    stack isn't installed.
    """
    name = name or os.getenv("RAG_RETRIEVER", "keyword")
    if name == "embedding":
        try:
            import faiss  # noqa: F401
            import sentence_transformers  # noqa: F401
            return embedding_retrieve
        except ImportError:
            logger.warning(
                "RAG_RETRIEVER=embedding requested but sentence-transformers/faiss "
                "aren't installed (pip install sentence-transformers faiss-cpu). "
                "Falling back to keyword retriever."
            )
            return keyword_retrieve
    return keyword_retrieve
