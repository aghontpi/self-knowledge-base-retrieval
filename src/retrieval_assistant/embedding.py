"""Local sentence-transformers embedder, one per domain.

Each :class:`~retrieval_assistant.config.DomainConfig` carries its own model and
query prefix. bge (prose) is asymmetric — the *query* gets an instruction
prefix, documents do not. The code model is symmetric — its ``query_prefix`` is
empty, so both sides are embedded identically. Both models L2-normalize so the
COSINE index computes a true cosine similarity.

``SentenceTransformer`` is imported lazily so a model is only downloaded /
loaded when embeddings are actually needed.
"""

from __future__ import annotations

import numpy as np

from .config import DomainConfig


class Embedder:
    def __init__(self, domain: DomainConfig):
        self._domain = domain
        self._model = None  # loaded on first use

    @property
    def model_name(self) -> str:
        return self._domain.embedding_model

    @property
    def dim(self) -> int:
        return self._domain.embedding_dim

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self._domain.embedding_model)
            # Cap the input length. Long-context models (e.g. Qwen3, 32k) will
            # otherwise try to build an attention buffer for the full sequence on
            # an oversized chunk and blow up memory ("Invalid buffer size").
            # Truncating to a sane window keeps memory bounded; chunks should be
            # well under this anyway. The tokenizer truncates past this length.
            if self._domain.max_seq_length:
                model.max_seq_length = self._domain.max_seq_length
            self._model = model
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages (no prefix). Returns an (n, dim) float32 array."""
        model = self._ensure_model()
        return np.asarray(
            model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 64,
            ),
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query (domain prefix applied). Returns a (dim,) array."""
        model = self._ensure_model()
        prefixed = f"{self._domain.query_prefix}{text}" if self._domain.query_prefix else text
        vec = model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vec, dtype=np.float32)
