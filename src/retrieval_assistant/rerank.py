"""Cross-encoder reranking.

A bi-encoder (the embedders) maps query and document to vectors *independently*,
then compares them — fast, but it never sees the pair together. A *cross-encoder*
reads ``(query, chunk)`` jointly and outputs a single relevance score, which is
more accurate and — crucially here — puts every candidate on **one consistent
scale** regardless of which domain model retrieved it. That fixes two problems
seen on the real corpus: code-model scores aren't comparable to bge scores, and
the code model over-rewards very short snippets.

Used as a second stage: retrieve a wide net per domain (cheap, approximate),
then rerank the merged candidates and keep the best. ``CrossEncoder`` is
imported lazily so the model only loads when reranking actually runs.
"""

from __future__ import annotations

from dataclasses import replace

from .config import Settings
from .store import SearchHit


class Reranker:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None

    @property
    def model_name(self) -> str:
        return self._settings.rerank_model

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._settings.rerank_model)
        return self._model

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        """Return ``hits`` re-scored by the cross-encoder, best first, capped at top_k."""
        if not hits:
            return []
        model = self._ensure_model()
        pairs = [(query, hit.text) for hit in hits]
        scores = model.predict(pairs)
        rescored = [replace(hit, rerank_score=float(s)) for hit, s in zip(hits, scores)]
        rescored.sort(key=lambda h: h.rerank_score, reverse=True)
        return rescored[:top_k]
