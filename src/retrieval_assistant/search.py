"""Query both domains.

The query is embedded separately per domain (with that domain's model and
prefix) and searched against that domain's collection.

Why results are returned *grouped by domain* rather than merged into one ranked
list: cosine scores from two different models are not comparable. In practice
the code model (st-codesearch) hands out systematically higher scores (~0.5-0.66)
than bge (~0.34-0.39) even for equally relevant hits, so a single raw-score
merge lets the code collection crowd out prose every time — e.g. a query whose
real answer is a prose note never surfaces. Grouping keeps each model's ranking
on its own scale and surfaces the best of *both* domains. ``search_merged`` is
kept for callers that still want the (caveated) flat list.
"""

from __future__ import annotations

from .config import Settings
from .embedding import Embedder
from .store import MilvusStore, SearchHit


def search_grouped(
    query: str, settings: Settings, top_k: int | None = None
) -> list[tuple[str, list[SearchHit]]]:
    """Return ``[(domain_key, hits), ...]`` — top-k per domain, on each own scale."""
    k = top_k if top_k is not None else settings.top_k

    grouped: list[tuple[str, list[SearchHit]]] = []
    for domain in settings.domains():
        store = MilvusStore(settings.db_path, domain.collection, domain.embedding_dim)
        embedder = Embedder(domain)
        query_vector = embedder.embed_query(query)
        hits = store.search(query_vector, top_k=k, source=domain.key)
        hits.sort(key=lambda h: h.score, reverse=True)  # highest cosine first
        grouped.append((domain.key, hits))
    return grouped


def search_merged(query: str, settings: Settings, top_k: int | None = None) -> list[SearchHit]:
    """Flat list across domains, sorted by raw score. Cross-model scores are not
    strictly comparable — prefer :func:`search_grouped` for display."""
    k = top_k if top_k is not None else settings.top_k
    hits: list[SearchHit] = []
    for _, domain_hits in search_grouped(query, settings, top_k=k):
        hits.extend(domain_hits)
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]
