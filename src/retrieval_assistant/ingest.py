"""Convergent ingestion across both domains: make each collection match the
files routed to it, exactly.

Discovery (:mod:`discovery`) yields the files worth indexing, each tagged with a
domain (prose/code). For each domain we diff the filesystem against the hashes
already stored in that domain's collection and reconcile:

* **new**       on disk, not indexed        -> parse, chunk, embed, insert
* **changed**   hash differs from stored    -> delete-by-doc_id, then re-insert
* **removed**   indexed, no longer routed   -> delete-by-doc_id
* **unchanged**                             -> skip

Re-running converges each collection to the current corpus — the whole answer to
"how do I update the knowledge base?" is: run it again.

Each row records its embedding model. Mixing models in one collection is
meaningless, so a stored/configured mismatch stops ingestion and asks for a
rebuild of that collection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_for_kind
from .config import DomainConfig, Settings
from .discovery import Discovered, discover_files
from .embedding import Embedder
from .parsing import parse_file
from .store import MilvusStore, chunk_pk


@dataclass
class DomainReport:
    domain: str
    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    chunks_written: int = 0

    def summary(self) -> str:
        return (
            f"[{self.domain}] new={len(self.new)} changed={len(self.changed)} "
            f"removed={len(self.removed)} unchanged={len(self.unchanged)} "
            f"chunks_written={self.chunks_written}"
        )


class EmbeddingModelMismatch(RuntimeError):
    """Raised when a collection was built with a different embedding model."""


def _sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _index_document(
    item: Discovered,
    file_hash: str,
    settings: Settings,
    domain: DomainConfig,
    embedder: Embedder,
    store: MilvusStore,
) -> int:
    blocks = parse_file(item.path, item.route.kind)
    chunks = chunk_for_kind(blocks, item.route.kind, settings, suffix=item.path.suffix)
    if not chunks:
        return 0

    vectors = embedder.embed_documents([c.text for c in chunks])
    rows = [
        {
            "pk": chunk_pk(item.doc_id, c.chunk_index),
            "doc_id": item.doc_id,
            "file_hash": file_hash,
            "chunk_index": c.chunk_index,
            "locator": c.locator[:255],
            "text": c.text[:65535],
            "embedding_model": domain.embedding_model,
            "vector": vectors[i].tolist(),
        }
        for i, c in enumerate(chunks)
    ]
    store.upsert(rows)
    return len(rows)


def _ingest_domain(
    domain: DomainConfig,
    items: list[Discovered],
    settings: Settings,
) -> DomainReport:
    store = MilvusStore(settings.db_path, domain.collection, domain.embedding_dim)
    store.ensure_collection()
    stored = store.existing_doc_state()

    for meta in stored.values():
        if meta["embedding_model"] != domain.embedding_model:
            raise EmbeddingModelMismatch(
                f"Collection {domain.collection!r} was built with "
                f"{meta['embedding_model']!r} but configured model is "
                f"{domain.embedding_model!r}. Changing the embedding model "
                f"requires rebuilding this collection (delete {settings.db_path} "
                f"or drop the collection, then re-ingest)."
            )

    corpus = {item.doc_id: item for item in items}
    report = DomainReport(domain=domain.key)
    fs_ids = set(corpus)
    stored_ids = set(stored)

    for doc_id in sorted(stored_ids - fs_ids):
        store.delete_by_doc_id(doc_id)
        report.removed.append(doc_id)

    embedder = Embedder(domain)
    for doc_id in sorted(fs_ids):
        item = corpus[doc_id]
        file_hash = _sha256_bytes(item.path)
        prior = stored.get(doc_id)
        if prior is None:
            report.chunks_written += _index_document(
                item, file_hash, settings, domain, embedder, store
            )
            report.new.append(doc_id)
        elif prior["file_hash"] != file_hash:
            store.delete_by_doc_id(doc_id)
            report.chunks_written += _index_document(
                item, file_hash, settings, domain, embedder, store
            )
            report.changed.append(doc_id)
        else:
            report.unchanged.append(doc_id)

    return report


def ingest(settings: Settings) -> list[DomainReport]:
    discovered = discover_files(settings)
    by_domain: dict[str, list[Discovered]] = {d.key: [] for d in settings.domains()}
    for item in discovered:
        by_domain.setdefault(item.route.domain, []).append(item)

    reports: list[DomainReport] = []
    for domain in settings.domains():
        reports.append(_ingest_domain(domain, by_domain.get(domain.key, []), settings))
    return reports
