"""Milvus Lite wrapper — one instance per collection.

A collection holds one row per chunk. Every row carries the metadata that makes
ingestion *convergent* rather than append-only:

* ``pk``              deterministic primary key = ``sha1(f"{doc_id}::{chunk_index}")``
* ``doc_id``          source path relative to the data dir
* ``file_hash``       sha256 of the source file's bytes
* ``chunk_index``     global chunk ordinal within the document
* ``locator``         human-readable source location (e.g. ``p.3``, ``L1-40``)
* ``text``            the chunk text
* ``embedding_model`` model used to produce the vector
* ``vector``          normalized float embedding

The deterministic primary key lets ``upsert`` overwrite a chunk in place;
``delete_by_doc_id`` removes every chunk of a document. COSINE matches the
L2-normalized embeddings; AUTOINDEX is appropriate for Milvus Lite.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

_TEXT_MAX_LEN = 65535
_DOC_ID_MAX_LEN = 1024
_LOCATOR_MAX_LEN = 256
_MODEL_MAX_LEN = 256


def chunk_pk(doc_id: str, chunk_index: int) -> str:
    """Deterministic primary key for a chunk (40-char sha1 hex)."""
    return hashlib.sha1(f"{doc_id}::{chunk_index}".encode("utf-8")).hexdigest()


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    locator: str
    text: str
    score: float
    chunk_index: int
    source: str = ""  # which domain/collection produced the hit


class MilvusStore:
    OUTPUT_FIELDS = ("doc_id", "file_hash", "chunk_index", "locator", "text", "embedding_model")

    def __init__(self, db_path: Path, collection: str, dim: int):
        self._db_path = db_path
        self._collection = collection
        self._dim = dim
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from pymilvus import MilvusClient

            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._client = MilvusClient(uri=str(self._db_path))
        return self._client

    def ensure_collection(self) -> None:
        from pymilvus import DataType

        client = self._ensure_client()
        if client.has_collection(self._collection):
            return

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=40)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=_DOC_ID_MAX_LEN)
        schema.add_field("file_hash", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("locator", DataType.VARCHAR, max_length=_LOCATOR_MAX_LEN)
        schema.add_field("text", DataType.VARCHAR, max_length=_TEXT_MAX_LEN)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=_MODEL_MAX_LEN)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_collection(
            collection_name=self._collection, schema=schema, index_params=index_params
        )

    def upsert(self, rows: list[dict]) -> None:
        if not rows:
            return
        client = self._ensure_client()
        client.upsert(collection_name=self._collection, data=rows)

    def delete_by_doc_id(self, doc_id: str) -> None:
        client = self._ensure_client()
        client.delete(collection_name=self._collection, filter=f'doc_id == "{_escape(doc_id)}"')

    def existing_doc_state(self) -> dict[str, dict]:
        """Map ``doc_id`` -> {"file_hash", "embedding_model"}; empty if no collection."""
        client = self._ensure_client()
        if not client.has_collection(self._collection):
            return {}
        client.load_collection(self._collection)

        state: dict[str, dict] = {}
        iterator = client.query_iterator(
            collection_name=self._collection,
            filter="chunk_index >= 0",
            output_fields=["doc_id", "file_hash", "embedding_model"],
            batch_size=1000,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    state[row["doc_id"]] = {
                        "file_hash": row["file_hash"],
                        "embedding_model": row["embedding_model"],
                    }
        finally:
            iterator.close()
        return state

    def search(self, query_vector, top_k: int, source: str = "") -> list[SearchHit]:
        client = self._ensure_client()
        if not client.has_collection(self._collection):
            return []
        client.load_collection(self._collection)
        results = client.search(
            collection_name=self._collection,
            data=[list(map(float, query_vector))],
            limit=top_k,
            output_fields=list(self.OUTPUT_FIELDS),
            search_params={"metric_type": "COSINE"},
        )
        hits: list[SearchHit] = []
        for hit in results[0]:
            entity = hit.get("entity", hit)
            hits.append(
                SearchHit(
                    doc_id=entity.get("doc_id", ""),
                    locator=entity.get("locator", ""),
                    text=entity.get("text", ""),
                    score=float(hit.get("distance", 0.0)),
                    chunk_index=int(entity.get("chunk_index", 0)),
                    source=source,
                )
            )
        return hits

    def count(self) -> int:
        client = self._ensure_client()
        if not client.has_collection(self._collection):
            return 0
        client.load_collection(self._collection)
        res = client.query(
            collection_name=self._collection,
            filter="chunk_index >= 0",
            output_fields=["count(*)"],
        )
        return int(res[0]["count(*)"]) if res else 0
