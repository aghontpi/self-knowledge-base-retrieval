"""Environment-backed configuration.

The corpus is heterogeneous (prose + code), so the index is split into two
*domains*, each with its own Milvus collection and embedding model:

* **prose** — markdown, text, PDF, docx, CSV/config -> ``bge-small`` (English)
* **code**  — source files & notebooks -> a code-trained embedding model

Vectors from different models are not comparable, so they must live in separate
collections. :mod:`retrieval_assistant.ingest` routes each file to exactly one
domain; :mod:`retrieval_assistant.search` queries both and merges.

All paths default to locations relative to the repo root. A ``.env`` file at the
repo root is loaded automatically if present (it is gitignored).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional convenience; absence must not break anything
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

# repo root = .../personal-retrieval-assistant
#   this file: <root>/src/retrieval_assistant/config.py
REPO_ROOT = Path(__file__).resolve().parents[2]

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")

# bge (prose) expects this instruction prefix on the *query* side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Qwen3-Embedding (code) is also asymmetric: an instruction on the query side,
# nothing on the document side. This is the model's own recommended prompt.
QWEN3_QUERY_PREFIX = (
    "Instruct: Given a question, retrieve code or documentation that answers it\nQuery:"
)


def _resolve(path_str: str) -> Path:
    """Resolve a config path against the repo root unless it is absolute."""
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


@dataclass(frozen=True)
class DomainConfig:
    """One content domain: its collection, embedding model, and query prefix."""

    key: str
    collection: str
    embedding_model: str
    embedding_dim: int
    query_prefix: str = ""
    max_seq_length: int = 512  # cap input tokens; guards long-context models from OOM
    encode_batch_size: int = 16  # chunks per forward pass; lower = less peak memory


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: _resolve(os.getenv("PRA_DATA_DIR", "data")))
    db_path: Path = field(default_factory=lambda: _resolve(os.getenv("PRA_DB_PATH", "pra.db")))

    prose: DomainConfig = field(
        default_factory=lambda: DomainConfig(
            key="prose",
            collection=os.getenv("PRA_PROSE_COLLECTION", "prose"),
            embedding_model=os.getenv("PRA_PROSE_MODEL", "BAAI/bge-small-en-v1.5"),
            embedding_dim=int(os.getenv("PRA_PROSE_DIM", "384")),
            query_prefix=BGE_QUERY_PREFIX,
        )
    )
    code: DomainConfig = field(
        default_factory=lambda: DomainConfig(
            key="code",
            collection=os.getenv("PRA_CODE_COLLECTION", "code"),
            # Default: Qwen3-Embedding-0.6B (June 2025, Qwen3 base, dim 1024,
            # sentence-transformers native, asymmetric query instruction).
            # Heavier on RAM (a small encode_batch_size keeps peak ~3-4GB on a
            # 16GB machine). Lighter alternative: set PRA_CODE_MODEL to
            # flax-sentence-embeddings/st-codesearch-distilroberta-base with
            # PRA_CODE_DIM=768 and PRA_CODE_PREFIX="".
            embedding_model=os.getenv("PRA_CODE_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
            embedding_dim=int(os.getenv("PRA_CODE_DIM", "1024")),
            query_prefix=os.getenv("PRA_CODE_PREFIX", QWEN3_QUERY_PREFIX),
            encode_batch_size=int(os.getenv("PRA_CODE_BATCH", "8")),
        )
    )

    # Prose chunking (character window + overlap).
    chunk_size: int = field(default_factory=lambda: int(os.getenv("PRA_CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("PRA_CHUNK_OVERLAP", "150")))

    # Code chunking (line window + overlap, used as fallback / for non-Python).
    code_max_lines: int = field(default_factory=lambda: int(os.getenv("PRA_CODE_MAX_LINES", "80")))
    code_overlap_lines: int = field(
        default_factory=lambda: int(os.getenv("PRA_CODE_OVERLAP_LINES", "10"))
    )

    # File-selection guards.
    max_file_bytes: int = field(
        default_factory=lambda: int(float(os.getenv("PRA_MAX_FILE_MB", "1")) * 1024 * 1024)
    )
    use_git: bool = field(
        default_factory=lambda: os.getenv("PRA_USE_GIT", "1").lower() not in ("0", "false", "no")
    )

    top_k: int = field(default_factory=lambda: int(os.getenv("PRA_TOP_K", "5")))

    # Reranking. A cross-encoder re-scores (query, chunk) pairs on one consistent
    # scale, which fixes cross-model score incomparability and under-weighting of
    # short snippets. Candidates are pulled per domain, then merged and reranked.
    rerank_enabled: bool = field(
        default_factory=lambda: os.getenv("PRA_RERANK", "1").lower() not in ("0", "false", "no")
    )
    rerank_model: str = field(
        default_factory=lambda: os.getenv("PRA_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    rerank_candidates: int = field(
        default_factory=lambda: int(os.getenv("PRA_RERANK_CANDIDATES", "20"))
    )

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})."
            )
        if self.code_overlap_lines >= self.code_max_lines:
            raise ValueError("code_overlap_lines must be smaller than code_max_lines")

    def domains(self) -> list[DomainConfig]:
        return [self.prose, self.code]

    def domain(self, key: str) -> DomainConfig:
        for d in self.domains():
            if d.key == key:
                return d
        raise KeyError(key)


def load_settings() -> Settings:
    """Build a fresh Settings snapshot from the current environment."""
    return Settings()
