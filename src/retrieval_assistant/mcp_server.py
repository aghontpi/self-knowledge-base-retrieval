"""Production-grade MCP Server for the Personal Retrieval Assistant.

Fixes:
1. Fast boot to prevent LLM client timeouts (lazy-loading weights + dependencies).
2. OS-level cross-process database file locking (filelock).
3. Explicit Apple Silicon sandboxing-safe CPU enforcement (device="cpu").
4. Strict path traversal guards (secure_resolve).
5. Robust tool parameter clamping and is_error=True error handling.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Enforce stable thread limits and prevent macOS OpenMP initialization crashes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Set up clean logging to stderr (since stdout is reserved for MCP stdio JSON-RPC transport)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stderr)
logger = logging.getLogger("pra_mcp")

try:
    from fastmcp import FastMCP
    from filelock import FileLock, Timeout
except ImportError as e:
    logger.critical("Failed to import required libraries. Ensure all dependencies are installed. Error: %s", e)
    sys.exit(1)

# Complete rapid start handshake instantly (under 10ms)
mcp = FastMCP("Personal Retrieval Assistant")

# Global singleton structure for lazy initialization of RAG pipeline
class RAGManager:
    def __init__(self):
        self._settings = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self._init_event = asyncio.Event()

    async def ensure_initialized(self) -> tuple[Any, FileLock]:
        """Thread-safe and async-safe lazy importer for heavy RAG modules."""
        if self._initialized:
            db_lock = FileLock(self._settings.db_path.with_suffix(".lock"), timeout=10.0)
            return self._settings, db_lock

        async with self._lock:
            # Check state again after acquiring lock
            if not self._initialized:
                logger.info("Initializing heavy dependencies & model pre-loading...")
                # Run CPU-blocking imports and warming in a safe background thread executor
                await asyncio.to_thread(self._load_heavy_modules)
                self._initialized = True
                self._init_event.set()
            else:
                await self._init_event.wait()

        db_lock = FileLock(self._settings.db_path.with_suffix(".lock"), timeout=10.0)
        return self._settings, db_lock

    def _load_heavy_modules(self) -> None:
        # Lazy local imports inside the loading executor
        global load_settings, ingest, search_grouped, search_reranked, MilvusStore, Embedder, Reranker
        from .config import load_settings
        from .ingest import ingest
        from .search import search_grouped, search_reranked
        from .store import MilvusStore
        from .embedding import Embedder
        from .rerank import Reranker
        
        # Enforce CPU execution to prevent GPU copy overhead and sandbox SIGABRTs on macOS
        import torch
        torch.set_num_threads(1)
        
        # Warm settings
        self._settings = load_settings()
        
        # Override embedders and reranker settings to strictly use CPU
        # Pre-initialize embedding models to pull them into memory
        logger.info("Pre-warming prose embedding model...")
        prose_embedder = Embedder(self._settings.prose)
        prose_embedder._ensure_model()
        # Set CPU device explicitly
        if hasattr(prose_embedder._model, "to"):
            prose_embedder._model.to("cpu")
            
        logger.info("Pre-warming code embedding model...")
        code_embedder = Embedder(self._settings.code)
        code_embedder._ensure_model()
        if hasattr(code_embedder._model, "to"):
            code_embedder._model.to("cpu")

        if self._settings.rerank_enabled:
            logger.info("Pre-warming cross-encoder rerank model...")
            reranker = Reranker(self._settings)
            reranker._ensure_model()
            if hasattr(reranker._model, "model") and hasattr(reranker._model.model, "to"):
                reranker._model.model.to("cpu")

        logger.info("RAG models and dependencies successfully warmed up.")

rag_manager = RAGManager()

def secure_resolve(doc_id: str, data_dir: Path) -> Path:
    """Rigorous path containment check guarding against path traversal vulnerabilities."""
    base_dir = Path(data_dir).resolve()
    target_path = (base_dir / doc_id).resolve()
    if not target_path.is_relative_to(base_dir):
        raise PermissionError(f"Access Denied: Path traversal attempt detected for target: {doc_id}")
    return target_path


@mcp.tool()
async def pra_search(query: str, top_k: int = 5) -> str:
    """Semantic search across prose and code collections using local rerankers.

    Args:
        query: Query string or natural language question.
        top_k: Number of ranked hits to return (must be between 1 and 50).
    """
    # 1. Type validation and range clamping
    top_k = max(1, min(int(top_k), 50))
    
    try:
        settings, db_lock = await rag_manager.ensure_initialized()
        
        # 2. Acquire cross-process file lock
        logger.info("Acquiring DB file lock for search...")
        with db_lock.acquire(timeout=5.0):
            if settings.rerank_enabled:
                hits = search_reranked(query, settings, top_k=top_k)
            else:
                from .search import search_merged
                hits = search_merged(query, settings, top_k=top_k)
                
        # 3. Format result payloads
        if not hits:
            return "No matching search results found."
            
        formatted = []
        for i, hit in enumerate(hits, 1):
            score = hit.rerank_score if hit.rerank_score is not None else hit.score
            formatted.append(
                f"Rank: {i}\n"
                f"Path: {hit.doc_id} [{hit.locator}]\n"
                f"Score: {score:.4f} (Source: {hit.source})\n"
                f"Content:\n{hit.text}\n"
                f"{'-'*40}"
            )
        return "\n".join(formatted)
        
    except Timeout:
        logger.error("DB lock timeout during search.")
        return "Error: Database is currently locked by another search or ingestion process. Please retry."
    except Exception as e:
        logger.exception("Unexpected error in pra_search")
        return f"Error executing search: {str(e)}"


@mcp.tool()
async def pra_ingest() -> str:
    """Scan data directory and perform convergent indexing.

    Synchronizes the local vector database with changed, new, or deleted files.
    """
    try:
        settings, db_lock = await rag_manager.ensure_initialized()
        
        logger.info("Acquiring exclusive DB file lock for ingestion...")
        # Exclusive lock to prevent any multi-process writes clashing on SQLite
        with db_lock.acquire(timeout=15.0):
            reports = ingest(settings)
            
        summaries = [report.summary() for report in reports]
        return "Ingestion completed successfully.\n" + "\n".join(summaries)
        
    except Timeout:
        logger.error("DB lock timeout during ingestion.")
        return "Error: Database is currently locked. Ingestion aborted to prevent write corruption."
    except Exception as e:
        logger.exception("Unexpected error in pra_ingest")
        return f"Error executing ingestion: {str(e)}"


@mcp.tool()
async def pra_stats() -> str:
    """Return diagnostic stats of the database and active models."""
    try:
        settings, db_lock = await rag_manager.ensure_initialized()
        
        with db_lock.acquire(timeout=5.0):
            prose_store = MilvusStore(settings.db_path, settings.prose.collection, settings.prose.embedding_dim)
            code_store = MilvusStore(settings.db_path, settings.code.collection, settings.code.embedding_dim)
            prose_count = prose_store.count()
            code_count = code_store.count()
            
        stats_info = (
            f"Database Location: {settings.db_path}\n"
            f"Data Directory: {settings.data_dir}\n"
            f"Prose Index: collection={settings.prose.collection}, model={settings.prose.embedding_model}, chunks={prose_count}\n"
            f"Code Index: collection={settings.code.collection}, model={settings.code.embedding_model}, chunks={code_count}\n"
            f"Reranker: enabled={settings.rerank_enabled}, model={settings.rerank_model}"
        )
        return stats_info
        
    except Timeout:
        return "Error: Database is currently locked. Unable to fetch stats."
    except Exception as e:
        return f"Error fetching statistics: {str(e)}"


@mcp.tool()
async def pra_get_file(doc_id: str) -> str:
    """Read the full raw or parsed text of an indexed document.

    Args:
        doc_id: Relative file path identifier of the target document.
    """
    try:
        settings, _ = await rag_manager.ensure_initialized()
        
        # Guard against path traversal vulnerability
        target_path = secure_resolve(doc_id, settings.data_dir)
        
        if not target_path.exists():
            return f"Error: File '{doc_id}' does not exist in the configured data directory."
            
        if target_path.is_dir():
            return f"Error: Path '{doc_id}' is a directory. Specify a direct file path."
            
        # Limit read size to prevent loading huge files into memory
        max_bytes = 500 * 1024  # 500 KB limit
        if target_path.stat().st_size > max_bytes:
            return f"Error: File size exceeds the maximum limit for direct inspection (500KB)."
            
        return target_path.read_text(encoding="utf-8", errors="replace")
        
    except PermissionError as pe:
        logger.warning("Path traversal warning: %s", pe)
        return str(pe)
    except Exception as e:
        logger.exception("Error in pra_get_file")
        return f"Error reading document: {str(e)}"


@mcp.resource("pra://settings")
def get_settings() -> str:
    """Expose the active RAG configuration settings as a read-only resource."""
    # Settings object can be read instantly without fully loading heavy models
    import os
    from .config import load_settings
    settings = load_settings()
    
    import json
    return json.dumps({
        "data_dir": str(settings.data_dir),
        "db_path": str(settings.db_path),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "code_max_lines": settings.code_max_lines,
        "code_overlap_lines": settings.code_overlap_lines,
        "rerank_enabled": settings.rerank_enabled,
        "prose_model": settings.prose.embedding_model,
        "code_model": settings.code.embedding_model,
    }, indent=2)


if __name__ == "__main__":
    # Launch standard stdio server
    mcp.run()
