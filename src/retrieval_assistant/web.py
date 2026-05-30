"""FastAPI web server for the Personal Retrieval Assistant.

Exposes REST APIs to perform semantic queries, trigger database sync/ingestion,
inspect system statistics, and safely retrieve original source files. Serves
the compiled React TypeScript SPA.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from filelock import FileLock, Timeout

# Thread capping and sandbox safety
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stderr)
logger = logging.getLogger("pra_web")

app = FastAPI(
    title="Personal Retrieval Assistant Web Console",
    description="Web control dashboard and query interface for the local vector RAG assistant.",
    version="0.1.0"
)

# Enable CORS for Vite dev server proxying
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singleton structure for lazy initialization of RAG pipeline
class WebRAGManager:
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
            if not self._initialized:
                logger.info("Initializing heavy deep learning models & packages...")
                await asyncio.to_thread(self._load_heavy_modules)
                self._initialized = True
                self._init_event.set()
            else:
                await self._init_event.wait()

        db_lock = FileLock(self._settings.db_path.with_suffix(".lock"), timeout=10.0)
        return self._settings, db_lock

    def _load_heavy_modules(self) -> None:
        global load_settings, ingest, search_grouped, search_merged, search_reranked, MilvusStore, Embedder, Reranker
        from .config import load_settings
        from .ingest import ingest
        from .search import search_grouped, search_merged, search_reranked
        from .store import MilvusStore
        from .embedding import Embedder
        from .rerank import Reranker
        
        import torch
        torch.set_num_threads(1)
        
        self._settings = load_settings()
        
        # Pre-warm embedding engines to keep them in memory
        logger.info("Pre-warming prose embedding model...")
        prose_embedder = Embedder(self._settings.prose)
        prose_embedder._ensure_model()
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

        logger.info("RAG models and dependencies successfully warmed in memory.")

rag_manager = WebRAGManager()

def secure_resolve(doc_id: str, data_dir: Path) -> Path:
    """Rigorous path containment check guarding against path traversal vulnerabilities."""
    base_dir = Path(data_dir).resolve()
    target_path = (base_dir / doc_id).resolve()
    if not target_path.is_relative_to(base_dir):
        raise PermissionError(f"Access Denied: Path traversal attempt detected: {doc_id}")
    return target_path


@app.get("/api/stats")
async def get_stats():
    """Fetch database metrics and collection sizes."""
    try:
        settings, db_lock = await rag_manager.ensure_initialized()
        
        with db_lock.acquire(timeout=5.0):
            prose_store = MilvusStore(settings.db_path, settings.prose.collection, settings.prose.embedding_dim)
            code_store = MilvusStore(settings.db_path, settings.code.collection, settings.code.embedding_dim)
            prose_count = prose_store.count()
            code_count = code_store.count()
            
        return {
            "db_path": str(settings.db_path),
            "data_dir": str(settings.data_dir),
            "prose": {
                "collection": settings.prose.collection,
                "model": settings.prose.embedding_model,
                "chunks": prose_count
            },
            "code": {
                "collection": settings.code.collection,
                "model": settings.code.embedding_model,
                "chunks": code_count
            },
            "reranker": {
                "enabled": settings.rerank_enabled,
                "model": settings.reranker_model if hasattr(settings, "reranker_model") else getattr(settings, "rerank_model", "")
            }
        }
    except Timeout:
        raise HTTPException(status_code=503, detail="Database is locked. Please try again.")
    except Exception as e:
        logger.exception("Error loading system metrics")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def run_search(
    query: str = Body(..., embed=True),
    top_k: int = Body(5, embed=True),
    use_rerank: bool = Body(True, embed=True),
    mode: str = Body("unified", embed=True)
):
    """Run a semantic query across prose and code domains."""
    top_k = max(1, min(int(top_k), 50))
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        settings, db_lock = await rag_manager.ensure_initialized()
        
        with db_lock.acquire(timeout=5.0):
            if mode == "grouped":
                # Returns [(domain_key, hits), ...]
                grouped_res = search_grouped(query, settings, top_k=top_k)
                res_dict = {
                    "hits": [],
                    "grouped": {
                        "prose": [],
                        "code": []
                    }
                }
                for domain_key, hits in grouped_res:
                    serialized_hits = [
                        {
                            "doc_id": h.doc_id,
                            "locator": h.locator,
                            "text": h.text,
                            "score": float(h.score),
                            "chunk_index": h.chunk_index,
                            "source": h.source,
                            "rerank_score": float(h.rerank_score) if h.rerank_score is not None else None
                        }
                        for h in hits
                    ]
                    if domain_key == "code":
                        res_dict["grouped"]["code"] = serialized_hits
                    else:
                        res_dict["grouped"]["prose"] = serialized_hits
                return res_dict
            else:
                # Unified ranked mode
                if settings.rerank_enabled and use_rerank:
                    hits = search_reranked(query, settings, top_k=top_k)
                else:
                    hits = search_merged(query, settings, top_k=top_k)
                    
                serialized_hits = [
                    {
                        "doc_id": h.doc_id,
                        "locator": h.locator,
                        "text": h.text,
                        "score": float(h.score),
                        "chunk_index": h.chunk_index,
                        "source": h.source,
                        "rerank_score": float(h.rerank_score) if h.rerank_score is not None else None
                    }
                    for h in hits
                ]
                return {"hits": serialized_hits}

    except Timeout:
        raise HTTPException(status_code=503, detail="Database is locked. Search aborted.")
    except Exception as e:
        logger.exception("Error executing vector search query")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest")
async def run_ingestion():
    """Trigger convergent filesystem indexing."""
    try:
        settings, db_lock = await rag_manager.ensure_initialized()
        
        logger.info("Acquiring database lock for web ingestion...")
        with db_lock.acquire(timeout=15.0):
            reports = ingest(settings)
            
        summaries = [report.summary() for report in reports]
        final_summary = " & ".join(summaries)
        
        details = []
        for r in reports:
            details.append(f"[{r.domain}] Sync complete. added={len(r.new)} changed={len(r.changed)} deleted={len(r.removed)} chunks={r.chunks_written}")
            
        return {
            "summary": f"Convergent sync successful: {final_summary}",
            "details": details
        }
    except Timeout:
        raise HTTPException(status_code=503, detail="Database file is locked by another index update task.")
    except Exception as e:
        logger.exception("Error running vector synchronization")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/file")
async def get_file(doc_id: str):
    """Safely fetch file text content, restricted by security resolve rules."""
    try:
        settings, _ = await rag_manager.ensure_initialized()
        
        target_path = secure_resolve(doc_id, settings.data_dir)
        
        if not target_path.exists():
            raise HTTPException(status_code=404, detail=f"File {doc_id} does not exist.")
            
        if target_path.is_dir():
            raise HTTPException(status_code=400, detail="Requested path is a directory.")
            
        # Max file size 500KB to protect against huge loads
        max_bytes = 500 * 1024
        if target_path.stat().st_size > max_bytes:
            raise HTTPException(status_code=400, detail="File size exceeds direct preview limit (500KB).")
            
        content = target_path.read_text(encoding="utf-8", errors="replace")
        return {"content": content}
        
    except PermissionError as pe:
        logger.warning("Path traversal guard trigger: %s", pe)
        raise HTTPException(status_code=403, detail=str(pe))
    except Exception as e:
        logger.exception("Error reading file text")
        raise HTTPException(status_code=500, detail=str(e))


# Route serving the compiled React App bundle static files
STATIC_DIR = Path(__file__).resolve().parent / "static"

@app.get("/{path_name:path}")
async def serve_frontend(path_name: str):
    """Fallback route serving index.html or target static files from UI build."""
    # Ensure static directory exists
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    
    # If the user requests an asset file (like css or js) and it exists, serve it
    file_path = STATIC_DIR / path_name
    if file_path.is_file() and not path_name.startswith("api/"):
        return FileResponse(file_path)
        
    # Serve index.html as fallback for React SPA client-side routing
    index_html = STATIC_DIR / "index.html"
    if index_html.is_file():
        return FileResponse(index_html)
        
    # If build static assets don't exist yet, return a clean HTML message
    return FileResponse(Path(__file__).resolve().parent / "fallback.html")


def start_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Uvicorn launcher helper."""
    import uvicorn
    # Create fallback.html in case static assets are not pre-compiled
    fallback_html = Path(__file__).resolve().parent / "fallback.html"
    if not fallback_html.exists():
        fallback_html.write_text(
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head><title>Personal Retrieval Assistant</title></head>\n"
            "<body style='background-color:#0b0f19; color:#fff; font-family:sans-serif; text-align:center; padding:100px;'>\n"
            "  <h1>Personal Retrieval Assistant Web Server</h1>\n"
            "  <p style='color:#a0aec0;'>The server is up and listening. However, the React frontend static bundle has not been compiled yet.</p>\n"
            "  <p style='color:#a0aec0;'>Please run <code>make web-build</code> or compile Vite in the <code>ui/</code> directory to activate the UI dashboard.</p>\n"
            "</body>\n"
            "</html>\n",
            encoding="utf-8"
        )
        
    logger.info("Starting FastAPI Uvicorn engine on %s:%s (reload=%s)", host, port, reload)
    uvicorn.run("retrieval_assistant.web:app", host=host, port=port, reload=reload)
