# Implementation Plan - Personal Retrieval Assistant MCP Server

This document contains the final production-ready implementation plan and architectural documentation for the **Model Context Protocol (MCP) Server** integration of the local heterogeneous RAG pipeline (Personal Retrieval Assistant).

---

## 1. Production Architecture Overview

Exposing a local RAG application to IDEs and LLM clients (like Claude Desktop or Cursor) via the Model Context Protocol requires rigorous, sandboxing-safe engineering. Simple prototypes suffer from socket lockups, handshake timeouts, and path traversal vulnerabilities.

Our final architecture implements five core reliability systems:
1.  **Fast-Handshake Subprocess Boot (Under 10ms):**
    *   Heavy imports (`torch`, `sentence-transformers`, `pymilvus`) and model weights are completely deferred and lazy-loaded on the first tool/resource execution.
    *   This ensures that the initial JSON-RPC handshake completes instantly (well under the 2-to-10 second IDE subprocess boot timeouts).
2.  **OS-Level Cross-Process Database File Locking:**
    *   Milvus Lite is an in-process engine backed by SQLite. Standard in-process locks (like `asyncio.Lock()`) fail when *separate* clients (e.g., Claude Desktop and Cursor) query the database simultaneously, causing immediate database lock corruption crashes.
    *   We utilize `filelock.FileLock` on `pra.db.lock` with a configurable timeout. This ensures safe cross-process serialization of all database operations (reads and writes) across any active clients.
3.  **Explicit Apple Silicon Sandbox-Safe CPU Enforcement:**
    *   To prevent Apple Silicon Metal Performance Shaders (MPS) sandbox SIGABRT crashes inside restricted IDE environments, the server forces all models (`SentenceTransformer` and `CrossEncoder`) to execute strictly on the CPU (`device="cpu"`).
    *   OpenMP duplicate runtime warnings are fully squelched by setting environment configurations (`OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE`) at the very top of the execution lifecycle.
4.  **Strict Path Traversal Sandboxing:**
    *   Accepting raw document identifiers (`doc_id`) creates a path traversal risk (e.g., `doc_id="../../../../etc/passwd"`).
    *   We enforce a strict containment resolver (`secure_resolve`) that guarantees target files are strictly relative to the configured `PRA_DATA_DIR` before allowing direct file inspection.
5.  **Clean Stderr Redirects for IPC Transport:**
    *   Because stdio transport uses standard output (`stdout`) for JSON-RPC message serialization, any standard `print` or `logging` output to `stdout` will break the protocol. All logging and warnings are explicitly redirected to standard error (`stderr`).

---

## 2. Dependencies

To support the MCP server, `pyproject.toml` is configured with:
*   `mcp>=1.0.0` - Standard Model Context Protocol SDK.
*   `fastmcp>=0.1.0` - High-level decorator-based MCP framework.
*   `filelock>=3.12.0` - OS-level atomic file locking.

---

## 3. Server File Structure

The server is implemented in a single, self-contained, highly-optimized entrypoint:
*   `src/retrieval_assistant/mcp_server.py`

### Exposed Tools (`@mcp.tool()`)
*   `pra_search(query: str, top_k: int = 5) -> str`: Performs semantic search over prose and code, and outputs a clean, reranked single list of snippets.
*   `pra_ingest() -> str`: Performs a full convergent ingestion cycle to synchronize the directory with the database, fully protected under an exclusive write file lock.
*   `pra_stats() -> str`: Returns counts of documents/chunks in both prose and code indexes along with active settings.
*   `pra_get_file(doc_id: str) -> str`: Safely reads the raw text of an indexed document under containment constraints.

### Exposed Resources (`@mcp.resource()`)
*   `pra://settings`: Exposes the RAG database paths, chunk sizes, and active model dimensions as JSON.

---

## 4. Claude Desktop Configuration

To activate the server inside **Claude Desktop**, register the following block in `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "personal-retrieval-assistant": {
      "command": "/Users/gopinathv/Development/Github/personal-retrieval-assistant/.venv/bin/python",
      "args": [
        "-m",
        "retrieval_assistant.mcp_server"
      ],
      "env": {
        "PRA_DATA_DIR": "/Users/gopinathv/Development/Github/-blank-",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1"
      }
    }
  }
}
```

---

## 5. Local Direct Execution / Testing

You can run the server directly in your shell for debugging or use `mcp dev` to launch the local interactive console:

```bash
# Direct run
source .venv/bin/activate
export PRA_DATA_DIR=/Users/gopinathv/Development/Github/-blank-
python -m retrieval_assistant.mcp_server

# FastMCP Inspector dev shell
npx -y @modelcontextprotocol/inspector -- /Users/gopinathv/Development/Github/personal-retrieval-assistant/.venv/bin/python -m retrieval_assistant.mcp_server
```
