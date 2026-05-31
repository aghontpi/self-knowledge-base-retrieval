"""Routing: decide what to do with a file based on its name/extension.

A file maps to a :class:`Route` (which *domain* — prose vs code — and which
*kind* — selects the parser and chunker), or to ``None`` meaning "skip".

This is a pure, fast, name-only decision. The heavier guards (size cap and the
binary sniff that catches mislabeled/compiled files) live in
:mod:`retrieval_assistant.discovery`, because they require touching the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- domain "code": source files and notebooks -----------------------------
CODE_SUFFIXES = {
    ".py", ".pyw", ".java", ".kt", ".kts", ".c", ".h", ".cpp", ".cc", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".js", ".jsx", ".ts", ".tsx", ".mjs",
    ".cjs", ".swift", ".scala", ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".sql", ".pl", ".lua", ".r", ".m", ".dart",
}
NOTEBOOK_SUFFIXES = {".ipynb"}

# --- domain "prose": everything textual that isn't source code --------------
MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx", ".rst"}
DOC_SUFFIXES = {".pdf", ".docx"}
TABULAR_SUFFIXES = {".csv", ".tsv"}
TEXT_SUFFIXES = {".txt", ".text", ".rtf"}
# Config/markup: treated as prose text (small, occasionally useful to search).
CONFIG_SUFFIXES = {
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".properties",
    ".xml", ".env", ".manifest", ".service", ".plist",
}

# --- never embed: binaries, archives, secrets, editor cruft -----------------
SKIP_SUFFIXES = {
    ".class", ".o", ".obj", ".so", ".dylib", ".dll", ".a", ".lib", ".exe",
    ".pyc", ".pyo", ".out", ".bin", ".db", ".db2", ".sqlite", ".sqlite3",
    ".zip", ".tar", ".gz", ".tgz", ".xz", ".bz2", ".7z", ".rar", ".jar", ".war",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".ico", ".tiff",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".pdf~",
    ".pem", ".key", ".p12", ".pfx", ".crt", ".cer", ".keystore",
    ".prof", ".lprof", ".lock", ".log",
    ".swp", ".swo", ".swn", ".tmp", ".temp", ".bak", ".orig",
}

# Backup / status suffixes that wrap a real file (e.g. ``5.py.save``,
# ``81.txt.completed``). These duplicate an active file -> skip to avoid
# near-duplicate chunks polluting retrieval.
BACKUP_SUFFIXES = {".save", ".completed", ".complete", ".bak", ".orig", ".old"}

# Exact filenames to ignore.
SKIP_NAMES = {".DS_Store", ".gitmodules", ".gitignore", ".gitattributes", "Thumbs.db"}

# Directory names whose contents are never useful to embed (IDE config,
# dependency/build trees). Skipped even when git happens to track them.
SKIP_DIRS = {
    ".idea", ".vscode", ".git", ".github", "node_modules", "__pycache__",
    ".venv", "venv", "env", "dist", "build", ".next", "target", "vendor",
    ".gradle", ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages",
    ".cache", "__MACOSX",
}


@dataclass(frozen=True)
class Route:
    domain: str  # "prose" | "code"
    kind: str    # parser/chunker selector: text|markdown|pdf|docx|tabular|code|notebook


def route_for(path: Path) -> Route | None:
    """Return the Route for a file, or None if it should be skipped."""
    name = path.name
    if name in SKIP_NAMES:
        return None
    if SKIP_DIRS.intersection(path.parts):
        return None

    suffix = path.suffix.lower()

    # Backup-wrapped files: last suffix is a backup marker.
    if suffix in BACKUP_SUFFIXES:
        return None
    if suffix in SKIP_SUFFIXES:
        return None

    if suffix in NOTEBOOK_SUFFIXES:
        return Route(domain="code", kind="notebook")
    if suffix in CODE_SUFFIXES:
        return Route(domain="code", kind="code")

    if suffix in MARKDOWN_SUFFIXES:
        return Route(domain="prose", kind="markdown")
    if suffix == ".pdf":
        return Route(domain="prose", kind="pdf")
    if suffix == ".docx":
        return Route(domain="prose", kind="docx")
    if suffix in TABULAR_SUFFIXES:
        return Route(domain="prose", kind="tabular")
    if suffix in TEXT_SUFFIXES or suffix in CONFIG_SUFFIXES:
        return Route(domain="prose", kind="text")

    # Unknown / no extension (e.g. compiled binaries with no suffix): skip.
    return None
