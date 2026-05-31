"""File discovery: turn a data directory into a list of ingestable files.

Selection happens in three gates, cheapest first:

1. **Listing** — if the data dir is a git repo, use ``git ls-files`` so we only
   see tracked files (this excludes ``node_modules``, virtualenvs, build output,
   and anything gitignored for free). Otherwise walk the tree.
2. **Routing** — :func:`routing.route_for` decides domain/kind by name, or skips.
3. **Content guards** — a size cap and a binary sniff (null byte / non-UTF-8 in
   the first 8 KB). The sniff is what catches *mislabeled* files: extension-less
   compiled binaries, ``.db2`` blobs, anything whose bytes aren't text.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .routing import Route, route_for

_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class Discovered:
    path: Path       # absolute path on disk
    doc_id: str      # path relative to data_dir (stable identity)
    route: Route


def _git_tracked_files(data_dir: Path) -> list[Path] | None:
    """Relative paths tracked by git under data_dir, or None if not a repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(data_dir), "ls-files", "-z"],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    rels = [r for r in out.stdout.decode("utf-8", "replace").split("\0") if r]
    return [data_dir / r for r in rels]


def looks_binary(path: Path) -> bool:
    """Heuristic: a null byte or invalid UTF-8 in the first chunk => binary."""
    try:
        with path.open("rb") as fh:
            chunk = fh.read(_SNIFF_BYTES)
    except OSError:
        return True
    if b"\x00" in chunk:
        return True
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def discover_files(settings: Settings) -> list[Discovered]:
    data_dir = settings.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    candidates: list[Path] | None = None
    if settings.use_git:
        candidates = _git_tracked_files(data_dir)
    if candidates is None:
        candidates = [p for p in data_dir.rglob("*") if p.is_file()]

    discovered: list[Discovered] = []
    for path in sorted(candidates):
        if not path.is_file():  # git can list submodule gitlinks / deleted files
            continue
        route = route_for(path)
        if route is None:
            continue
        try:
            if path.stat().st_size > settings.max_file_bytes:
                continue
        except OSError:
            continue
        if looks_binary(path):
            continue
        doc_id = path.relative_to(data_dir).as_posix()
        discovered.append(Discovered(path=path, doc_id=doc_id, route=route))
    return discovered
