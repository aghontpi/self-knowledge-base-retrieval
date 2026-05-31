"""Chunking strategies, dispatched by content kind.

* **prose** (text/markdown/pdf/docx) -> :func:`chunk_blocks`: a character window
  with overlap over the joined blocks. Deterministic; pinned by the test suite.
* **code** -> :func:`chunk_code`: Python is split on function/class boundaries
  via ``ast``; every other language falls back to a line window with overlap.
* **tabular** -> :func:`chunk_rows`: one row block becomes one chunk.

Every Chunk carries a global, monotonic ``chunk_index`` (half of the
deterministic primary key) and a ``locator`` pointing back into the source.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .config import Settings
from .parsing import Block

_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class Chunk:
    text: str
    locator: str
    chunk_index: int


def chunk_for_kind(blocks: list[Block], kind: str, settings: Settings, *, suffix: str = "") -> list[Chunk]:
    """Pick a chunking strategy from the routing ``kind``."""
    if kind == "tabular":
        return chunk_rows(blocks)
    if kind in ("code", "notebook"):
        return chunk_code(blocks, settings, suffix=suffix)
    return chunk_blocks(blocks, settings.chunk_size, settings.chunk_overlap)


# --------------------------------------------------------------------------
# Prose: fixed-size character window + overlap
# --------------------------------------------------------------------------
def chunk_blocks(blocks: list[Block], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("require 0 <= chunk_overlap < chunk_size")

    chunks: list[Chunk] = []
    index = 0
    buf = ""
    buf_locator: str | None = None

    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        if buf_locator is None:
            buf_locator = block.locator
        buf = f"{buf}{_SEPARATOR}{text}" if buf else text

        while len(buf) >= chunk_size:
            chunks.append(Chunk(text=buf[:chunk_size], locator=buf_locator, chunk_index=index))
            index += 1
            buf = buf[chunk_size - chunk_overlap :]
            buf_locator = block.locator

    if buf.strip():
        chunks.append(Chunk(text=buf, locator=buf_locator or "", chunk_index=index))

    return chunks


# --------------------------------------------------------------------------
# Tabular: one row -> one chunk
# --------------------------------------------------------------------------
def chunk_rows(blocks: list[Block]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, block in enumerate(blocks):
        text = block.text.strip()
        if text:
            chunks.append(Chunk(text=text, locator=block.locator, chunk_index=index))
    return chunks


# --------------------------------------------------------------------------
# Code: Python via AST, everything else via line window
# --------------------------------------------------------------------------
def chunk_code(blocks: list[Block], settings: Settings, *, suffix: str = "") -> list[Chunk]:
    """Split source code. ``blocks`` holds a single whole-file block."""
    if not blocks:
        return []
    source = blocks[0].text
    max_lines = settings.code_max_lines
    overlap_lines = settings.code_overlap_lines

    if suffix.lower() in (".py", ".pyw"):
        spans = _python_spans(source)
        if spans is not None:
            return _spans_to_chunks(source, spans, max_lines, overlap_lines)

    # Generic / fallback: line window with overlap.
    return _line_window(source, 1, _line_count(source), max_lines, overlap_lines, start_index=0)


def _line_count(source: str) -> int:
    return source.count("\n") + 1


def _python_spans(source: str) -> list[tuple[int, int]] | None:
    """Top-level (start,end) line spans: module preamble + each def/class.

    Returns None if the source does not parse, so the caller falls back.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    total = _line_count(source)
    tops = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not tops:
        return [(1, total)]

    spans: list[tuple[int, int]] = []
    cursor = 1
    for node in tops:
        start = node.lineno
        end = getattr(node, "end_lineno", start) or start
        # Anything between the previous span and this def (imports, top code).
        if start > cursor:
            spans.append((cursor, start - 1))
        spans.append((start, end))
        cursor = end + 1
    if cursor <= total:
        spans.append((cursor, total))
    return spans


def _spans_to_chunks(
    source: str, spans: list[tuple[int, int]], max_lines: int, overlap_lines: int
) -> list[Chunk]:
    lines = source.splitlines()
    chunks: list[Chunk] = []
    index = 0
    for start, end in spans:
        text = "\n".join(lines[start - 1 : end]).strip()
        if not text:
            continue
        span_len = end - start + 1
        if span_len <= max_lines:
            chunks.append(Chunk(text=text, locator=f"L{start}-{end}", chunk_index=index))
            index += 1
        else:
            # Oversized def/class: sub-split by line window.
            sub = _line_window(source, start, end, max_lines, overlap_lines, start_index=index)
            chunks.extend(sub)
            index += len(sub)
    return chunks


def _line_window(
    source: str,
    start_line: int,
    end_line: int,
    max_lines: int,
    overlap_lines: int,
    *,
    start_index: int,
) -> list[Chunk]:
    if max_lines <= 0:
        raise ValueError("code_max_lines must be positive")
    if not 0 <= overlap_lines < max_lines:
        raise ValueError("require 0 <= code_overlap_lines < code_max_lines")

    lines = source.splitlines()
    chunks: list[Chunk] = []
    index = start_index
    pos = start_line  # 1-based, inclusive
    step = max_lines - overlap_lines
    while pos <= end_line:
        win_end = min(pos + max_lines - 1, end_line)
        text = "\n".join(lines[pos - 1 : win_end]).strip()
        if text:
            chunks.append(Chunk(text=text, locator=f"L{pos}-{win_end}", chunk_index=index))
            index += 1
        if win_end >= end_line:
            break
        pos += step
    return chunks
