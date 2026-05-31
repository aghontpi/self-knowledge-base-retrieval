"""Format-aware parsing: a file becomes a list of :class:`Block`.

A Block is a logical unit of text with a human-readable ``locator`` describing
where it came from (``p.3`` for a PDF page, ``row.12`` for a CSV row,
``L1`` for the start of a source file). Chunking consumes Blocks and carries the
locator forward so search hits can be traced back to their source.

The parser to use is selected by the ``kind`` from :mod:`retrieval_assistant.routing`
rather than re-deriving it from the suffix here. Heavy parsers (PyMuPDF,
python-docx) are imported lazily.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Block:
    text: str
    locator: str


def parse_file(path: Path, kind: str) -> list[Block]:
    """Parse a file into text blocks according to its routing ``kind``."""
    if kind == "pdf":
        return _parse_pdf(path)
    if kind == "docx":
        return _parse_docx(path)
    if kind == "markdown":
        return _parse_markdown(path)
    if kind == "tabular":
        return _parse_tabular(path)
    if kind == "notebook":
        return _parse_notebook(path)
    if kind == "code":
        return _parse_code(path)
    if kind == "text":
        return _parse_text(path)
    raise ValueError(f"Unknown parse kind {kind!r} for {path}")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_pdf(path: Path) -> list[Block]:
    import fitz  # PyMuPDF

    blocks: list[Block] = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                blocks.append(Block(text=text, locator=f"p.{page_index}"))
    return blocks


def _parse_docx(path: Path) -> list[Block]:
    import docx

    document = docx.Document(str(path))
    blocks: list[Block] = []
    for para_index, para in enumerate(document.paragraphs, start=1):
        text = para.text.strip()
        if text:
            blocks.append(Block(text=text, locator=f"para.{para_index}"))
    return blocks


def _parse_text(path: Path) -> list[Block]:
    raw = _read_text(path)
    blocks: list[Block] = []
    para_index = 1
    for para in raw.split("\n\n"):
        text = para.strip()
        if text:
            blocks.append(Block(text=text, locator=f"¶{para_index}"))
            para_index += 1
    return blocks


def _parse_markdown(path: Path) -> list[Block]:
    """Split on ATX headings so each section is its own block (locator = heading)."""
    raw = _read_text(path)
    blocks: list[Block] = []
    current_heading = path.name
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            blocks.append(Block(text=text, locator=current_heading[:255]))

    for line in raw.splitlines():
        if line.lstrip().startswith("#"):
            flush()
            buf = [line]
            current_heading = line.lstrip("# ").strip() or current_heading
        else:
            buf.append(line)
    flush()
    return blocks


def _parse_tabular(path: Path) -> list[Block]:
    """One block per row: ``col: value | col: value`` (locator = ``row.N``)."""
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    blocks: list[Block] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        header: list[str] | None = None
        for row_index, row in enumerate(reader, start=1):
            if header is None:
                header = [c.strip() for c in row]
                continue
            cells = [c.strip() for c in row]
            if not any(cells):
                continue
            pairs = [
                f"{header[i] if i < len(header) else f'col{i}'}: {val}"
                for i, val in enumerate(cells)
                if val
            ]
            text = " | ".join(pairs)
            if text:
                blocks.append(Block(text=text, locator=f"row.{row_index}"))
    return blocks


def _parse_notebook(path: Path) -> list[Block]:
    """Extract code + markdown cell sources from a Jupyter notebook."""
    try:
        nb = json.loads(_read_text(path))
    except json.JSONDecodeError:
        return []
    blocks: list[Block] = []
    for cell_index, cell in enumerate(nb.get("cells", []), start=1):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        text = src.strip()
        if text:
            kind = cell.get("cell_type", "cell")
            blocks.append(Block(text=text, locator=f"cell.{cell_index}[{kind}]"))
    return blocks


def _parse_code(path: Path) -> list[Block]:
    """Whole source as a single block; the code chunker does the splitting."""
    text = _read_text(path)
    if not text.strip():
        return []
    return [Block(text=text, locator="L1")]
