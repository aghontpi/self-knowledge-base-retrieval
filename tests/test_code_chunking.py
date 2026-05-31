"""Tests for code-aware and tabular chunking."""

from __future__ import annotations

from retrieval_assistant.chunking import chunk_code, chunk_rows
from retrieval_assistant.config import load_settings
from retrieval_assistant.parsing import Block

SETTINGS = load_settings()

PY_SOURCE = '''\
import os
import sys

CONST = 42


def foo(x):
    return x + 1


class Bar:
    def method(self):
        return foo(self.value)


def baz():
    return CONST
'''


def _code(source: str, suffix: str):
    return chunk_code([Block(source, "L1")], SETTINGS, suffix=suffix)


def test_python_splits_on_definitions():
    chunks = _code(PY_SOURCE, ".py")
    # preamble (imports + CONST) + foo + Bar + baz == 4 logical chunks
    assert len(chunks) == 4
    assert chunks[0].chunk_index == 0
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]
    assert "import os" in chunks[0].text
    assert "def foo" in chunks[1].text
    assert "class Bar" in chunks[2].text
    assert "def baz" in chunks[3].text


def test_python_locators_are_line_ranges():
    chunks = _code(PY_SOURCE, ".py")
    assert all(c.locator.startswith("L") for c in chunks)


def test_invalid_python_falls_back_to_line_window():
    broken = "def f(:\n  this is ( not python\n" + "x = 1\n" * 5
    chunks = _code(broken, ".py")
    assert len(chunks) >= 1
    assert all(c.locator.startswith("L") for c in chunks)


def test_non_python_uses_line_window():
    java = "\n".join(f"int x{i} = {i};" for i in range(200))
    chunks = _code(java, ".java")
    assert len(chunks) > 1  # 200 lines > code_max_lines
    assert all(c.locator.startswith("L") for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_empty_code_yields_nothing():
    assert chunk_code([], SETTINGS, suffix=".py") == []
    assert chunk_code([Block("   \n  ", "L1")], SETTINGS, suffix=".py") == []


def test_chunk_rows_one_per_row():
    blocks = [Block(f"id: {i} | name: n{i}", f"row.{i}") for i in range(1, 6)]
    chunks = chunk_rows(blocks)
    assert len(chunks) == 5
    assert chunks[0].locator == "row.1"
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3, 4]
