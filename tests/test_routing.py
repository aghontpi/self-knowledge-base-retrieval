"""Tests for file routing (domain/kind classification and skip rules)."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrieval_assistant.routing import route_for


@pytest.mark.parametrize(
    "name,domain,kind",
    [
        ("main.py", "code", "code"),
        ("App.java", "code", "code"),
        ("server.go", "code", "code"),
        ("script.sh", "code", "code"),
        ("analysis.ipynb", "code", "notebook"),
        ("README.md", "prose", "markdown"),
        ("notes.txt", "prose", "text"),
        ("report.pdf", "prose", "pdf"),
        ("resume.docx", "prose", "docx"),
        ("spent_transactions.csv", "prose", "tabular"),
        ("config.yaml", "prose", "text"),
        ("settings.json", "prose", "text"),
    ],
)
def test_routes(name, domain, kind):
    route = route_for(Path(name))
    assert route is not None, name
    assert (route.domain, route.kind) == (domain, kind)


@pytest.mark.parametrize(
    "name",
    [
        "l.db2",            # huge db blob
        "Main.class",       # compiled
        "archive.zip",
        "photo.png",
        "id_rsa.pem",
        "core.swp",         # editor swap
        ".DS_Store",
        ".gitmodules",      # explicitly ignored
        "5.py.save",        # backup-wrapped
        "81.txt.completed", # status-wrapped
        "hello",            # no extension (likely compiled binary)
        "data.bin",
    ],
)
def test_skipped(name):
    assert route_for(Path(name)) is None, name


def test_case_insensitive_extension():
    assert route_for(Path("MODULE.PY")).domain == "code"
    assert route_for(Path("DOC.PDF")).kind == "pdf"
