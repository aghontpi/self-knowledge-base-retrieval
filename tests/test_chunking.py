"""Tests for the deterministic chunker.

These pin the properties ingestion relies on: bounded window size, correct
overlap, contiguous global indices, locator propagation, and full coverage of
the source text.
"""

from __future__ import annotations

import pytest

from retrieval_assistant.chunking import Chunk, chunk_blocks
from retrieval_assistant.parsing import Block
from retrieval_assistant.store import chunk_pk


def _blocks(*pairs: tuple[str, str]) -> list[Block]:
    return [Block(text=t, locator=loc) for t, loc in pairs]


def test_short_input_is_a_single_chunk():
    chunks = chunk_blocks(_blocks(("hello world", "p.1")), chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].locator == "p.1"
    assert chunks[0].chunk_index == 0


def test_indices_are_contiguous_and_zero_based():
    text = "x" * 1000
    chunks = chunk_blocks(_blocks((text, "p.1")), chunk_size=100, chunk_overlap=20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_window_size_is_bounded():
    text = "abcde" * 500  # 2500 chars
    chunks = chunk_blocks(_blocks((text, "p.1")), chunk_size=300, chunk_overlap=50)
    assert all(len(c.text) <= 300 for c in chunks)
    assert len(chunks) > 1


def test_overlap_is_respected():
    text = "".join(str(i % 10) for i in range(1000))
    size, overlap = 200, 40
    chunks = chunk_blocks(_blocks((text, "p.1")), chunk_size=size, chunk_overlap=overlap)
    # The tail of each full window must reappear at the head of the next.
    for a, b in zip(chunks, chunks[1:]):
        if len(a.text) == size:
            assert a.text[-overlap:] == b.text[:overlap]


def test_full_text_is_covered():
    text = "".join(chr(ord("a") + (i % 26)) for i in range(1234))
    size, overlap = 250, 60
    chunks = chunk_blocks(_blocks((text, "p.1")), chunk_size=size, chunk_overlap=overlap)
    # Reconstruct by removing the overlap that each subsequent chunk repeats.
    rebuilt = chunks[0].text + "".join(c.text[overlap:] for c in chunks[1:])
    assert rebuilt == text


def test_blocks_are_joined_with_separator():
    chunks = chunk_blocks(
        _blocks(("alpha", "p.1"), ("beta", "p.2")), chunk_size=100, chunk_overlap=10
    )
    assert chunks[0].text == "alpha\n\nbeta"


def test_empty_blocks_are_skipped():
    chunks = chunk_blocks(
        _blocks(("", "p.1"), ("   ", "p.2"), ("real", "p.3")),
        chunk_size=100,
        chunk_overlap=10,
    )
    assert len(chunks) == 1
    assert chunks[0].text == "real"
    assert chunks[0].locator == "p.3"


def test_no_blocks_produces_no_chunks():
    assert chunk_blocks([], chunk_size=100, chunk_overlap=10) == []


def test_locator_tracks_opening_block():
    first = "a" * 90
    chunks = chunk_blocks(
        _blocks((first, "p.1"), ("b" * 50, "p.2")), chunk_size=100, chunk_overlap=10
    )
    assert chunks[0].locator == "p.1"


@pytest.mark.parametrize("overlap", [-1, 100, 200])
def test_invalid_overlap_rejected(overlap):
    with pytest.raises(ValueError):
        chunk_blocks(_blocks(("x", "p.1")), chunk_size=100, chunk_overlap=overlap)


def test_chunk_pk_is_deterministic_and_unique():
    assert chunk_pk("a/b.pdf", 0) == chunk_pk("a/b.pdf", 0)
    assert chunk_pk("a/b.pdf", 0) != chunk_pk("a/b.pdf", 1)
    assert chunk_pk("a/b.pdf", 0) != chunk_pk("a/c.pdf", 0)
    assert len(chunk_pk("a/b.pdf", 0)) == 40
