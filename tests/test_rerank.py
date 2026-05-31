"""Tests for the cross-encoder reranker logic (model mocked — no download)."""

from __future__ import annotations

from retrieval_assistant.config import load_settings
from retrieval_assistant.rerank import Reranker
from retrieval_assistant.store import SearchHit


def _hit(doc_id: str, text: str, score: float, source: str) -> SearchHit:
    return SearchHit(doc_id=doc_id, locator="L1", text=text, score=score,
                     chunk_index=0, source=source)


class _FakeModel:
    """Stand-in CrossEncoder: score = length of the chunk text."""

    def predict(self, pairs):
        return [float(len(chunk)) for _, chunk in pairs]


def _reranker_with_fake():
    r = Reranker(load_settings())
    r._model = _FakeModel()  # bypass lazy load / download
    return r


def test_rerank_reorders_by_cross_encoder_score():
    hits = [
        _hit("a.py", "x", score=0.99, source="code"),       # high cosine, short text
        _hit("b.md", "much longer text here", score=0.10, source="prose"),
    ]
    out = _reranker_with_fake().rerank("q", hits, top_k=5)
    # Fake scorer favors longer text, so the prose hit wins despite lower cosine.
    assert out[0].doc_id == "b.md"
    assert out[1].doc_id == "a.py"


def test_rerank_sets_rerank_score_and_keeps_cosine():
    hits = [_hit("a.py", "abc", score=0.5, source="code")]
    out = _reranker_with_fake().rerank("q", hits, top_k=5)
    assert out[0].rerank_score == 3.0   # len("abc")
    assert out[0].score == 0.5          # original cosine preserved


def test_rerank_respects_top_k():
    hits = [_hit(f"f{i}", "t" * i, score=0.1, source="code") for i in range(1, 6)]
    out = _reranker_with_fake().rerank("q", hits, top_k=2)
    assert len(out) == 2
    assert [h.doc_id for h in out] == ["f5", "f4"]  # longest two


def test_rerank_empty():
    assert _reranker_with_fake().rerank("q", [], top_k=5) == []
