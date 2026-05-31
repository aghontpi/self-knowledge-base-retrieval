"""Command-line interface: ``pra ingest`` / ``pra query`` / ``pra stats``."""

from __future__ import annotations

import argparse
import sys
import textwrap

from .config import load_settings
from .ingest import EmbeddingModelMismatch, ingest
from .search import search_grouped
from .store import MilvusStore


def _cmd_ingest(args: argparse.Namespace) -> int:
    settings = load_settings()
    try:
        reports = ingest(settings)
    except (EmbeddingModelMismatch, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for report in reports:
        print(f"ingest complete: {report.summary()}")
        if args.verbose:
            for label, ids in (
                ("new", report.new),
                ("changed", report.changed),
                ("removed", report.removed),
            ):
                for doc_id in ids:
                    print(f"  [{report.domain}/{label}] {doc_id}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    settings = load_settings()
    grouped = search_grouped(args.query, settings, top_k=args.top_k)

    # Results are grouped by domain because cross-model scores are not
    # comparable — see search.py. Each section is ranked on its own scale.
    total = sum(len(hits) for _, hits in grouped)
    if total == 0:
        print("no results (is the index empty? run `pra ingest`)")
        return 0

    for domain_key, hits in grouped:
        print(f"=== {domain_key} ===")
        if not hits:
            print("    (no matches)\n")
            continue
        for rank, hit in enumerate(hits, start=1):
            snippet = textwrap.shorten(" ".join(hit.text.split()), width=280, placeholder=" …")
            print(f"#{rank}  score={hit.score:.4f}  {hit.doc_id} ({hit.locator})")
            print(textwrap.indent(snippet, "    "))
            print()
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    settings = load_settings()
    print(f"db_path  : {settings.db_path}")
    print(f"data_dir : {settings.data_dir}")
    print(f"use_git  : {settings.use_git}   max_file_mb : {settings.max_file_bytes / 1048576:.2f}")
    for domain in settings.domains():
        store = MilvusStore(settings.db_path, domain.collection, domain.embedding_dim)
        state = store.existing_doc_state()
        models = {meta["embedding_model"] for meta in state.values()}
        print(
            f"\n[{domain.key}] collection={domain.collection} "
            f"model={domain.embedding_model} (dim {domain.embedding_dim})"
        )
        print(f"    documents : {len(state)}")
        print(f"    chunks    : {store.count()}")
        print(f"    stored    : {', '.join(sorted(models)) if models else '(none)'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pra",
        description="Personal Retrieval Assistant — local vector RAG over your documents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="converge both collections with the data dir")
    p_ingest.add_argument("-v", "--verbose", action="store_true", help="list affected documents")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_query = sub.add_parser("query", help="retrieve top-k chunks across both collections")
    p_query.add_argument("query", help="the natural-language query")
    p_query.add_argument("-k", "--top-k", type=int, default=None, help="number of results")
    p_query.set_defaults(func=_cmd_query)

    p_stats = sub.add_parser("stats", help="show per-collection statistics")
    p_stats.set_defaults(func=_cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
