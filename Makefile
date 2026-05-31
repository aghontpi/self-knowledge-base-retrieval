# Personal Retrieval Assistant — common tasks.
# Override the query string: make query Q="your question here"

PYTHON ?= python3
Q ?= What is this corpus about?

.PHONY: install ingest query stats test clean

install:  ## Create/refresh editable install with dev deps
	$(PYTHON) -m pip install -e ".[dev]"

ingest:  ## Converge the index with the contents of data/
	pra ingest

query:  ## Run a retrieval query: make query Q="..."
	pra query "$(Q)"

stats:  ## Show collection stats
	pra stats

test:  ## Run the test suite
	pytest -q

clean:  ## Remove caches and build artifacts (keeps the index .db)
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
