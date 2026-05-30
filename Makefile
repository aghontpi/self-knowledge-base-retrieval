# Personal Retrieval Assistant — common tasks.
# Override the query string: make query Q="your question here"

PYTHON ?= python3
Q ?= What is this corpus about?

# Runtime env for ingest/query:
# - KMP_DUPLICATE_LIB_OK: PyTorch and pymilvus each link an OpenMP runtime;
#   on macOS the duplicate aborts with "OMP: Error #15" unless this is set.
# - OMP_NUM_THREADS: cap CPU threads so embedding stays responsive.
# - TOKENIZERS_PARALLELISM=false: silence HF tokenizer fork warnings.
PRA_ENV = KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

.PHONY: install ingest query stats test clean

install:  ## Create/refresh editable install with dev deps
	$(PYTHON) -m pip install -e ".[dev]"

ingest:  ## Converge the index with the contents of data/
	$(PRA_ENV) pra ingest

query:  ## Run a retrieval query: make query Q="..."
	$(PRA_ENV) pra query "$(Q)"

stats:  ## Show collection stats
	$(PRA_ENV) pra stats

test:  ## Run the test suite
	$(PYTHON) -m pytest -q

clean:  ## Remove caches and build artifacts (keeps the index .db)
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
