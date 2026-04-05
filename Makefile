.PHONY: test lint format clean install-dev docs help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install-dev:  ## Install package in dev mode with all extras
	pip install -e ".[dev]"

test:  ## Run all tests with coverage
	python -m pytest tests/ -v --tb=short --cov=xldvp_seg --cov-report=term-missing

lint:  ## Run ruff check and black check
	python -m ruff check .
	python -m black --check .

format:  ## Auto-format with ruff and black
	python -m ruff check --fix .
	python -m black .

docs:  ## Build docs locally (mkdocs serve)
	python -m mkdocs serve

docs-build:  ## Build static docs site
	python -m mkdocs build

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info/ .eggs/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
