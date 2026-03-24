.PHONY: install install-dev setup run test lint format clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install with dev dependencies (pytest, black, mypy)
	pip install -e ".[dev]"

setup: ## Download MediaPipe model files
	python setup_models.py

run: ## Launch IrisFlow (requires webcam)
	python -m src

test: ## Run test suite with coverage
	python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint: ## Type-check with mypy
	python -m mypy src/ --ignore-missing-imports

format: ## Auto-format code with black
	python -m black src/ tests/

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
