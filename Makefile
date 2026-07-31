# =============================================================================
# RAG Pipeline with Hybrid Search — Makefile
# =============================================================================

PYTHON   := .venv/bin/python
PIP      := .venv/bin/pip
UV       := uv
VENV_BIN := .venv/bin
COMPOSE   := podman compose
SRC      := src/rag_pipeline
TESTS    := tests
CONFIG   := config

# ---- Phase 0: Project Setup ------------------------------------------------

.PHONY: install
install: ## Install project dependencies
	$(UV) pip install -e ".[dev]"

.PHONY: install-prod
install-prod: ## Install production dependencies only
	$(UV) pip install -e .

.PHONY: init-structure
init-structure: ## Create project directory structure
	mkdir -p $(SRC)/api/routes
	mkdir -p $(SRC)/api/schemas
	mkdir -p $(SRC)/data
	mkdir -p $(SRC)/embeddings/backends
	mkdir -p $(SRC)/storage
	mkdir -p $(SRC)/retrieval
	mkdir -p $(SRC)/generation
	mkdir -p $(SRC)/evaluation
	mkdir -p $(SRC)/observability
	mkdir -p $(TESTS)/unit
	mkdir -p $(TESTS)/integration
	mkdir -p $(TESTS)/evaluation
	mkdir -p $(TESTS)/fixtures
	mkdir -p $(CONFIG)
	mkdir -p scripts
	mkdir -p evaluation
	@touch $(SRC)/__init__.py
	@touch $(SRC)/api/__init__.py
	@touch $(SRC)/api/routes/__init__.py
	@touch $(SRC)/api/schemas/__init__.py
	@touch $(SRC)/data/__init__.py
	@touch $(SRC)/embeddings/__init__.py
	@touch $(SRC)/embeddings/backends/__init__.py
	@touch $(SRC)/storage/__init__.py
	@touch $(SRC)/retrieval/__init__.py
	@touch $(SRC)/generation/__init__.py
	@touch $(SRC)/evaluation/__init__.py
	@touch $(SRC)/observability/__init__.py
	@touch $(TESTS)/__init__.py
	@touch $(TESTS)/unit/__init__.py
	@touch $(TESTS)/integration/__init__.py
	@touch $(TESTS)/evaluation/__init__.py
	@echo "✅ Project structure created"

# ---- Code Quality -----------------------------------------------------------

.PHONY: lint
lint: ## Run linter (ruff check)
	$(VENV_BIN)/ruff check $(SRC) $(TESTS)

.PHONY: format
format: ## Format code (ruff format)
	$(VENV_BIN)/ruff format $(SRC) $(TESTS)

.PHONY: format-check
format-check: ## Check formatting without modifying
	$(VENV_BIN)/ruff format --check $(SRC) $(TESTS)

.PHONY: typecheck
typecheck: ## Run type checker (mypy)
	$(VENV_BIN)/mypy $(SRC)

.PHONY: check
check: lint typecheck format-check ## Run all checks (lint + typecheck + format)

# ---- Testing ---------------------------------------------------------------

.PHONY: test
test: ## Run all tests
	$(UV) run pytest tests/ -x

.PHONY: test-evaluation
test-evaluation: ## Run evaluation tests
	$(PYTHON) -m pytest $(TESTS)/evaluation -v --tb=short

.PHONY: test-all
test-all: ## Run all tests
	$(PYTHON) -m pytest $(TESTS) -v --tb=short

.PHONY: test-unit
test-unit: ## Run unit tests only
	$(UV) run pytest tests/unit/ -x

.PHONY: test-integration
test-integration: ## Run integration tests (requires services)
	$(UV) run pytest tests/integration/ -x -m integration

.PHONY: coverage
coverage: ## Run tests with coverage report
	$(UV) run pytest tests/ --cov=rag_pipeline --cov-report=term-missing --cov-report=html -x

# ---- Podman ----------------------------------------------------------------

.PHONY: build
build: ## Build container images
	$(COMPOSE) build

.PHONY: up
up: ## Start all services (detached)
	$(COMPOSE) up -d

.PHONY: up-dev
up-dev: ## Start services with hot reload (dev mode)
	$(COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml up

.PHONY: down
down: ## Stop all services
	$(COMPOSE) down

.PHONY: logs
logs: ## Tail service logs
	$(COMPOSE) logs -f

# ---- Run -------------------------------------------------------------------

.PHONY: run
run: ## Start API server locally (uvicorn)
	$(PYTHON) -m uvicorn rag_pipeline.api.app:app --reload --host 0.0.0.0 --port 8000

.PHONY: ingest
ingest: ## Ingest files (usage: make ingest ARGS='file path/to/doc.pdf')
	$(PYTHON) scripts/ingest.py $(ARGS)

.PHONY: run-eval
run-eval: ## Run evaluation suite
	$(PYTHON) scripts/run_eval.py

.PHONY: run-experiment
run-experiment: ## Run full experiment with tracking
	$(PYTHON) scripts/run_experiment.py

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	$(PYTHON) scripts/benchmark.py

# ---- Utilities -------------------------------------------------------------

.PHONY: clean
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info .coverage coverage.xml

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
