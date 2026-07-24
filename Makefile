.PHONY: help sync verify-cpu format format-check lint types type test check pre-commit build clean summary

.DEFAULT_GOAL := help

UV_CACHE_DIR ?= .uv_cache
PRE_COMMIT_HOME ?= .pre-commit-cache
UV := UV_CACHE_DIR=$(UV_CACHE_DIR) uv
RUN := $(UV) run --no-sync
RUFF_TARGETS := geoIR tests research/beir_euclidean_vs_geo.py run_exps.py

help:
	@echo "Targets: sync verify-cpu format format-check lint types test check pre-commit build"

sync:
	$(UV) sync --locked --extra dev --extra hf

verify-cpu: sync
	$(RUN) python -c 'import torch; assert torch.version.cuda is None, f"expected CPU-only torch, got CUDA {torch.version.cuda}"; print(f"torch={torch.__version__} backend=cpu")'

format: sync
	$(RUN) ruff format $(RUFF_TARGETS)

format-check: sync
	$(RUN) ruff format --check $(RUFF_TARGETS)

lint: sync
	$(RUN) ruff check $(RUFF_TARGETS)

types: sync
	$(RUN) mypy geoIR

type: types

test: sync
	$(RUN) pytest tests -q

check: verify-cpu format-check lint types test

pre-commit: sync
	PRE_COMMIT_HOME=$(PRE_COMMIT_HOME) $(RUN) pre-commit run --all-files

build: sync
	$(UV) build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

summary:
	@echo "No summary script available"

activate_env:
	source .venv/bin/activate
