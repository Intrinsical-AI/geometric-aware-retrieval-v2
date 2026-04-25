PYTHON ?= python
RUFF ?= $(PYTHON) -m ruff
MYPY ?= $(PYTHON) -m mypy
PYTEST ?= $(PYTHON) -m pytest

format:
	$(RUFF) format .
	$(RUFF) check . --fix --show-fixes

lint:
	$(RUFF) format --check .
	$(RUFF) check .

type:
	$(MYPY) geoIR

test:
	# Run unit tests
	$(PYTEST) tests/ -v

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
