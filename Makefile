lint:
	# Format files first (in place)
	ruff format geoIR tests
	# Then, check for linting issues and fix them (in place)
	ruff check geoIR tests --fix --show-fixes

type:
	mypy geoIR

test:
	# Run unit tests
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

summary:
	python scripts/concat.py

activate_env:
	source .venv/bin/activate

