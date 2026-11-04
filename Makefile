.PHONY: install lint format typecheck test check

install:
	pip install -e ".[dev,serve,hnsw]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy

test:
	pytest -q

check: lint typecheck test
