.PHONY: install dev test lint format run clean eval

install:
	python -m pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload

test:
	python -m pytest

lint:
	python -m ruff check .

format:
	python -m ruff check . --fix

run:
	uvicorn app.main:app

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

eval:
	python scripts/run_retrieval_eval.py