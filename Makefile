.PHONY: test lint typecheck reproduce figures verify verify-hashes all clean

PY ?= python3

all: lint test reproduce figures

test:
	$(PY) -m pytest tests/ -q

lint:
	ruff check src tests scripts

typecheck:
	mypy src

reproduce:
	$(PY) scripts/reproduce_toy_results.py --out results/

figures:
	$(PY) scripts/generate_figures.py --results results/ --out figures/

verify-hashes: reproduce
	$(PY) scripts/verify_result_hashes.py results/ expected_hashes.json

verify: reproduce
	$(PY) scripts/verify_results_tolerance.py results/ expected_results.json

clean:
	rm -rf results/ .pytest_cache .ruff_cache .mypy_cache
