.PHONY: test lint typecheck reproduce figures verify all clean

PY ?= python3

all: lint test reproduce figures

test:
	$(PY) -m pytest tests/ -q

lint:
	ruff check src tests scripts

typecheck:
	mypy src

# Regenerate every scientific result from scratch, deterministically.
reproduce:
	$(PY) scripts/reproduce_toy_results.py --out results/

# Regenerate every figure in the README from the current results/*.json.
figures:
	$(PY) scripts/generate_figures.py --results results/ --out figures/

# Recompute results and fail if any number moved — the CI regression gate.
# Scientific regression detection: hashes of deterministic JSON reports.
verify: reproduce
	$(PY) scripts/verify_result_hashes.py results/ expected_hashes.json

clean:
	rm -rf results/ .pytest_cache .ruff_cache .mypy_cache
