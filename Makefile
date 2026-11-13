# Phase 0 task runner; grows into the full pipeline in Phase 11.
.PHONY: help test lint verify-data mlflow-ui
check-sources:  ## Check every HTTP data source is reachable (writes reports/source_check.md)
	$(PY) -m climate_impact.ingest.source_check

PY := python

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

test:  ## Run the test suite
	$(PY) -m pytest -q

lint:  ## Run every pre-commit check on all files
	pre-commit run --all-files

verify-data:  ## Re-hash raw data against the manifest
	$(PY) -m climate_impact.provenance verify

mlflow-ui:  ## Browse experiments at http://127.0.0.1:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
