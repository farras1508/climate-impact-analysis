# Phase 0 task runner; grows into the full pipeline in Phase 11.
.PHONY: help test lint verify-data mlflow-ui
check-sources:  ## Check every HTTP data source is reachable (writes reports/source_check.md)
	$(PY) -m climate_impact.ingest.source_check
data-http:  ## Download all direct-URL HTTP datasets into data/raw (provenance-recorded)
	$(PY) -m climate_impact.ingest.http all
data-era5:  ## Download ERA5 monthly means by year (CDS queue; resumable)
	$(PY) -m climate_impact.ingest.era5

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
profile:  ## Step 3.1: profile every raw dataset (writes reports/profiling/)
	$(PY) -m climate_impact.quality.profile
quality:  ## Step 3.2: score every raw table on the data-quality dimensions (reports/quality/)
	$(PY) -m climate_impact.quality.checks
crosschecks:  ## Step 3.2b: consistency/integrity checks, ERA5 extremes, EM-DAT lag (reports/quality/)
	$(PY) -m climate_impact.quality.crosschecks
domain:  ## Step 3.3: baselines, state successions, reporting effort (reports/quality/)
	$(PY) -m climate_impact.quality.domain
