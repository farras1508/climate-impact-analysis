"""Experiment tracking: the MLflow store works and runs carry reproducibility tags."""

import re

import mlflow

from climate_impact.tracking import configure_mlflow, get_or_create_experiment, run_context_tags

SHA1 = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def test_context_tags_identify_code_data_and_config():
    tags = run_context_tags()
    assert SHA1.fullmatch(tags["git_commit"])
    assert SHA256.fullmatch(tags["config_sha256"])
    assert SHA256.fullmatch(tags["datasets_sha256"])
    assert tags["manifest_sha256"] == "missing" or SHA256.fullmatch(tags["manifest_sha256"])
    assert tags["git_dirty"] in {"true", "false"}


def test_mlflow_run_roundtrip(tmp_path):
    configure_mlflow(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    exp_id = get_or_create_experiment("smoke-test", artifact_root=tmp_path / "artifacts")
    assert get_or_create_experiment("smoke-test", artifact_root=tmp_path / "artifacts") == exp_id

    tags = run_context_tags()
    with mlflow.start_run(experiment_id=exp_id, tags=tags) as run:
        mlflow.log_param("model", "naive_baseline")
        mlflow.log_metric("rmse", 1.23)

    stored = mlflow.get_run(run.info.run_id)
    assert stored.data.metrics["rmse"] == 1.23
    assert stored.data.tags["git_commit"] == tags["git_commit"]
