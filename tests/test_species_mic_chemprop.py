import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import amr_multiview.species_mic_chemprop as chemprop_module
from amr_multiview.species_mic_chemprop import (
    command_payload,
    evaluate_predictions,
    prepare_dataset,
    refresh_runtime_metadata,
    run_chemprop,
)


def _pairs() -> pd.DataFrame:
    rows = []
    species = [
        ("Species alpha", 1, "gram-positive"),
        ("Species beta", 2, "gram-negative"),
        ("Species gamma", 3, "acid-fast"),
    ]
    for compound in range(30):
        for offset, (organism, tax_id, pathogen_class) in enumerate(species):
            if (compound + offset) % 4 == 0:
                continue
            rows.append(
                {
                    "compound_inchikey": f"KEY-{compound:03d}",
                    "canonical_smiles": "C" * (compound + 1),
                    "organism": organism,
                    "tax_id": tax_id,
                    "pathogen_class": pathogen_class,
                    "log2_mic": float(compound % 7 + offset),
                }
            )
    return pd.DataFrame(rows)


def test_prepare_and_command_preserve_compound_splits(tmp_path):
    source = tmp_path / "pairs.csv"
    _pairs().to_csv(source, index=False)
    manifest = prepare_dataset(source, tmp_path / "prepared", seed=42)
    wide = pd.read_csv(manifest["dataset"])

    assert len(wide) == 30
    assert manifest["species_tasks"] == 3
    assert set(wide["split"]) == {"train", "val", "test"}
    assert len(manifest["split_fingerprint_sha256"]) == 64
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["checkpoint_selection"]["tracking_metric"].startswith("standard Chemprop val/mae")
    assert np.isclose(np.mean(manifest["task_weights"]), 1.0)
    assert wide[manifest["target_columns"]].isna().any().any()

    payload = command_payload(
        manifest["dataset"],
        tmp_path / "prepared" / "chemprop_manifest.json",
        tmp_path / "run",
        epochs=1,
        batch_size=8,
        accelerator="cpu",
        devices="1",
        warmup_epochs=0,
        patience=1,
    )
    train = payload["train_command"]
    assert "--task-weights" in train
    assert "--splits-column" in train
    assert train[train.index("--splits-column") + 1] == "split"
    assert train[train.index("--tracking-metric") + 1] == "mae"
    assert payload["prediction_compound_count_expected"] == 30
    assert payload["prediction_task_count"] == 3


def test_evaluate_sparse_multitask_predictions(tmp_path):
    source = tmp_path / "pairs.csv"
    _pairs().to_csv(source, index=False)
    prepared_dir = tmp_path / "prepared"
    manifest = prepare_dataset(source, prepared_dir, seed=100)
    prepared = pd.read_csv(manifest["dataset"])
    predicted = prepared.copy()
    for column in manifest["target_columns"]:
        predicted[column] = predicted[column].fillna(0.0) + 0.25
    prediction_path = tmp_path / "predictions.csv"
    predicted.to_csv(prediction_path, index=False)

    result = evaluate_predictions(
        manifest["dataset"],
        prediction_path,
        prepared_dir / "chemprop_manifest.json",
        tmp_path / "evaluation",
    )
    metrics = pd.read_csv(tmp_path / "evaluation" / "metrics.csv")
    macro = metrics.loc[metrics["scope"].eq("macro")].iloc[0]

    assert result["status"] == "complete"
    assert result["test_species"] == 3
    assert result["point_metrics_only"] is True
    assert result["validation_reused_for_checkpoint_selection"] is True
    assert result["uncertainty_evaluation"] == "not_evaluated"
    assert np.isclose(macro["mae"], 0.25)
    assert not any("conformal" in column for column in metrics.columns)
    output_predictions = pd.read_csv(tmp_path / "evaluation" / "predictions.csv.gz")
    assert not any("conformal" in column or column.startswith(("lower_", "upper_")) for column in output_predictions)
    assert json.loads((tmp_path / "evaluation" / "evaluation_manifest.json").read_text())["seed"] == 100


def test_evaluate_rejects_non_finite_predictions(tmp_path):
    source = tmp_path / "pairs.csv"
    _pairs().to_csv(source, index=False)
    prepared_dir = tmp_path / "prepared"
    manifest = prepare_dataset(source, prepared_dir, seed=2026)
    predicted = pd.read_csv(manifest["dataset"])
    for column in manifest["target_columns"]:
        predicted[column] = predicted[column].fillna(0.0)
    predicted.loc[0, manifest["target_columns"][0]] = np.inf
    prediction_path = tmp_path / "predictions.csv"
    predicted.to_csv(prediction_path, index=False)

    with pytest.raises(ValueError, match="non-finite"):
        evaluate_predictions(
            manifest["dataset"],
            prediction_path,
            prepared_dir / "chemprop_manifest.json",
            tmp_path / "evaluation",
        )


def test_runtime_separates_deployable_and_training_artifacts(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    train_dir = output_dir / "train" / "model_0"
    checkpoint_dir = train_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (train_dir / "best.pt").write_bytes(b"best-model")
    (checkpoint_dir / "epoch-1.ckpt").write_bytes(b"checkpoint-one")
    (checkpoint_dir / "epoch-2.ckpt").write_bytes(b"checkpoint-two")
    predictions = output_dir / "chemprop_predictions.csv"
    pd.DataFrame({"smiles": ["C", "CC"]}).to_csv(predictions, index=False)
    monkeypatch.setattr(chemprop_module, "_run_logged", lambda command, log_path: 1.25)
    payload = {
        "train_command": ["train"],
        "predict_command": ["predict"],
        "predictions": str(predictions),
        "prediction_scope": "all_prepared_compounds_all_species_tasks",
        "prediction_compound_count_expected": 2,
        "prediction_task_count": 3,
    }

    runtime = run_chemprop(payload, output_dir)

    assert runtime["status"] == "complete"
    assert runtime["deployable_best_pt_bytes"] == len(b"best-model")
    assert runtime["training_checkpoint_bytes"] == len(b"checkpoint-one") + len(b"checkpoint-two")
    assert runtime["total_model_artifact_bytes"] == (
        len(b"best-model") + len(b"checkpoint-one") + len(b"checkpoint-two")
    )
    assert runtime["prediction_compound_count"] == 2
    assert runtime["prediction_task_count"] == 3
    assert "model_bytes" not in runtime


def test_refresh_runtime_metadata_repairs_legacy_schema(tmp_path):
    output_dir = tmp_path / "run"
    train_dir = output_dir / "train" / "model_0"
    checkpoint_dir = train_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (train_dir / "best.pt").write_bytes(b"best-model")
    (checkpoint_dir / "best.ckpt").write_bytes(b"checkpoint")
    pd.DataFrame({"smiles": ["C", "CC"]}).to_csv(output_dir / "chemprop_predictions.csv", index=False)
    (output_dir / "runtime.json").write_text(
        json.dumps({"status": "complete", "train_seconds": 2.0, "predict_seconds": 1.0, "model_bytes": 99}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"prepared_compounds": 2, "species_tasks": 3}), encoding="utf-8")

    runtime = refresh_runtime_metadata(manifest_path, output_dir)

    assert runtime["train_seconds"] == 2.0
    assert runtime["deployable_best_pt_bytes"] == len(b"best-model")
    assert runtime["training_checkpoint_bytes"] == len(b"checkpoint")
    assert runtime["prediction_compound_count"] == 2
    assert "model_bytes" not in runtime
