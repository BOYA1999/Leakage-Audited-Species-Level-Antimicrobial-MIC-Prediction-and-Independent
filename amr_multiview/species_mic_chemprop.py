from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .species_mic_experiment import _metrics, scaffold_partition


def prepare_dataset(
    data_path: str | Path,
    output_dir: str | Path,
    seed: int,
    max_compounds: int | None = None,
) -> dict:
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(data_path)
    full_source_rows = len(pairs)
    required = {
        "compound_inchikey",
        "canonical_smiles",
        "organism",
        "tax_id",
        "pathogen_class",
        "log2_mic",
    }
    missing = required - set(pairs)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if pairs.duplicated(["compound_inchikey", "organism"]).any():
        raise ValueError("Input must contain one aggregated row per compound-species pair")

    smiles_counts = pairs.groupby("compound_inchikey")["canonical_smiles"].nunique()
    if (smiles_counts != 1).any():
        raise ValueError("Each compound_inchikey must map to exactly one canonical_smiles")
    compounds = pairs[["compound_inchikey", "canonical_smiles"]].drop_duplicates().reset_index(drop=True)
    partition, split_report = scaffold_partition(compounds, seed)
    selected = _select_compounds(partition, max_compounds, seed)
    compounds = compounds.loc[selected].reset_index(drop=True)
    selected_partition = partition[selected]
    selected_partition = np.where(selected_partition == "valid", "val", selected_partition)
    split_by_key = dict(zip(compounds["compound_inchikey"], selected_partition, strict=True))
    pairs = pairs[pairs["compound_inchikey"].isin(split_by_key)].copy()

    species_meta = (
        pairs[["organism", "tax_id", "pathogen_class"]]
        .drop_duplicates()
        .sort_values("organism")
        .reset_index(drop=True)
    )
    if species_meta["organism"].duplicated().any():
        raise ValueError("An organism maps to more than one tax_id or pathogen_class")

    pivot = pairs.pivot(index="compound_inchikey", columns="organism", values="log2_mic")
    wide = compounds.rename(columns={"canonical_smiles": "smiles"}).set_index("compound_inchikey")
    wide["split"] = pd.Series(split_by_key)
    tasks = []
    for index, row in species_meta.iterrows():
        target = f"target_{index:03d}"
        wide[target] = pivot.reindex(wide.index)[row["organism"]]
        tasks.append(
            {
                "target_column": target,
                "organism": row["organism"],
                "tax_id": int(row["tax_id"]),
                "pathogen_class": row["pathogen_class"],
            }
        )
    wide = wide.reset_index()

    target_columns = [task["target_column"] for task in tasks]
    train_counts = wide.loc[wide["split"].eq("train"), target_columns].notna().sum()
    missing_train = train_counts[train_counts.eq(0)].index.tolist()
    if missing_train:
        raise ValueError(f"Tasks without training labels: {missing_train}")
    inverse = 1.0 / train_counts.to_numpy(dtype=float)
    task_weights = inverse / inverse.mean()
    for task, count, weight in zip(tasks, train_counts, task_weights, strict=True):
        task["train_label_count"] = int(count)
        task["task_weight"] = float(weight)

    dataset_path = output_dir / "chemprop_wide.csv"
    wide.to_csv(dataset_path, index=False)
    pd.DataFrame(tasks).to_csv(output_dir / "species_map.csv", index=False)
    split_fingerprint = _split_fingerprint(wide[["compound_inchikey", "split"]])
    manifest = {
        "schema_version": "1.0",
        "task": "species_mic_regression",
        "model": "chemprop_2.2.4_dmpnn_sparse_multitask",
        "organism_context": "one_species_specific_output_task_per_target_column",
        "source": str(data_path.resolve()),
        "source_sha256": _sha256(data_path),
        "dataset": str(dataset_path.resolve()),
        "seed": int(seed),
        "max_compounds": max_compounds,
        "smoke_only": max_compounds is not None,
        "full_source_rows": int(full_source_rows),
        "full_source_compounds": int(len(partition)),
        "prepared_compounds": int(len(wide)),
        "prepared_labels": int(wide[target_columns].notna().sum().sum()),
        "species_tasks": int(len(tasks)),
        "split_counts": {str(k): int(v) for k, v in wide["split"].value_counts().sort_index().items()},
        "split_fingerprint_sha256": split_fingerprint,
        "full_split_report": split_report,
        "tasks": tasks,
        "target_columns": target_columns,
        "task_weights": [float(value) for value in task_weights],
        "dataset_sha256": _sha256(dataset_path),
        "checkpoint_selection": {
            "validation_split": "explicit val split inherited from the scaffold partition",
            "tracking_metric": "standard Chemprop val/mae on train-standardized targets",
            "selected_checkpoint": "best checkpoint by validation MAE with patience-based early stopping",
            "uncertainty_evaluation": "not_evaluated; validation is reserved for checkpoint selection",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "chemprop": importlib.metadata.version("chemprop"),
        },
    }
    _write_json(output_dir / "chemprop_manifest.json", manifest)
    return manifest


def command_payload(
    dataset: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    epochs: int,
    batch_size: int,
    accelerator: str,
    devices: str,
    warmup_epochs: int,
    patience: int,
) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    targets = manifest["target_columns"]
    weights = [f"{value:.12g}" for value in manifest["task_weights"]]
    seed = str(manifest["seed"])
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    predictions = output_dir / "chemprop_predictions.csv"
    train = [
        "chemprop",
        "train",
        "-q",
        "-i",
        str(Path(dataset)),
        "-o",
        str(train_dir),
        "-s",
        "smiles",
        "--target-columns",
        *targets,
        "--ignore-columns",
        "compound_inchikey",
        "split",
        "--splits-column",
        "split",
        "--task-type",
        "regression",
        "--loss-function",
        "mae",
        "--metrics",
        "mae",
        "rmse",
        "--tracking-metric",
        "mae",
        "--task-weights",
        *weights,
        "--epochs",
        str(epochs),
        "--warmup-epochs",
        str(min(warmup_epochs, max(0, epochs - 1))),
        "--patience",
        str(patience),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        "0",
        "--accelerator",
        accelerator,
        "--devices",
        devices,
        "--message-hidden-dim",
        "300",
        "--ffn-hidden-dim",
        "300",
        "--depth",
        "3",
        "--dropout",
        "0.1",
        "--data-seed",
        seed,
        "--pytorch-seed",
        seed,
        "--save-data-splits",
    ]
    predict = [
        "chemprop",
        "predict",
        "-q",
        "-i",
        str(Path(dataset)),
        "-o",
        str(predictions),
        "-s",
        "smiles",
        "--model-paths",
        str(train_dir),
        "--num-workers",
        "0",
        "--accelerator",
        accelerator,
        "--devices",
        devices,
    ]
    return {
        "train_command": train,
        "predict_command": predict,
        "predictions": str(predictions),
        "prediction_scope": "all_prepared_compounds_all_species_tasks",
        "prediction_compound_count_expected": int(manifest["prepared_compounds"]),
        "prediction_task_count": int(manifest["species_tasks"]),
    }


def write_commands(payload: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "chemprop_commands.json", payload)
    (output_dir / "train_command.ps1").write_text(_ps_script(payload["train_command"]), encoding="utf-8")
    (output_dir / "predict_command.ps1").write_text(_ps_script(payload["predict_command"]), encoding="utf-8")


def run_chemprop(payload: dict, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = {"status": "running"}
    try:
        runtime["train_seconds"] = _run_logged(payload["train_command"], output_dir / "train.log")
        runtime["predict_seconds"] = _run_logged(payload["predict_command"], output_dir / "predict.log")
        artifacts = [path for path in (output_dir / "train").rglob("*") if path.suffix in {".pt", ".ckpt"}]
        deployable = [path for path in artifacts if path.name == "best.pt"]
        training_checkpoints = [path for path in artifacts if path.suffix == ".ckpt"]
        prediction_compounds = len(pd.read_csv(payload["predictions"], usecols=["smiles"]))
        expected_compounds = int(payload["prediction_compound_count_expected"])
        if prediction_compounds != expected_compounds:
            raise ValueError(
                f"Prediction compound count differs from prepared data: {prediction_compounds} != {expected_compounds}"
            )
        runtime.update(
            {
                "deployable_best_pt_count": len(deployable),
                "deployable_best_pt_bytes": int(sum(path.stat().st_size for path in deployable)),
                "training_checkpoint_count": len(training_checkpoints),
                "training_checkpoint_bytes": int(sum(path.stat().st_size for path in training_checkpoints)),
                "total_model_artifact_count": len(artifacts),
                "total_model_artifact_bytes": int(sum(path.stat().st_size for path in artifacts)),
                "prediction_scope": payload["prediction_scope"],
                "prediction_compound_count": int(prediction_compounds),
                "prediction_task_count": int(payload["prediction_task_count"]),
            }
        )
        runtime["status"] = "complete"
    except Exception:
        runtime["status"] = "failed"
        _write_json(output_dir / "runtime.json", runtime)
        raise
    _write_json(output_dir / "runtime.json", runtime)
    return runtime


def refresh_runtime_metadata(manifest_path: str | Path, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    runtime_path = output_dir / "runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    artifacts = [path for path in (output_dir / "train").rglob("*") if path.suffix in {".pt", ".ckpt"}]
    deployable = [path for path in artifacts if path.name == "best.pt"]
    training_checkpoints = [path for path in artifacts if path.suffix == ".ckpt"]
    predictions = output_dir / "chemprop_predictions.csv"
    prediction_compounds = len(pd.read_csv(predictions, usecols=["smiles"]))
    expected_compounds = int(manifest["prepared_compounds"])
    if prediction_compounds != expected_compounds:
        raise ValueError(
            f"Prediction compound count differs from prepared data: {prediction_compounds} != {expected_compounds}"
        )
    runtime.pop("model_bytes", None)
    runtime.pop("checkpoint_count", None)
    runtime.update(
        {
            "deployable_best_pt_count": len(deployable),
            "deployable_best_pt_bytes": int(sum(path.stat().st_size for path in deployable)),
            "training_checkpoint_count": len(training_checkpoints),
            "training_checkpoint_bytes": int(sum(path.stat().st_size for path in training_checkpoints)),
            "total_model_artifact_count": len(artifacts),
            "total_model_artifact_bytes": int(sum(path.stat().st_size for path in artifacts)),
            "prediction_scope": "all_prepared_compounds_all_species_tasks",
            "prediction_compound_count": int(prediction_compounds),
            "prediction_task_count": int(manifest["species_tasks"]),
        }
    )
    _write_json(runtime_path, runtime)
    return runtime


def evaluate_predictions(
    prepared_path: str | Path,
    prediction_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = pd.read_csv(prepared_path)
    predicted = pd.read_csv(prediction_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if len(prepared) != len(predicted):
        raise ValueError("Prepared and prediction row counts differ")
    if "compound_inchikey" in predicted and not prepared["compound_inchikey"].equals(predicted["compound_inchikey"]):
        raise ValueError("Prediction rows do not match prepared compound order")
    missing = set(manifest["target_columns"]) - set(predicted)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    prediction_values = predicted[manifest["target_columns"]].to_numpy(dtype=float)
    if not np.isfinite(prediction_values).all():
        raise ValueError("Prediction target columns contain non-finite values")

    validation = _long_predictions(prepared, predicted, manifest, "val")
    test = _long_predictions(prepared, predicted, manifest, "test")
    if validation.empty or test.empty:
        raise ValueError("Validation and test observations are required")
    test["model"] = "chemprop_dmpnn"
    test["seed"] = manifest["seed"]

    rows = [_metric_row(test, "pooled", "__pooled__")]
    rows.extend(_metric_row(group, "species", organism) for organism, group in test.groupby("organism", sort=False))
    species_rows = pd.DataFrame(rows[1:])
    macro = {
        key: float(species_rows[key].mean())
        for key in [
            "mae",
            "rmse",
            "spearman",
            "within_one_dilution",
            "within_two_dilutions",
        ]
    }
    macro.update(
        {
            "model": "chemprop_dmpnn",
            "seed": manifest["seed"],
            "scope": "macro",
            "organism": "__macro__",
            "n": int(len(test)),
        }
    )
    rows.append(macro)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    test.to_csv(output_dir / "predictions.csv.gz", index=False, compression="gzip")
    metrics[metrics["scope"].isin(["pooled", "macro"])].to_csv(output_dir / "metrics_summary.csv", index=False)
    evaluation = {
        "status": "complete",
        "seed": int(manifest["seed"]),
        "validation_labels": int(len(validation)),
        "test_labels": int(len(test)),
        "test_species": int(test["organism"].nunique()),
        "point_metrics_only": True,
        "validation_reused_for_checkpoint_selection": True,
        "validation_role": "standard Chemprop validation metric and checkpoint selection",
        "uncertainty_evaluation": "not_evaluated",
        "uncertainty_reason": (
            "The validation split was reused for checkpoint selection and therefore was not treated as an "
            "independent conformal calibration set."
        ),
    }
    _write_json(output_dir / "evaluation_manifest.json", evaluation)
    return evaluation


def _long_predictions(prepared: pd.DataFrame, predicted: pd.DataFrame, manifest: dict, split: str) -> pd.DataFrame:
    mask = prepared["split"].eq(split)
    frames = []
    for task in manifest["tasks"]:
        column = task["target_column"]
        observed = mask & prepared[column].notna()
        if not observed.any():
            continue
        frame = prepared.loc[observed, ["compound_inchikey", "smiles"]].copy()
        frame["organism"] = task["organism"]
        frame["tax_id"] = task["tax_id"]
        frame["pathogen_class"] = task["pathogen_class"]
        frame["y_true"] = prepared.loc[observed, column].to_numpy(dtype=float)
        frame["y_pred"] = predicted.loc[observed, column].to_numpy(dtype=float)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _metric_row(frame: pd.DataFrame, scope: str, organism: str) -> dict:
    row = _metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
    row.update(
        {
            "model": "chemprop_dmpnn",
            "seed": int(frame["seed"].iloc[0]),
            "scope": scope,
            "organism": organism,
            "n": int(len(frame)),
        }
    )
    return row


def _select_compounds(partition: np.ndarray, max_compounds: int | None, seed: int) -> np.ndarray:
    if max_compounds is None or max_compounds >= len(partition):
        return np.arange(len(partition))
    if max_compounds < 3:
        raise ValueError("max_compounds must be at least three")
    rng = np.random.default_rng(seed)
    quotas = {"train": round(max_compounds * 0.70), "val": round(max_compounds * 0.15)}
    quotas["test"] = max_compounds - quotas["train"] - quotas["val"]
    selected = []
    for split, quota in quotas.items():
        source_split = "valid" if split == "val" else split
        candidates = np.flatnonzero(partition == source_split)
        selected.extend(rng.choice(candidates, min(quota, len(candidates)), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def _split_fingerprint(frame: pd.DataFrame) -> str:
    content = "\n".join(
        f"{row.compound_inchikey}\t{row.split}"
        for row in frame.sort_values("compound_inchikey").itertuples(index=False)
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_logged(command: list[str], log_path: Path) -> float:
    started = time.perf_counter()
    environment = os.environ.copy()
    environment.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=environment)
    return float(time.perf_counter() - started)


def _ps_script(command: list[str]) -> str:
    quoted = []
    for value in command:
        quoted.append("'" + value.replace("'", "''") + "'" if any(char in value for char in " \t;") else value)
    return "\n".join(
        [
            "$env:PYTHONUTF8='1'",
            "$env:PYTHONIOENCODING='utf-8'",
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8",
            " ".join(quoted),
            "",
        ]
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare, run, and evaluate sparse multitask Chemprop MIC models.")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--data", required=True)
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--seed", type=int, required=True)
    prep.add_argument("--max-compounds", type=int)

    for name in ["write-command", "run"]:
        command = sub.add_parser(name)
        command.add_argument("--dataset", required=True)
        command.add_argument("--manifest", required=True)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--epochs", type=int, default=50)
        command.add_argument("--batch-size", type=int, default=256)
        command.add_argument("--accelerator", default="gpu")
        command.add_argument("--devices", default="1")
        command.add_argument("--warmup-epochs", type=int, default=3)
        command.add_argument("--patience", type=int, default=5)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--prepared", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output-dir", required=True)

    refresh = sub.add_parser("refresh-runtime")
    refresh.add_argument("--manifest", required=True)
    refresh.add_argument("--output-dir", required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_dataset(args.data, args.output_dir, args.seed, args.max_compounds)
    elif args.command in {"write-command", "run"}:
        result = command_payload(
            args.dataset,
            args.manifest,
            args.output_dir,
            args.epochs,
            args.batch_size,
            args.accelerator,
            args.devices,
            args.warmup_epochs,
            args.patience,
        )
        write_commands(result, args.output_dir)
        if args.command == "run":
            result = run_chemprop(result, args.output_dir)
    elif args.command == "evaluate":
        result = evaluate_predictions(args.prepared, args.predictions, args.manifest, args.output_dir)
    else:
        result = refresh_runtime_metadata(args.manifest, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
