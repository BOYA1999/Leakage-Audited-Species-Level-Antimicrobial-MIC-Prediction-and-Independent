from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from amr_multiview.species_mic_jcim_baselines import (
    _validate_species_mic_input,
    _verify_xgb_device,
    run_experiment,
    split_fingerprint,
)


def _species_mic_frame(n_compounds=30):
    rows = []
    species = [("Species alpha", 101, "gram-positive"), ("Species beta", 202, "gram-negative")]
    for index in range(1, n_compounds + 1):
        for species_index, (organism, tax_id, pathogen_class) in enumerate(species):
            rows.append(
                {
                    "compound_inchikey": f"KEY{index:04d}",
                    "canonical_smiles": "C" * index,
                    "organism": organism,
                    "tax_id": tax_id,
                    "pathogen_class": pathogen_class,
                    "log2_mic": float(np.log2(index + 1) + species_index),
                }
            )
    return pd.DataFrame(rows)


def test_split_fingerprint_is_order_independent_and_assignment_sensitive():
    keys = np.array(["B", "A", "C"])
    partition = np.array(["train", "valid", "test"])
    assert split_fingerprint(keys, partition) == split_fingerprint(keys[::-1], partition[::-1])
    assert split_fingerprint(keys, partition) != split_fingerprint(keys, np.array(["train", "test", "valid"]))


def test_broad_five_class_input_is_rejected():
    frame = _species_mic_frame(3)
    frame["best_class"] = "gram-positive"
    with pytest.raises(ValueError, match="five-class"):
        _validate_species_mic_input(frame)


def test_cuda_request_cannot_silently_fall_back_to_cpu():
    class Booster:
        @staticmethod
        def save_config():
            return '{"learner":{"generic_param":{"device":"cpu"}}}'

    class Model:
        @staticmethod
        def get_booster():
            return Booster()

    with pytest.raises(RuntimeError, match="no fallback"):
        _verify_xgb_device(Model(), "cuda")


def test_small_matched_estimator_smoke_writes_complete_contract(tmp_path):
    data = tmp_path / "species_mic.csv"
    output = tmp_path / "run"
    _species_mic_frame().to_csv(data, index=False)
    args = SimpleNamespace(
        data=str(data),
        output_dir=str(output),
        models=["lightgbm", "rf", "xgboost"],
        representations=["morgan"],
        seeds=[42],
        n_estimators=2,
        n_jobs=1,
        xgb_device="cpu",
        max_compounds=20,
        command=["pytest-smoke"],
    )

    manifest = run_experiment(args)

    expected = {
        "metrics.csv",
        "predictions.csv.gz",
        "metrics_summary.csv",
        "runtime.csv",
        "split_fingerprints.csv",
        "metric_contract.json",
        "manifest.json",
    }
    assert expected == {path.name for path in output.iterdir()}
    assert manifest["dataset_kind"] == "species_mic_regression"
    assert manifest["compound_limit_applied"] is True
    assert manifest["compounds"] == 20
    assert manifest["species"] == 2

    runtime = pd.read_csv(output / "runtime.csv")
    assert set(runtime["model"]) == {"lightgbm_morgan", "rf_morgan", "xgboost_morgan"}
    assert runtime["fit_seconds"].gt(0).all()
    assert runtime["predict_seconds"].ge(0).all()
    assert runtime["serialized_model_bytes"].gt(0).all()
    assert runtime.loc[runtime["estimator"].eq("xgboost"), "actual_device"].eq("cpu").all()

    predictions = pd.read_csv(output / "predictions.csv.gz")
    assert set(predictions["model"]) == {"lightgbm_morgan", "rf_morgan", "xgboost_morgan"}
    assert "best_class" not in predictions.columns
    fingerprints = pd.read_csv(output / "split_fingerprints.csv")
    assert fingerprints["split_sha256"].str.len().eq(64).all()
