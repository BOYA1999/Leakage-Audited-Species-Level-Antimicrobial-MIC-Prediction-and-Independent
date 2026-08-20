import numpy as np
import pandas as pd

from amr_multiview.features import GRAPH_FEATURES, PHYS_DESCRIPTORS
from amr_multiview.species_mic_experiment import _finite_sample_quantile, _metrics, scaffold_partition


def test_metrics_use_log2_dilution_steps():
    result = _metrics(np.array([0.0, 2.0, 4.0]), np.array([1.0, 2.0, 2.0]))
    assert result["mae"] == 1.0
    assert result["within_one_dilution"] == 2 / 3
    assert result["within_two_dilutions"] == 1.0


def test_scaffold_groups_do_not_overlap():
    compounds = pd.DataFrame(
        {
            "compound_inchikey": [f"KEY{i:02d}" for i in range(12)],
            "canonical_smiles": [
                "c1ccccc1C",
                "c1ccccc1O",
                "c1ccncc1C",
                "c1ccncc1O",
                "C1CCCCC1C",
                "C1CCCCC1O",
                "c1ccc2ccccc2c1",
                "c1ccc2ccccc2c1O",
                "CCO",
                "CCN",
                "CCCC",
                "CCCO",
            ],
        }
    )
    partition, report = scaffold_partition(compounds, seed=42)
    assert set(partition) <= {"train", "valid", "test"}
    assert report["scaffold_overlap_train_valid"] == 0
    assert report["scaffold_overlap_train_test"] == 0
    assert report["scaffold_overlap_valid_test"] == 0
    assert partition[0] == partition[1]
    assert partition[2] == partition[3]


def test_finite_sample_conformal_quantile():
    residual = np.arange(1, 11, dtype=float)
    assert _finite_sample_quantile(residual, coverage=0.80) == 9.0


def test_handcrafted_feature_inventory_is_frozen():
    assert len(PHYS_DESCRIPTORS) == 12
    assert len(GRAPH_FEATURES) == 15
    assert len(set(PHYS_DESCRIPTORS + GRAPH_FEATURES)) == 27
