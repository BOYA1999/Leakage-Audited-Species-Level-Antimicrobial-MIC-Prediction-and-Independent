import numpy as np
import pandas as pd

from scripts.analyze_species_mic_results import METRICS, coverage_audit, full_vs_maccs_audit


def test_equal_compound_coverage_is_not_pair_weighted(tmp_path):
    rows = []
    for seed in range(5):
        for compound, organism, outcome in [('A', 'X', 0), ('A', 'Y', 0), ('B', 'X', 1)]:
            rows.append(dict(model='morgan', seed=seed, compound_inchikey=compound,
                             organism=organism, y_true=outcome, lower_90=0, upper_90=0))
    pd.DataFrame(rows).to_csv(tmp_path / 'predictions.csv.gz', index=False)
    coverage_audit(tmp_path, tmp_path)
    result = pd.read_csv(tmp_path / 'coverage_weighting_summary.csv').iloc[0]
    assert np.isclose(result.pair_weighted_mean, 2 / 3)
    assert np.isclose(result.equal_compound_mean, 0.5)
    assert result.seeds == 5


def test_full_vs_reduced_comparison_pairs_seeds(tmp_path):
    rows = []
    differences = [-0.01, 0.01, -0.005, 0.005, 0.0]
    for seed, difference in enumerate(differences):
        for model, offset in [('multiview', difference), ('morgan_maccs', 0)]:
            rows.append(dict(scope='macro', seed=seed, model=model,
                             **{metric: 1 + seed / 10 + offset for metric in METRICS[:5]}))
    pd.DataFrame(rows[::-1]).to_csv(tmp_path / 'metrics.csv', index=False)
    full_vs_maccs_audit(tmp_path, tmp_path)
    result = pd.read_csv(tmp_path / 'full_vs_maccs_paired.csv')
    assert len(result) == 5
    assert np.allclose(result.mean_difference, 0)
    assert (result.ci_low < 0).all() and (result.ci_high > 0).all()
    assert (result.favorable_splits == 2).all()
