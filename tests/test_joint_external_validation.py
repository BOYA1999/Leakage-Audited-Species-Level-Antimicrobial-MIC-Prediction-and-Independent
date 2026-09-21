import numpy as np
import pandas as pd
import pytest

from amr_multiview.joint_external_validation import check_replay, metric_pair, primary_mask


def test_joint_primary_excludes_any_species_overlap():
    frame = pd.DataFrame({'exact_training_overlap': [False, False, True],
                          'joint_corpus_exact_overlap': [False, True, False]})
    assert primary_mask(frame).tolist() == [True, False, False]


def test_replay_aligns_keys_and_rejects_prediction_or_membership_drift():
    expected = pd.DataFrame({'compound_inchikey': ['synthetic-a', 'synthetic-b'], 'organism': ['A', 'B'],
                             'y_true': [1.0, 2.0], 'y_pred': [1.2, 1.9], 'upper_90': [3.2, 3.9]})
    assert check_replay(expected, expected.iloc[::-1], 1e-6)['y_pred'] == 0
    changed = expected.copy()
    changed.loc[0, 'y_pred'] += 0.1
    with pytest.raises(ValueError, match='prediction gate'):
        check_replay(expected, changed, 1e-6)
    with pytest.raises(ValueError, match='membership'):
        check_replay(expected, expected.iloc[:1], 1e-6)


def test_single_class_metrics_are_not_estimable():
    assert np.isnan(metric_pair(np.ones(4), np.arange(4))).all()
    np.testing.assert_allclose(metric_pair([0, 1], [0.1, 0.9]), [1, 1])
