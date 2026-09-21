import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amr_multiview.species_mic_experiment import _metrics
parser = argparse.ArgumentParser(description='Replay the two fixed-prediction CBC controls.')
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
DATA = args.data
inputs = [DATA / 'identifiers/temporal_pairs.csv.gz', DATA / 'identifiers/temporal_predictions.csv.gz',
          DATA / 'tables/jcim_temporal_metrics.csv', DATA / 'identifiers/joint_mean_scores.csv.gz',
          DATA / 'identifiers/joint_overlap_flags.csv.gz', DATA / 'evidence/external_labels_mw.csv.gz',
          DATA / 'tables/joint_external_metrics.csv']
manifest = {'seed': 20260912, 'bootstrap': 5000, 'inputs': {
    str(p.relative_to(DATA)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
keys = ['compound_inchikey', 'organism']
pairs = pd.read_csv(inputs[0], float_precision='round_trip')
train = pairs[pairs.cohort.eq('train')]
test = pairs[pairs.cohort.eq('test')].copy()
assert len(train) == 179208 and len(test) == 13999
assert not set(train.compound_inchikey) & set(test.compound_inchikey)
medians = train.groupby('organism').log2_mic.median()
test['y_true'] = test.log2_mic
test['y_pred'] = test.organism.map(medians)
assert np.isfinite(test[['y_true', 'y_pred']]).all().all()
saved_pred = pd.read_csv(inputs[1], float_precision='round_trip')
saved_metrics = pd.read_csv(inputs[2])
rows, replay = [], []
for model, frame in [('species_median', test), *list(saved_pred.groupby('model'))]:
    match = frame[keys + ['y_true']].merge(test[keys + ['log2_mic']], on=keys, validate='one_to_one')
    assert len(match) == 13999
    assert np.allclose(match.y_true, match.log2_mic, atol=1e-6, rtol=0)
    species = []
    for organism, group in frame.groupby('organism', sort=True):
        species.append({'model': model, 'scope': 'species', 'organism': organism,
                        'n_pairs': len(group), **_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())})
    metric_cols = list(_metrics(test.y_true.to_numpy(), test.y_pred.to_numpy()))
    pooled = {'model': model, 'scope': 'pooled', 'organism': '__pooled__', 'n_pairs': len(frame),
              **_metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy())}
    macro = {'model': model, 'scope': 'macro', 'organism': '__macro__', 'n_pairs': len(frame),
             **pd.DataFrame(species)[metric_cols].mean().to_dict()}
    observed = pd.DataFrame([*species, pooled, macro])
    if model != 'species_median':
        merged = observed.merge(saved_metrics[saved_metrics.model.eq(model)], on=['model', 'scope', 'organism'], suffixes=('_new', '_old'), validate='one_to_one')
        deltas = {c: float((merged[c + '_new'] - merged[c + '_old']).abs().max()) for c in metric_cols}
        delta = max(deltas.values())
        print(model, deltas, flush=True)
        assert delta < 1e-7, merged.loc[(merged.rmse_new - merged.rmse_old).abs().nlargest(3).index].to_string()
        replay.append({'model': model, 'maximum_metric_delta': delta})
    rows.extend(observed.to_dict('records'))
pd.DataFrame(rows).to_csv(OUT / 'temporal_with_median.csv', index=False)
test[keys + ['tax_id', 'y_true', 'y_pred']].to_csv(OUT / 'temporal_median_predictions.csv.gz', index=False)
medians.rename('training_median_log2_mic').to_csv(OUT / 'temporal_training_medians.csv')
manifest['temporal_replay'] = replay
print(pd.DataFrame(rows).query("scope == 'macro'").to_string(index=False), flush=True)

scores = pd.read_csv(inputs[3])
audit = pd.read_csv(inputs[4]).set_index('prestwick_ID')
mapping = pd.read_csv(inputs[5])
reference = pd.read_csv(inputs[6])
def rank_metrics(y, x):
    return np.array([roc_auc_score(y, x), average_precision_score(y, x)])

rng = np.random.default_rng(20260912)
contrasts, marginal, labels_export = [], [], []
for organism, group in mapping.groupby('organism', sort=True):
    external = group.loc[group['label'].notna()].set_index('prestwick_ID')
    labels = external.label.astype(int)
    wide = scores[scores.organism.eq(organism) & scores.family.eq('joint')].pivot(index='prestwick_ID', columns='representation', values='score')
    wide = wide.reindex(external.index)
    wide['molecular_weight_only'] = external.molecular_weight_only
    assert np.isfinite(wide).all().all()
    labels_export.append(pd.DataFrame({'prestwick_ID': external.index, 'organism': organism, 'label': labels.to_numpy(), 'molecular_weight_only': wide.molecular_weight_only.to_numpy()}))
    for cohort in ['primary', 'fingerprint_nonidentity', 'parent_nonidentity']:
        ids = audit.index[audit[cohort]].intersection(external.index, sort=False)
        y, matrix = labels.loc[ids].to_numpy(), wide.loc[ids].to_numpy()
        points = np.array([rank_metrics(y, matrix[:, j]) for j in range(matrix.shape[1])])
        ref = reference[reference.organism.eq(organism) & reference.cohort.eq(cohort) & reference.endpoint.str.endswith(' any strain')].set_index('model')
        for j, model in enumerate(wide.columns):
            name = 'concentration_only' if model == 'molecular_weight_only' else 'joint:' + model
            assert np.allclose(points[j], ref.loc[name, ['auroc', 'ap']].to_numpy(dtype=float), atol=1e-12, rtol=0)
            marginal.append({'organism': organism, 'cohort': cohort, 'model': model, 'n': len(y), 'positives': int(y.sum()), 'auroc': points[j, 0], 'ap': points[j, 1]})
        draws = np.empty((5000, matrix.shape[1], 2))
        valid, rejected = 0, 0
        while valid < 5000:
            idx = rng.integers(0, len(y), len(y))
            if np.unique(y[idx]).size != 2:
                rejected += 1
                continue
            for j in range(matrix.shape[1]):
                draws[valid, j] = rank_metrics(y[idx], matrix[idx, j])
            valid += 1
        for j, model in enumerate(wide.columns[:-1]):
            limits = np.quantile(draws[:, j] - draws[:, -1], [0.025, 0.975], axis=0)
            for k, metric in enumerate(['auroc', 'ap']):
                contrasts.append({'organism': organism, 'cohort': cohort, 'model': model, 'metric': metric,
                                  'difference': points[j, k] - points[-1, k], 'ci_low': limits[0, k], 'ci_high': limits[1, k],
                                  'n': len(y), 'positives': int(y.sum()), 'bootstrap': valid, 'rejected': rejected, 'seed': 20260912})
        pd.DataFrame(contrasts).to_csv(OUT / 'joint_minus_molecular_weight.csv', index=False)
        print('Completed paired bootstrap:', organism, cohort, flush=True)
pd.DataFrame(marginal).to_csv(OUT / 'joint_marginal_replay.csv', index=False)
pd.concat(labels_export).to_csv(OUT / 'external_labels_mw.csv.gz', index=False)
manifest['joint_marginals_reproduced'] = len(marginal)
(OUT / 'controls_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
