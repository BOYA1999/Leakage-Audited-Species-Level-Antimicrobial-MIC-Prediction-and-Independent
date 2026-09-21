from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from .features import build_feature_matrix
from .maier_external_validation import _regression_model
from .species_mic_experiment import (
    _finite_sample_quantile, _pair_features, _species_balanced_weights, scaffold_partition,
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_replay(expected, observed, tolerance):
    keys = ['compound_inchikey', 'organism']
    joined = observed.merge(expected, on=keys, suffixes=('_new', '_old'), validate='one_to_one')
    if len(joined) != len(expected) or len(joined) != len(observed):
        raise ValueError('Replay test membership differs from the original run')
    errors = {field: float(np.max(np.abs(joined[field + '_new'] - joined[field + '_old'])))
              for field in ['y_true', 'y_pred', 'upper_90']}
    if not all(np.isfinite(value) and value <= tolerance for value in errors.values()):
        raise ValueError(f'Replay prediction gate failed: {errors}')
    return errors


def primary_mask(frame):
    return ~frame['exact_training_overlap'] & ~frame['joint_corpus_exact_overlap']


def metric_pair(y, score):
    if len(np.unique(y)) != 2:
        return np.array([np.nan, np.nan])
    return np.array([roc_auc_score(y, score), average_precision_score(y, score)])


def overlap_audit(compounds, external, output, log):
    result = external[['prestwick_ID', 'exact_training_overlap']].copy()
    result['joint_corpus_exact_overlap'] = external['pchem_inchikey'].isin(compounds['compound_inchikey'])
    training_fps, training_parents = [], set()
    for i, smiles in enumerate(compounds['canonical_smiles']):
        mol = Chem.MolFromSmiles(smiles)
        training_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        parent = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(mol))
        training_parents.add(Chem.MolToSmiles(parent, canonical=True, isomericSmiles=False))
        if (i + 1) % 10000 == 0:
            log(f'Joint-corpus chemical audit: {i + 1}/{len(compounds)}')
    similarities, parent_identity = [], []
    for smiles in external['rdkit_no_salt']:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        similarities.append(max(DataStructs.BulkTanimotoSimilarity(fp, training_fps)))
        parent = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(mol))
        parent_identity.append(Chem.MolToSmiles(parent, canonical=True, isomericSmiles=False) in training_parents)
    result['joint_corpus_nearest_tanimoto'] = similarities
    result['joint_corpus_fingerprint_identity'] = result['joint_corpus_nearest_tanimoto'].eq(1.0)
    result['joint_corpus_parent_identity'] = parent_identity
    result['primary'] = primary_mask(result)
    result['fingerprint_nonidentity'] = result['primary'] & ~result['joint_corpus_fingerprint_identity']
    result['parent_nonidentity'] = result['primary'] & ~result['joint_corpus_parent_identity']
    result.to_csv(output / 'overlap_audit.csv', index=False)
    log('All-species overlap audit complete')
    return result


def evaluate(predictions, external, mapping, audit, config, output, log):
    scores = predictions.groupby(['prestwick_ID', 'organism', 'family', 'representation'], as_index=False)['score'].mean()
    scores.to_csv(output / 'mean_scores.csv.gz', index=False)
    seed_rows, metric_rows, topk_rows, contrast_rows, flow_rows = [], [], [], [], []
    rng = np.random.default_rng(config['bootstrap_seed'])
    cohorts = ['primary', 'fingerprint_nonidentity', 'parent_nonidentity']
    contrasts = [('joint:multiview', 'joint:morgan'), ('joint:multiview', 'joint:morgan_maccs'),
                 ('joint:multiview', 'single:multiview')]
    for organism, group in mapping.groupby('organism', sort=True):
        strains = group['source_strain'].tolist()
        endpoints = {organism + ' any strain': external[strains].max(axis=1).to_numpy(dtype=int)}
        if len(strains) > 1:
            endpoints.update({strain: external[strain].to_numpy(dtype=int) for strain in strains})
        wide = scores[scores['organism'].eq(organism)].pivot(index='prestwick_ID', columns=['family', 'representation'], values='score')
        wide.columns = [a + ':' + b for a, b in wide.columns]
        wide = wide.reindex(external['prestwick_ID'])
        wide['concentration_only'] = np.log2(0.02 * external['ExactMolWt'].to_numpy())
        if not np.isfinite(wide.to_numpy()).all():
            raise ValueError('Non-finite external score')
        for cohort in cohorts:
            mask = audit[cohort].to_numpy(dtype=bool)
            index = external.loc[mask, 'prestwick_ID']
            for endpoint, labels in endpoints.items():
                y = labels[mask]
                flow_rows.append({'organism': organism, 'endpoint': endpoint, 'cohort': cohort,
                                  'n': len(y), 'positives': int(y.sum()), 'prevalence': float(y.mean())})
                for model in wide:
                    score = wide.loc[index, model].to_numpy()
                    values = metric_pair(y, score)
                    metric_rows.append({'organism': organism, 'endpoint': endpoint, 'cohort': cohort,
                                        'model': model, 'n': len(y), 'positives': int(y.sum()),
                                        'auroc': values[0], 'ap': values[1],
                                        'status': 'estimable' if np.isfinite(values).all() else 'one_class_not_estimable'})
                    if endpoint.endswith(' any strain'):
                        order = np.argsort(-score, kind='stable')
                        for k in [25, 50, 100]:
                            take = min(k, len(y))
                            precision = float(y[order[:take]].mean())
                            topk_rows.append({'organism': organism, 'cohort': cohort, 'model': model, 'k': take,
                                              'hits': int(y[order[:take]].sum()), 'precision': precision,
                                              'enrichment': precision / y.mean() if y.mean() else np.nan})
                if not endpoint.endswith(' any strain') or len(np.unique(y)) != 2:
                    continue
                by_seed = predictions[predictions['organism'].eq(organism) & predictions['prestwick_ID'].isin(index)]
                label_lookup = pd.Series(y, index=index)
                for (family, representation, seed), subset in by_seed.groupby(['family', 'representation', 'seed']):
                    values = metric_pair(label_lookup.loc[subset['prestwick_ID']].to_numpy(), subset['score'].to_numpy())
                    seed_rows.append({'organism': organism, 'cohort': cohort, 'family': family,
                                      'representation': representation, 'seed': seed, 'auroc': values[0], 'ap': values[1]})
                models = list(dict.fromkeys(name for pair in contrasts for name in pair))
                matrix = wide.loc[index, models].to_numpy()
                draws = np.empty((config['bootstrap'], len(models), 2))
                valid = 0
                while valid < config['bootstrap']:
                    sample = rng.integers(0, len(y), len(y))
                    if len(np.unique(y[sample])) != 2:
                        continue
                    for j in range(len(models)):
                        draws[valid, j] = metric_pair(y[sample], matrix[sample, j])
                    valid += 1
                    if valid % 1000 == 0:
                        log(f'Bootstrap {organism} / {cohort}: {valid}/{config["bootstrap"]}')
                for left, right in contrasts:
                    a, b = models.index(left), models.index(right)
                    point = metric_pair(y, matrix[:, a]) - metric_pair(y, matrix[:, b])
                    limits = np.quantile(draws[:, a] - draws[:, b], [0.025, 0.975], axis=0)
                    for j, metric in enumerate(['auroc', 'ap']):
                        contrast_rows.append({'organism': organism, 'endpoint': endpoint, 'cohort': cohort,
                                              'comparison': left + '_minus_' + right, 'metric': metric,
                                              'difference': point[j], 'ci_low': limits[0, j], 'ci_high': limits[1, j],
                                              'bootstrap': config['bootstrap'], 'interval_type': 'paired compound percentile; descriptive post hoc'})
                pd.DataFrame(contrast_rows).to_csv(output / 'paired_differences.csv', index=False)
    for name, rows in [('metrics', metric_rows), ('metrics_by_seed', seed_rows), ('topk_enrichment', topk_rows), ('cohort_flow', flow_rows)]:
        pd.DataFrame(rows).to_csv(output / (name + '.csv'), index=False)


def run(config_path):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'models').mkdir()

    def log(message):
        entry = datetime.now(timezone.utc).isoformat() + ' ' + message
        print(entry, flush=True)
        with (output / 'run.log').open('a', encoding='utf-8') as handle:
            handle.write(entry + '\n')

    source = json.loads((Path(config['species_dir']) / 'manifest.json').read_text())
    legacy = json.loads((Path(config['legacy_external_dir']) / 'manifest.json').read_text())
    paths = [Path(config_path), Path(source['data']), Path(config['taxonomy_mapping']),
             Path(legacy['maier_screen']), Path(legacy['maier_library']), Path(__file__),
             Path(config['species_dir']) / 'manifest.json', Path(config['species_dir']) / 'predictions.csv.gz',
             Path(config['ablation_dir']) / 'predictions.csv.gz', Path(config['legacy_external_dir']) / 'manifest.json',
             Path(config['legacy_external_dir']) / 'compound_predictions.csv.gz',
             Path(__file__).with_name('species_mic_experiment.py'), Path(__file__).with_name('features.py'),
             Path(__file__).with_name('maier_external_validation.py')]
    manifest = {'status': 'running', 'config': config, 'input_and_code_hashes': {str(p.resolve()): file_hash(p) for p in paths},
                'environment': {'python': sys.version, 'platform': platform.platform(), 'lightgbm': lgb.__version__,
                                'rdkit': rdBase.rdkitVersion, 'numpy': np.__version__, 'pandas': pd.__version__},
                'scope': 'prediction-verified reconstruction; external phenotypic ranking, not quantitative MIC validation'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    assert file_hash(legacy['maier_screen']) == 'c61d2e5f10efd4381ec5dd6a7c2c5e99d4b059235583140078caf4c7e4a815da'
    assert file_hash(legacy['maier_library']) == 'ee2e65b35f928d5e95bb206b4cf402a8c6a9c459208d6529d883997ab313be97'
    if config['seeds'] != source['seeds'] or config['n_estimators'] != source['n_estimators']:
        raise ValueError('Frozen joint-model parameters differ')
    pairs = pd.read_csv(source['data'])
    compounds = pairs[['compound_inchikey', 'canonical_smiles']].drop_duplicates().reset_index(drop=True)
    assert len(pairs) == 214068 and len(compounds) == 63486
    assert compounds['compound_inchikey'].is_unique and not pairs.duplicated(['compound_inchikey', 'organism']).any()
    lookup = pd.Series(np.arange(len(compounds)), index=compounds['compound_inchikey'])
    pair_indices = pairs['compound_inchikey'].map(lookup).to_numpy(dtype=int)
    encoder = OneHotEncoder(handle_unknown='error', sparse_output=True, dtype=np.float32)
    species_features = encoder.fit_transform(pairs[['organism']])
    assert species_features.shape[1] == 48
    screen = pd.read_csv(legacy['maier_screen'], sep='\t')
    external = pd.read_csv(legacy['maier_library'], sep='\t').merge(screen, on='prestwick_ID', validate='one_to_one')
    external = external[external['rdkit_no_salt'].notna()].reset_index(drop=True)
    legacy_flags = pd.read_csv(Path(config['legacy_external_dir']) / 'compound_predictions.csv.gz', usecols=['prestwick_ID', 'exact_training_overlap'])
    external = external.merge(legacy_flags, on='prestwick_ID', validate='one_to_one')
    assert len(external) == 1193 and int((~external['exact_training_overlap']).sum()) == 1015
    assert external['prestwick_ID'].is_unique and np.isfinite(external['ExactMolWt']).all() and (external['ExactMolWt'] > 0).all()
    mapping = pd.read_csv(config['taxonomy_mapping'])
    assert set(mapping['source_strain']) == set(screen.columns[1:])
    internal_species = pairs[['tax_id', 'organism']].drop_duplicates()
    mapping = mapping.merge(internal_species, left_on='accepted_tax_id', right_on='tax_id', validate='many_to_one')
    assert set(mapping['accepted_tax_id']) == {562, 1496} and len(mapping) == 3
    mapping.to_csv(output / 'matched_species.csv', index=False)
    assert not external[mapping['source_strain']].isna().any().any()
    assert set(np.unique(external[mapping['source_strain']].to_numpy())) <= {0, 1}
    log('Building the frozen classical features once; no GPU used')
    views = ['morgan', 'maccs', 'physchem', 'graph']
    internal_full = sparse.csr_matrix(build_feature_matrix(compounds['canonical_smiles'], views).x)
    external_full = sparse.csr_matrix(build_feature_matrix(external['rdkit_no_salt'], views).x)
    lengths = {'morgan': 1024, 'morgan_maccs': 1191, 'multiview': 1218}
    audit = overlap_audit(compounds, external, output, log)
    log('External cohort sizes: ' + str({name: int(audit[name].sum()) for name in ['primary', 'fingerprint_nonidentity', 'parent_nonidentity']}))
    columns = ['compound_inchikey', 'organism', 'model', 'seed', 'y_true', 'y_pred', 'upper_90']
    original = pd.read_csv(Path(config['species_dir']) / 'predictions.csv.gz', usecols=columns)
    reduced = pd.read_csv(Path(config['ablation_dir']) / 'predictions.csv.gz', usecols=columns)
    reference = pd.concat([original[original['model'].isin(['morgan', 'multiview'])], reduced[reduced['model'].eq('morgan_maccs')]])
    del original, reduced
    y = pairs['log2_mic'].to_numpy(dtype=np.float32)
    args = SimpleNamespace(n_estimators=config['n_estimators'], n_jobs=config['n_jobs'])
    replay_rows, predictions, split_reports = [], [], []
    concentration = np.log2(0.02 * external['ExactMolWt'].to_numpy())
    organisms = sorted(mapping['organism'].unique())
    for seed in config['seeds']:
        partition, report = scaffold_partition(compounds, seed)
        assert report == next(item for item in source['split_reports'] if item['seed'] == seed)
        split_reports.append(report)
        row_partition = partition[pair_indices]
        train, valid, test = [np.flatnonzero(row_partition == name) for name in ['train', 'valid', 'test']]
        weights = _species_balanced_weights(pairs.iloc[train]['organism'])
        for representation in config['representations']:
            log(f'Fitting joint {representation}, seed {seed}')
            internal_x = internal_full[:, :lengths[representation]]
            external_x = external_full[:, :lengths[representation]]
            joint_x = _pair_features(internal_x, pair_indices, species_features)
            model = _regression_model(seed, args)
            model.fit(joint_x[train], y[train], sample_weight=weights)
            pred_test = model.predict(joint_x[test])
            q = _finite_sample_quantile(np.abs(y[valid] - model.predict(joint_x[valid])))
            observed = pairs.iloc[test][['compound_inchikey', 'organism']].copy()
            observed['y_true'], observed['y_pred'], observed['upper_90'] = y[test], pred_test, pred_test + q
            expected = reference[reference['model'].eq(representation) & reference['seed'].eq(seed)].drop(columns=['model', 'seed'])
            errors = check_replay(expected, observed, config['replay_absolute_tolerance'])
            replay_rows.append({'model': representation, 'seed': seed, 'test_rows': len(test),
                                'max_y_difference': errors['y_true'], 'max_prediction_difference': errors['y_pred'],
                                'max_upper_difference': errors['upper_90'], 'pooled_q': q, 'passed': True})
            pd.DataFrame(replay_rows).to_csv(output / 'internal_replay.csv', index=False)
            model.booster_.save_model(str(output / 'models' / f'joint_{representation}_{seed}.txt'))
            log(f'Replay passed: {representation}/{seed}, max prediction difference={errors["y_pred"]:.3g}')
            for organism in organisms:
                codes = encoder.transform(pd.DataFrame({'organism': [organism] * len(external)}))
                pred = model.predict(_pair_features(external_x, np.arange(len(external)), codes))
                predictions.append(pd.DataFrame({'prestwick_ID': external['prestwick_ID'], 'organism': organism,
                                                'family': 'joint', 'representation': representation, 'seed': seed,
                                                'pred_log2_mic': pred, 'score': concentration - pred}))
            del model, joint_x
            for organism in organisms:
                subset = train[pairs.iloc[train]['organism'].eq(organism).to_numpy()]
                model = _regression_model(seed, args)
                model.fit(internal_x[pair_indices[subset]], y[subset],
                          sample_weight=_species_balanced_weights(pairs.iloc[subset]['organism']))
                pred = model.predict(external_x)
                predictions.append(pd.DataFrame({'prestwick_ID': external['prestwick_ID'], 'organism': organism,
                                                'family': 'single', 'representation': representation, 'seed': seed,
                                                'pred_log2_mic': pred, 'score': concentration - pred}))
                model.booster_.save_model(str(output / 'models' / f'single_{organism.replace(" ", "_")}_{representation}_{seed}.txt'))
                del model
            pd.concat(predictions, ignore_index=True).to_csv(output / 'seed_predictions.csv.gz', index=False, compression='gzip')
            gc.collect()
    prediction_frame = pd.concat(predictions, ignore_index=True)
    assert not prediction_frame.duplicated(['prestwick_ID', 'organism', 'family', 'representation', 'seed']).any()
    assert np.isfinite(prediction_frame[['pred_log2_mic', 'score']].to_numpy()).all()
    evaluate(prediction_frame, external, mapping, audit, config, output, log)
    manifest.update(status='complete', split_reports=split_reports, species_encoder_categories=encoder.categories_[0].tolist(),
                    joint_fits=len(replay_rows), matched_single_fits=len(replay_rows) * len(organisms),
                    replay_passed=all(row['passed'] for row in replay_rows), predictions=len(prediction_frame))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    log('Complete: prediction-verified joint external phenotypic evaluation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruct and externally evaluate the actual joint MIC models.')
    parser.add_argument('--config', required=True)
    run(parser.parse_args().config)
