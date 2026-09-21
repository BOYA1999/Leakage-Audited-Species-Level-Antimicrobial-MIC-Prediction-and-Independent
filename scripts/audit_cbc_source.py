import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amr_multiview.species_mic import _canonical_structures, _looks_like_species

parser = argparse.ArgumentParser(description='Replay the ChEMBL 34 record-lineage audit without training models.')
parser.add_argument('--chembl-db', type=Path, required=True)
parser.add_argument('--pairs', type=Path, required=True)
parser.add_argument('--class-mapping', type=Path, required=True)
parser.add_argument('--taxonomy', type=Path, default=ROOT / 'docs/taxonomy_mapping.csv')
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
DB, PAIRS, MAP, CLASS = args.chembl_db, args.pairs, args.taxonomy, args.class_mapping
sql = '''SELECT a.activity_id, a.assay_id, a.doc_id, a.molregno,
a.standard_relation, a.standard_value, a.standard_units, a.data_validity_comment,
a.type AS original_type, a.relation AS original_relation, a.value AS original_value,
a.units AS original_units, a.potential_duplicate, a.standard_flag,
s.assay_id AS joined_assay_id, s.assay_organism AS raw_organism,
s.assay_tax_id AS source_tax_id, s.assay_strain AS strain,
s.chembl_id AS assay_chembl_id, cs.molregno AS joined_structure_id,
cs.standard_inchi_key AS compound_inchikey, cs.canonical_smiles AS compound_smiles
FROM activities a LEFT JOIN assays s ON a.assay_id=s.assay_id
LEFT JOIN compound_structures cs ON a.molregno=cs.molregno
WHERE a.standard_type='MIC' '''
with sqlite3.connect(f'file:{DB.as_posix()}?mode=ro', uri=True) as con:
    raw = pd.read_sql_query(sql, con)
    documents = pd.read_sql_query('SELECT doc_id,chembl_id AS doc_chembl_id,year,journal,doi,pubmed_id,title FROM docs', con)
print('All MIC source records:', len(raw), flush=True)
assert raw.activity_id.is_unique
raw['standard_value'] = pd.to_numeric(raw.standard_value, errors='coerce')
distributions = []
for column in ['standard_relation', 'standard_units', 'data_validity_comment', 'potential_duplicate', 'standard_flag']:
    counts = raw[column].fillna('<missing>').astype(str).value_counts(dropna=False)
    distributions.extend({'field': column, 'value': value, 'records': int(n), 'denominator': len(raw)} for value, n in counts.items())
pd.DataFrame(distributions).to_csv(OUT / 'source_prefilter_distributions.csv', index=False)
flow = [{'step': 'All ChEMBL 34 standard-type MIC records', 'retained': len(raw), 'excluded_at_step': 0}]
raw['exclusion_reason'] = 'retained_sql'
criteria = [
    ('missing assay or compound-structure join', raw.joined_assay_id.notna() & raw.joined_structure_id.notna()),
    ('non-exact or missing standardized relation', raw.standard_relation.eq('=')),
    ('unit other than ug.mL-1 or missing', raw.standard_units.eq('ug.mL-1')),
    ('missing nonfinite or nonpositive standardized value', np.isfinite(raw.standard_value) & raw.standard_value.gt(0)),
    ('missing organism or source taxonomy ID', raw.raw_organism.notna() & raw.source_tax_id.notna()),
    ('missing InChIKey or source SMILES', raw.compound_inchikey.notna() & raw.compound_smiles.notna()),
    ('nonmissing data-validity comment', raw.data_validity_comment.isna())]
for name, mask in criteria:
    reject = raw.exclusion_reason.eq('retained_sql') & ~mask
    raw.loc[reject, 'exclusion_reason'] = name
    flow.append({'step': name, 'retained': int(raw.exclusion_reason.eq('retained_sql').sum()), 'excluded_at_step': int(reject.sum())})
assert raw.exclusion_reason.eq('retained_sql').sum() == 442271
taxonomy = pd.read_csv(MAP).set_index('tax_id')
classes = json.loads(CLASS.read_text(encoding='utf-8'))
pairs = pd.read_csv(PAIRS, float_precision='round_trip')
raw['organism'] = raw.source_tax_id.map(taxonomy.species_name)
raw['tax_id'] = raw.source_tax_id.map(taxonomy.species_tax_id)
raw['relation_group'] = np.select([raw.standard_relation.eq('='), raw.standard_relation.isin(['>', '>=', '<', '<='])], ['exact', 'censored'], default='other_or_missing')
relation = raw[raw.organism.isin(pairs.organism.unique())].groupby(['organism', 'relation_group'], dropna=False).size().unstack(fill_value=0)
relation['all_mapped_records'] = relation.sum(axis=1)
relation['censored_fraction'] = relation.get('censored', 0) / relation.all_mapped_records
relation.to_csv(OUT / 'source_species_relations.csv')
raw[['activity_id', 'assay_id', 'doc_id', 'molregno', 'source_tax_id', 'exclusion_reason']].to_csv(OUT / 'source_sql_selection.csv.gz', index=False)
selected = raw[raw.exclusion_reason.eq('retained_sql')].copy()
del raw
selected['pathogen_class'] = selected.raw_organism.map(classes)
selected['pipeline_status'] = 'retained'
filters = [
    ('unmapped pathogen class', selected.pathogen_class.notna()),
    ('non-binomial or out-of-scope organism label', selected.raw_organism.map(_looks_like_species)),
    ('no species ancestor', selected.organism.notna() & selected.tax_id.notna())]
for name, mask in filters:
    reject = selected.pipeline_status.eq('retained') & ~mask
    selected.loc[reject, 'pipeline_status'] = name
    flow.append({'step': name, 'retained': int(selected.pipeline_status.eq('retained').sum()), 'excluded_at_step': int(reject.sum())})
structure = _canonical_structures(selected.loc[selected.pipeline_status.eq('retained'), ['compound_inchikey', 'compound_smiles']])
reject = selected.pipeline_status.eq('retained') & selected.compound_inchikey.map(structure).isna()
selected.loc[reject, 'pipeline_status'] = 'invalid representative structure'
flow.append({'step': 'invalid representative structure', 'retained': int(selected.pipeline_status.eq('retained').sum()), 'excluded_at_step': int(reject.sum())})
selected.loc[selected.pipeline_status.eq('retained') & ~selected.organism.isin(pairs.organism.unique()), 'pipeline_status'] = 'below species eligibility threshold'
selected[['activity_id', 'compound_inchikey', 'source_tax_id', 'tax_id', 'organism', 'pipeline_status']].to_csv(OUT / 'source_postsql_selection.csv.gz', index=False)
retained = selected[selected.pipeline_status.eq('retained')].copy()
del selected
retained['pathogen_class'] = retained.organism.map(classes).fillna(retained.pathogen_class)
retained['tax_id'] = retained.tax_id.astype(int)
retained['source_tax_id'] = retained.source_tax_id.astype(int)
retained['log2_mic'] = np.log2(retained.standard_value)
keys = ['compound_inchikey', 'organism', 'pathogen_class', 'tax_id']
grouped = retained.groupby(keys, sort=False, observed=True)
replay = grouped.log2_mic.agg(['median', 'size']).reset_index()
q = grouped.log2_mic.quantile([0.25, 0.75]).unstack().reset_index()
q['replayed_iqr'] = q[0.75] - q[0.25]
replay = replay.merge(q[keys + ['replayed_iqr']], on=keys, validate='one_to_one').merge(pairs, on=keys, validate='one_to_one')
assert len(replay) == len(pairs) == 214068
assert (replay['size'] == replay.n_measurements).all()
errors = {'median': float((replay['median'] - replay.log2_mic).abs().max()), 'iqr': float((replay.replayed_iqr - replay.log2_mic_iqr).abs().max())}
assert max(errors.values()) < 1e-12
assert len(retained) == pairs.n_measurements.sum()
retained = retained.merge(documents, on='doc_id', how='left', validate='many_to_one')
lineage = ['activity_id', 'compound_inchikey', 'molregno', 'source_tax_id', 'tax_id', 'organism', 'raw_organism', 'strain', 'assay_id', 'assay_chembl_id', 'doc_id', 'doc_chembl_id', 'year', 'standard_value', 'original_type', 'original_relation', 'original_value', 'original_units', 'potential_duplicate']
retained[lineage].to_csv(OUT / 'retained_activity_lineage.csv.gz', index=False)
support = retained.groupby(['organism', 'tax_id']).agg(records=('activity_id', 'size'), documents=('doc_id', 'nunique'), assays=('assay_id', 'nunique'), strain_labels=('strain', 'nunique'), source_organism_labels=('raw_organism', 'nunique'), missing_strain_records=('strain', lambda x: int(x.isna().sum())), source_taxids=('source_tax_id', 'nunique')).reset_index()
support.to_csv(OUT / 'species_source_support.csv', index=False)
extreme = pairs[pairs.log2_mic_iqr.ge(8)].copy()
extreme.drop(columns='canonical_smiles', errors='ignore').to_csv(OUT / 'extreme_dispersion_pairs.csv', index=False)
extreme_records = retained.merge(extreme[keys + ['log2_mic_iqr']], on=keys, validate='many_to_one')
extreme_records[lineage + ['journal', 'doi', 'pubmed_id', 'title', 'standard_relation', 'standard_units', 'data_validity_comment', 'log2_mic_iqr']].to_csv(OUT / 'extreme_dispersion_records.csv.gz', index=False)
unit_flags, unit_summaries = [], []
for (key, organism), group in extreme_records.groupby(['compound_inchikey', 'organism']):
    normalized = group.original_units.fillna('').str.lower().str.replace(' ', '', regex=False).str.replace('.', '', regex=False)
    factors = normalized.map({'ugml-1': 1.0, 'ug/ml': 1.0, 'mg/l': 1.0, 'mgl-1': 1.0, 'mg/ml': 1000.0, 'mgml-1': 1000.0, 'ug/l': 0.001, 'ugl-1': 0.001, 'ng/ml': 0.001})
    numer = pd.to_numeric(group.original_value, errors='coerce')
    checked = factors.notna() & numer.notna()
    mismatch = checked & ~np.isclose(numer * factors, group.standard_value, rtol=1e-6, atol=0)
    flagged = group[['activity_id', 'compound_inchikey', 'organism', 'doc_id', 'original_value', 'original_units', 'standard_value']].copy()
    flagged['conversion_to_ug_ml'] = factors
    flagged['converted_original_value'] = numer * factors
    flagged['difference_standard_minus_converted'] = group.standard_value - numer * factors
    flagged['difference_above_tolerance'] = mismatch
    unit_flags.append(flagged)
    unit_summaries.append({'compound_inchikey': key, 'organism': organism, 'records': len(group), 'documents': group.doc_id.nunique(),
        'assays': group.assay_id.nunique(), 'strain_labels': group.strain.nunique(), 'source_organism_labels': group.raw_organism.nunique(),
        'original_units': '; '.join(sorted(group.original_units.fillna('<missing>').astype(str).unique())),
        'missing_original_value': int(numer.isna().sum()), 'unit_conversion_checked_records': int(checked.sum()), 'unit_conversion_mismatch_records': int(mismatch.sum()),
        'potential_duplicate_records': int(group.potential_duplicate.fillna(0).eq(1).sum()), 'iqr_log2': group.log2_mic_iqr.iloc[0], 'document_fulltext_reviewed': False})
pd.concat(unit_flags).to_csv(OUT / 'extreme_unit_conversion_audit.csv', index=False)
pd.DataFrame(unit_summaries).to_csv(OUT / 'extreme_dispersion_audit_summary.csv', index=False)
pd.DataFrame(flow).to_csv(OUT / 'source_filter_flow.csv', index=False)
pairs.drop(columns='canonical_smiles', errors='ignore').to_csv(OUT / 'pair_endpoints.csv.gz', index=False)
report = {'all_mic_records': flow[0]['retained'], 'sql_retained': 442271, 'eligible_records': len(retained), 'pairs_replayed': len(replay), 'maximum_absolute_difference': errors,
          'extreme_threshold_iqr': 8, 'extreme_pairs': len(extreme), 'extreme_records': len(extreme_records), 'extreme_documents': int(extreme_records.doc_id.nunique()),
          'all_retained_documents': int(retained.doc_id.nunique()), 'all_retained_assays': int(retained.assay_id.nunique()),
          'sql_sha256': hashlib.sha256(sql.encode()).hexdigest(), 'pair_file_sha256': hashlib.sha256((PAIRS).read_bytes()).hexdigest(),
          'taxonomy_sha256': hashlib.sha256(MAP.read_bytes()).hexdigest(), 'relation_mapping_scope': 'Frozen source-taxonomy mapping only; unmapped MIC source taxids are not assigned new ancestors.',
          'fulltext_verification': 'not performed; original and standardized fields and source document identifiers audited, biological comparability not established'}
(OUT / 'source_audit_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
(OUT / 'source_audit_query.sql').write_text(sql, encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
