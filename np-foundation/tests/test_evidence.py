"""Meaningful evidence-boundary, split-leakage and structure-identity checks."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from npfm.evidence import EvidenceStore, EvidenceValidationError, validate_record
from npfm.evaluate import (evaluate_predictions, evaluate_multitask, group_split,
                           scaffold_groups, scaffold_split, time_split)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def store():
    return EvidenceStore.from_jsonl(ROOT/'examples/evidence.jsonl')


def test_source_records_are_valid(store):
    assert len(store.records) == 17
    assert all(r['sources'] and r['retrieved_at'] for r in store.records)
    assert all(r['clinical_efficacy_probability'] is None for r in store.records)


def test_paclitaxel_approved_indications_retain_context(store):
    report = store.indication_report('紫杉醇')
    approvals = report['approved_indications']
    assert len(approvals) == 4
    assert {r['object'] for r in approvals} == {
        'ovarian cancer','breast cancer','non-small cell lung cancer','AIDS-related Kaposi sarcoma'}
    assert all(r['context']['jurisdiction'] == 'US' for r in approvals)
    assert all(r['context']['formulation'] == 'conventional IV injection' for r in approvals)
    assert store.query('paclitaxel') == store.query('紫杉醇')


def test_parent_does_not_inherit_derivative_trial(store):
    parent = store.indication_report('artemisinin','colorectal cancer')
    derivative = store.indication_report('artesunate','colorectal cancer')
    assert parent['status'] == 'unknown' and not parent['clinical_exploration']
    assert not parent['approved_indications']
    assert len(derivative['clinical_exploration']) == 1
    assert not derivative['approved_indications']
    assert derivative['clinical_exploration'][0]['context']['randomized_participants'] == 23


def test_act_recommendation_is_not_parent_approval(store):
    assert len(store.indication_report('ACT')['guideline_recommendations']) == 1
    assert not store.indication_report('artemisinin')['guideline_recommendations']
    assert not store.indication_report('ACT')['approved_indications']
    assert len(store.indication_report('oral artemisinin monotherapy')['not_recommended']) == 1


def test_unknown_is_not_negative_and_matching_is_exact(store):
    report = store.indication_report('artemisin','colorectal cancer')
    assert report['status'] == 'unknown'
    assert report['evidence_records'] == []
    assert report['clinical_efficacy_probability'] is None


def test_invalid_approval_and_missing_source_are_rejected(store):
    example = deepcopy(store.records[0])
    example['evidence_level'] = 'computational'
    with pytest.raises(EvidenceValidationError): validate_record(example)
    example = deepcopy(store.records[0]);example['context'].pop('jurisdiction')
    with pytest.raises(EvidenceValidationError): validate_record(example)
    example = deepcopy(store.records[0]);example['sources'] = []
    with pytest.raises(EvidenceValidationError): validate_record(example)


def test_computational_hypothesis_remains_separate(store):
    record = deepcopy(store.records[0])
    record.update(id='synthetic_test_hypothesis',predicate='hypothesized_for',
                  evidence_level='computational',summary='Test-only hypothesis; not experimental evidence.')
    local = EvidenceStore([record])
    report = local.indication_report('paclitaxel')
    assert len(report['computational_hypotheses']) == 1
    assert not report['approved_indications']
    assert local.query('paclitaxel',include_hypotheses=False) == []


def test_paths_are_observed_edges_and_not_approval_inference(store):
    paths = store.trace_paths('paclitaxel','mitotic progression')
    assert paths
    assert all(p['interpretation']=='retrieved_edges_not_causal_or_clinical_proof' for p in paths)
    assert all(edge['sources'] for p in paths for edge in p['edges'])
    assert store.trace_paths('artemisinin','colorectal cancer') == []


def test_regression_ignores_missing_targets_and_reports_coverage():
    result = evaluate_predictions([1, None, 3, 5],[2,100,2,np.nan])
    assert result['mae'] == 1
    assert result['rmse'] == 1
    assert result['n_labeled'] == 3 and result['n_evaluated'] == 2
    assert result['coverage'] == 0.5
    assert result['prediction_coverage'] == pytest.approx(2/3)


def test_classification_metrics_on_observed_labels():
    result = evaluate_predictions([0,1,0,1,None],[.1,.9,.2,.8,.99],task='classification')
    assert result['roc_auc'] == 1
    assert result['pr_auc'] == 1
    assert result['average_precision'] == 1
    assert result['brier'] == pytest.approx(.025)
    assert result['n_evaluated'] == 4


def test_single_class_ranking_is_undefined():
    result = evaluate_predictions([1,1],[.9,.8],task='classification')
    assert result['roc_auc'] is None and result['pr_auc'] is None
    assert result['average_precision'] is None
    assert result['brier'] == pytest.approx(.025)


def test_empty_metrics_and_invalid_probabilities():
    assert evaluate_predictions([None],[.3])['mae'] is None
    with pytest.raises(ValueError):
        evaluate_predictions([0,1],[.1,2.0],task='classification')


def test_multitask_masks_each_endpoint_separately():
    result=evaluate_multitask([[0,1],[1,None]],[[.1,2],[.9,100]],['classification','regression'],['active','measurement'])
    assert result['tasks']['active']['n_evaluated'] == 2
    assert result['tasks']['measurement']['n_evaluated'] == 1
    assert result['tasks']['measurement']['mae'] == 1


def test_group_split_has_no_shared_identity():
    groups=['lab_a']*4+['lab_b']*3+['lab_c']*2+['lab_d']+['lab_e']+['lab_f']
    parts=group_split(groups,seed=17)
    assert parts == group_split(groups,seed=17)
    assert sorted(i for rows in parts.values() for i in rows) == list(range(len(groups)))
    sets=[{groups[i] for i in rows} for rows in parts.values()]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3))


def test_scaffold_split_keeps_scaffold_and_acyclic_families_together():
    smiles=['c1ccccc1O','c1ccccc1C','CCO','CCC','C1CCCCC1','c1ncccc1']
    groups=scaffold_groups(smiles)
    assert groups[0] == groups[1]
    assert groups[2] == groups[3] == '__ACYCLIC__'
    parts=scaffold_split(smiles,seed=4)
    sets=[{groups[i] for i in rows} for rows in parts.values()]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3))


def test_time_split_promotes_entire_future_group():
    dates=['2019-01-01','2022-01-01','2020-07-01','2019-03-01']
    parts=time_split(dates,'2020-01-01','2021-01-01',groups=['same','same','mid','past'])
    assert parts == {'train':[3],'validation':[2],'test':[0,1]}


def test_pubchem_structures_match_responses_and_full_inchikeys():
    from rdkit import Chem
    rows=list(csv.DictReader((ROOT/'examples/compounds.csv').open()))
    provenance=[json.loads(line) for line in (ROOT/'examples/pubchem_provenance.jsonl').read_text().splitlines()]
    assert len(rows)==len(provenance)==36
    assert len({r['cid'] for r in rows})==36
    by_name={r['name']:r for r in provenance}
    for row in rows:
        source=by_name[row['name']]
        prop=source['response']['PropertyTable']['Properties'][0]
        assert row['isomeric_smiles']==row['smiles']==prop['SMILES']
        assert row['canonical_smiles']==prop['ConnectivitySMILES']
        assert row['inchikey']==prop['InChIKey']
        mol=Chem.MolFromSmiles(row['smiles'])
        assert mol is not None
        assert Chem.MolToInchiKey(mol)==row['inchikey']
        assert row['label']=='' and row['label_status']=='unlabeled'
        assert source['http_status']==200
