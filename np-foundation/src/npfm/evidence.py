"""Source-preserving evidence retrieval; this module does not predict efficacy.

Evidence is keyed by exact identities and explicitly supplied aliases.  A parent
compound never inherits approvals for derivatives, formulations, or combinations.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

EVIDENCE_LEVELS = frozenset({
    'regulatory_label', 'guideline', 'clinical_trial', 'animal', 'cell',
    'biochemical', 'mechanistic', 'computational', 'unknown',
})
CLAIM_STATUSES = frozenset({'positive', 'negative', 'mixed', 'unknown'})
ENTITY_TYPES = frozenset({
    'compound', 'compound_class', 'formulation', 'regimen', 'protein', 'target',
    'pathway', 'cell_state', 'disease', 'organism', 'biochemical_factor',
})


class EvidenceValidationError(ValueError):
    """An evidence record is incomplete or its claimed level is inconsistent."""


def _identity(value: str) -> str:
    return ' '.join(value.strip().casefold().split())


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate and return an independent copy of one evidence record.

    ``claim_status`` is the status of the specific relation, not an efficacy
    probability. ``unknown`` is never converted into a negative observation.
    """
    if not isinstance(record, dict):
        raise EvidenceValidationError('Each JSONL record must be an object')
    r = deepcopy(record)
    for key in ('id', 'subject', 'subject_type', 'predicate', 'object',
                'object_type', 'evidence_level', 'claim_status', 'summary'):
        if not isinstance(r.get(key), str) or not r[key].strip():
            raise EvidenceValidationError(f'Missing or invalid field: {key}')
    if r['evidence_level'] not in EVIDENCE_LEVELS:
        raise EvidenceValidationError('Unsupported evidence_level')
    if r['claim_status'] not in CLAIM_STATUSES:
        raise EvidenceValidationError('Unsupported claim_status')
    if r['subject_type'] not in ENTITY_TYPES or r['object_type'] not in ENTITY_TYPES:
        raise EvidenceValidationError('Unsupported entity type')
    for key in ('subject_aliases', 'object_aliases'):
        r.setdefault(key, [])
        if not isinstance(r[key], list) or any(not isinstance(a, str) or not a.strip() for a in r[key]):
            raise EvidenceValidationError(f'{key} must contain nonempty strings')
    r.setdefault('context', {})
    if not isinstance(r['context'], dict):
        raise EvidenceValidationError('context must be an object')
    if not isinstance(r.get('sources'), list) or not r['sources']:
        raise EvidenceValidationError('At least one original source is required')
    for source in r['sources']:
        if not isinstance(source, dict) or not source.get('title'):
            raise EvidenceValidationError('Each source needs a title')
        parsed = urlparse(str(source.get('url', '')))
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
            raise EvidenceValidationError('Each source needs an HTTP(S) URL')
    if r['predicate'] == 'approved_for':
        if r['evidence_level'] != 'regulatory_label' or r['claim_status'] != 'positive':
            raise EvidenceValidationError('approved_for requires positive regulatory-label evidence')
        if not r['context'].get('jurisdiction'):
            raise EvidenceValidationError('approved_for requires an explicit jurisdiction')
        if r['object_type'] != 'disease':
            raise EvidenceValidationError('approved_for must identify a disease')
    if r['predicate'] in {'recommended_for', 'not_recommended_for'}:
        if r['evidence_level'] != 'guideline':
            raise EvidenceValidationError('Recommendation relations require guideline evidence')
    if r['predicate'] == 'hypothesized_for' and r['evidence_level'] != 'computational':
        raise EvidenceValidationError('hypothesized_for requires a computational evidence level')
    if r.get('clinical_efficacy_probability') is not None:
        raise EvidenceValidationError('Retrieval records cannot claim a clinical efficacy probability')
    return r


class EvidenceStore:
    """Small, transparent JSONL evidence store with exact-identity retrieval."""

    def __init__(self, records: Iterable[dict[str, Any]]):
        self.records = [validate_record(r) for r in records]
        ids = [r['id'] for r in self.records]
        if len(ids) != len(set(ids)):
            raise EvidenceValidationError('Duplicate evidence IDs')
        self._aliases: dict[str, str] = {}
        for r in self.records:
            for side in ('subject', 'object'):
                canonical = _identity(r[side])
                for alias in [r[side], *r.get(side + '_aliases', [])]:
                    key = _identity(alias)
                    old = self._aliases.get(key)
                    if old is not None and old != canonical:
                        raise EvidenceValidationError(f'Ambiguous entity alias: {alias}')
                    self._aliases[key] = canonical

    @classmethod
    def from_jsonl(cls, path: str | Path) -> 'EvidenceStore':
        rows = []
        with Path(path).open(encoding='utf-8') as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise EvidenceValidationError(f'Invalid JSON on line {line_no}') from exc
        return cls(rows)

    def _canonical(self, name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Entity names must be nonempty strings')
        return self._aliases.get(_identity(name), _identity(name))

    def query(self, compound: str, disease: str | None = None,
              include_hypotheses: bool = True) -> list[dict[str, Any]]:
        """Return direct subject records; no substring or derivative propagation."""
        subject = self._canonical(compound)
        target = self._canonical(disease) if disease is not None else None
        matched = []
        for r in self.records:
            if self._canonical(r['subject']) != subject:
                continue
            if target is not None and (r['object_type'] != 'disease' or self._canonical(r['object']) != target):
                continue
            if not include_hypotheses and r['evidence_level'] == 'computational':
                continue
            matched.append(deepcopy(r))
        return matched

    def related_records(self, entity: str) -> list[dict[str, Any]]:
        """Return one-hop context. Related facts do not become entity approvals."""
        key = self._canonical(entity)
        return [deepcopy(r) for r in self.records
                if self._canonical(r['subject']) == key or self._canonical(r['object']) == key]

    def indication_report(self, compound: str, disease: str | None = None) -> dict[str, Any]:
        records = self.query(compound, disease)
        disease_records = [r for r in records if r['object_type'] == 'disease']
        positive = lambda r: r['claim_status'] == 'positive'
        return {
            'compound': compound,
            'canonical_subject': self._canonical(compound),
            'requested_disease': disease,
            'status': 'retrieved_evidence' if disease_records else 'unknown',
            'clinical_efficacy_probability': None,
            'approved_indications': [r for r in disease_records if r['predicate'] == 'approved_for' and positive(r)],
            'guideline_recommendations': [r for r in disease_records if r['predicate'] == 'recommended_for' and positive(r)],
            'not_recommended': [r for r in disease_records if r['predicate'] == 'not_recommended_for' and positive(r)],
            'clinical_exploration': [r for r in disease_records if r['evidence_level'] == 'clinical_trial'],
            'experimental_evidence': [r for r in disease_records if r['evidence_level'] in {'animal', 'cell', 'biochemical', 'mechanistic'}],
            'computational_hypotheses': [r for r in disease_records if r['evidence_level'] == 'computational'],
            'unknown': [r for r in disease_records if r['claim_status'] == 'unknown' or r['evidence_level'] == 'unknown'],
            'indications': disease_records,
            'evidence_records': records,
            'coverage': 'Curated illustrative records; not an exhaustive regulatory or literature audit.',
            'interpretation': ('Retrieved sources and evidence levels are not a validated clinical efficacy score. '
                               'No record means unknown, not inactive. Derivatives and combinations retain separate identities.'),
        }

    def trace_paths(self, start: str, end: str, max_hops: int = 4) -> list[dict[str, Any]]:
        """Enumerate positive directed paths present in the file, without inference.

        Every edge retains its study context and source. A retrieved path does not
        establish matching contexts, causality, efficacy, or a new indication.
        """
        if not isinstance(max_hops, int) or not 1 <= max_hops <= 8:
            raise ValueError('max_hops must be an integer between 1 and 8')
        source, target = self._canonical(start), self._canonical(end)
        graph: dict[str, list[dict[str, Any]]] = {}
        for r in self.records:
            if r['claim_status'] != 'positive' or r['evidence_level'] == 'unknown':
                continue
            graph.setdefault(self._canonical(r['subject']), []).append(r)
        result: list[dict[str, Any]] = []

        def visit(node: str, nodes: list[str], edges: list[dict[str, Any]]) -> None:
            if len(edges) >= max_hops or len(result) >= 256:
                return
            for r in graph.get(node, []):
                nxt = self._canonical(r['object'])
                if nxt in nodes:
                    continue
                chain = [*edges, r]
                if nxt == target:
                    result.append({'nodes': [*nodes, nxt], 'edges': deepcopy(chain),
                                   'interpretation': 'retrieved_edges_not_causal_or_clinical_proof',
                                   'context_alignment': 'not_assessed',
                                   'clinical_efficacy_probability': None})
                else:
                    visit(nxt, [*nodes, nxt], chain)
        visit(source, [source], [])
        return result


def query_evidence(path: str | Path, compound: str, disease: str | None = None) -> dict[str, Any]:
    return EvidenceStore.from_jsonl(path).indication_report(compound, disease)
