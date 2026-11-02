"""Exact-count analysis for a named, already-fixed positive cohort.

This route does not infer cohort membership, authenticate execution, or recover
records from rounded percentages.  It checks whether supplied sufficient
statistics are internally consistent and identifies only the estimands they can
support.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
from pathlib import Path
import sys
from typing import Any, Mapping
from .bounds import intersection_bounds
from .constants import (
    COUNT_ANALYSIS_SCHEMA, COUNT_REPORT_SCHEMA, COUNT_SCHEMA, STRICT_CONTRACT,
)
from .io import EvidenceError, digest, load_rows, write_new_json
from .records import KINDS, COMPARATORS, checked_object, text, number

SCHEMA = COUNT_SCHEMA
STRATA = ('experiment_id', 'method', 'model', 'source_group', 'language', 'edit', 'variant_id')
EXCLUSIONS = (
    'unsupported', 'support_unknown', 'unchanged', 'change_unknown',
    'source_identity_missing', 'original_validation_unavailable',
    'original_checks_failed', 'original_checks_unknown',
    'edited_validation_unavailable', 'edited_checks_failed', 'edited_checks_unknown',
    'check_contract_unknown', 'check_contract_changed', 'decision_rule_unknown',
    'decision_rule_changed', 'original_decision_unavailable', 'edited_decision_unavailable',
)


def count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise EvidenceError(
            f'{name}: exact nonnegative integer required, not a percentage or Boolean')
    return value


def sha256(value: Any, name: str) -> str:
    value = text(value, name).lower()
    if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise EvidenceError(f'{name}: expected 64 hexadecimal SHA-256 characters')
    return value


def fraction(numerator: int, denominator: int) -> dict[str, Any] | None:
    if not denominator:
        return None
    value = Fraction(numerator, denominator)
    return {
        'numerator': value.numerator,
        'denominator': value.denominator,
        'decimal': float(value),
    }


def analyze(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one exact-count disclosure and derive point estimates and sharp bounds."""
    data = checked_object(
        document,
        'cohort_counts',
        {
            'schema_version', 'population', 'evidence_kind', 'analysis_contract',
            'cohort_sha256', 'stratum', 'decision_rule', 'counts', 'transitions',
            'attempted_pairs', 'first_exclusion_counts',
        },
    )
    for key, expected in (
        ('schema_version', SCHEMA),
        ('population', 'watermarked_positive'),
        ('analysis_contract', STRICT_CONTRACT),
    ):
        if data.get(key) != expected:
            raise EvidenceError(f'{key}: expected {expected!r}')
    evidence_kind = text(data.get('evidence_kind'), 'evidence_kind')
    if evidence_kind not in KINDS:
        raise EvidenceError(f'evidence_kind must be one of {KINDS}')
    cohort_hash = sha256(data.get('cohort_sha256'), 'cohort_sha256')
    raw_stratum = checked_object(data.get('stratum'), 'stratum', set(STRATA))
    stratum = {key: text(raw_stratum.get(key), f'stratum.{key}') for key in STRATA}
    raw_rule = checked_object(
        data.get('decision_rule'), 'decision_rule', {'rule_id', 'threshold', 'comparator'})
    decision_rule = {
        'rule_id': text(raw_rule.get('rule_id'), 'rule_id'),
        'threshold': number(raw_rule.get('threshold'), 'threshold'),
        'comparator': text(raw_rule.get('comparator'), 'comparator'),
    }
    if decision_rule['threshold'] is None or decision_rule['comparator'] not in COMPARATORS:
        raise EvidenceError('A complete fixed decision rule is required')
    raw_counts = checked_object(
        data.get('counts'), 'counts', {'n', 'before_positive', 'after_positive'})
    counts = {
        key: count(raw_counts.get(key), 'counts.' + key)
        for key in ('n', 'before_positive', 'after_positive')
    }
    n = counts['n']
    before_positive = counts['before_positive']
    after_positive = counts['after_positive']
    if max(after_positive, before_positive) > n:
        raise EvidenceError('A positive count exceeds the fixed cohort size')

    attempted = count(data.get('attempted_pairs'), 'attempted_pairs')
    if 'first_exclusion_counts' not in data or data['first_exclusion_counts'] is None:
        raise EvidenceError(
            'first_exclusion_counts: an explicit object, possibly empty, is required')
    raw_exclusions = checked_object(
        data.get('first_exclusion_counts'), 'first_exclusion_counts', set(EXCLUSIONS))
    exclusions = {
        key: count(value, 'exclusion.' + key)
        for key, value in raw_exclusions.items()
    }
    if attempted != n + sum(exclusions.values()):
        raise EvidenceError('attempted_pairs must equal n plus first-exclusion counts')

    transitions = None
    if data.get('transitions') is not None:
        raw_transitions = checked_object(
            data['transitions'], 'transitions', {'n00', 'n01', 'n10', 'n11'})
        transitions = {
            key: count(raw_transitions.get(key), 'transitions.' + key)
            for key in ('n00', 'n01', 'n10', 'n11')
        }
        if (
            sum(transitions.values()) != n
            or transitions['n10'] + transitions['n11'] != before_positive
            or transitions['n01'] + transitions['n11'] != after_positive
        ):
            raise EvidenceError('Joint transition table contradicts cohort counts')
        overlap_lower = overlap_upper = transitions['n11']
    else:
        overlap_lower, overlap_upper = intersection_bounds(
            n, before_positive, after_positive)

    if before_positive == 0:
        survival = None
        survival_status = 'undefined_no_initial_detections'
        survival_bounds = {'lower': None, 'upper': None}
    else:
        survival = (
            fraction(overlap_lower, before_positive)
            if overlap_lower == overlap_upper else None
        )
        survival_status = 'identified' if overlap_lower == overlap_upper else 'partially_identified'
        survival_bounds = {
            'lower': fraction(overlap_lower, before_positive),
            'upper': fraction(overlap_upper, before_positive),
        }

    change = fraction(after_positive - before_positive, n)
    loss = fraction(before_positive - after_positive, n)
    return {
        'schema_version': COUNT_ANALYSIS_SCHEMA,
        'evidence_kind': evidence_kind,
        'population': data['population'],
        'analysis_contract': data['analysis_contract'],
        'cohort_sha256': cohort_hash,
        'stratum': stratum,
        'decision_rule': decision_rule,
        'counts': counts,
        'transitions': transitions,
        'attempted_pairs': attempted,
        'first_exclusion_counts': exclusions,
        'status': 'identified' if n else 'undefined_empty_cohort',
        'tpr_before': fraction(before_positive, n),
        'tpr_after': fraction(after_positive, n),
        'net_detection_change': change,
        'net_detection_loss': loss,
        'loss': loss,
        'survival': survival,
        'survival_status': survival_status,
        'survival_bounds': survival_bounds,
        'claim_boundary': (
            'These counts describe one asserted positive cohort under one fixed rule. '
            'They do not identify FPR, AUROC, calibration, prevalence, or posterior provenance probability.'
        ),
        'membership_verified': False,
        'synthetic_warning': 'NOT AN EMPIRICAL RESULT' if evidence_kind == 'synthetic_test' else None,
    }


def from_pair_stratum(stratum: Mapping[str, Any], evidence_kind: str) -> dict[str, Any]:
    """Export sufficient counts from one nonempty strict pair-analysis stratum."""
    estimate = stratum['estimates']['changed_only']
    if not estimate['admitted_pairs'] or not estimate['cohort_sha256']:
        raise EvidenceError('An empty pair cohort has no established cohort fingerprint to export')
    document = {
        'schema_version': SCHEMA,
        'population': 'watermarked_positive',
        'evidence_kind': evidence_kind,
        'analysis_contract': STRICT_CONTRACT,
        'cohort_sha256': estimate['cohort_sha256'],
        'stratum': {key: stratum[key] for key in STRATA},
        'decision_rule': {
            'rule_id': stratum['original_rule_id'],
            'threshold': stratum['original_threshold'],
            'comparator': stratum['original_comparator'],
        },
        'counts': {
            'n': estimate['admitted_pairs'],
            'before_positive': estimate['n10'] + estimate['n11'],
            'after_positive': estimate['n01'] + estimate['n11'],
        },
        'transitions': {key: estimate[key] for key in ('n00', 'n01', 'n10', 'n11')},
        'attempted_pairs': estimate['attempted_pairs'],
        'first_exclusion_counts': estimate['first_exclusion_counts'],
    }
    analyze(document)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-observations', action='store_true')
    parser.add_argument('--require-estimate', action='store_true')
    args = parser.parse_args(argv)
    try:
        input_hash = digest(args.input)
        documents = load_rows(args.input)
        analyses = [analyze(document) for document in documents]
        if digest(args.input) != input_hash:
            raise EvidenceError('Input changed during analysis')
        kinds = {analysis['evidence_kind'] for analysis in analyses}
        if len(kinds) != 1:
            raise EvidenceError('Synthetic and saved counts cannot be mixed')
        if args.require_observations and kinds != {'saved_observation'}:
            raise EvidenceError('Synthetic counts cannot be reported as saved observations')
        keys = [
            (analysis['cohort_sha256'], tuple(analysis['stratum'][key] for key in STRATA))
            for analysis in analyses
        ]
        if len(keys) != len(set(keys)):
            raise EvidenceError('Repeated cohort/stratum; no silent deduplication or pooling')
        result = {
            'schema_version': COUNT_REPORT_SCHEMA,
            'input_files': [{'name': args.input.name, 'sha256': input_hash}],
            'analyses': analyses,
            'no_pooling': True,
            'membership_verified': False,
        }
        write_new_json(args.out, result)
        print(f'COUNT_REPORT_CREATED cohorts={len(analyses)} kind={next(iter(kinds))}')
        return 3 if args.require_estimate and any(a['counts']['n'] == 0 for a in analyses) else 0
    except (EvidenceError, OSError, UnicodeError, ValueError) as exc:
        print(f'EVIDENCE_ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
