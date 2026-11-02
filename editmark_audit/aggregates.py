"""Exact-count analysis for a named, already-fixed positive cohort.

This route does not infer cohort membership, authenticate execution, or recover
records from rounded percentages.  It checks whether supplied sufficient
statistics are internally consistent and identifies only the estimands they can
support.
"""
from __future__ import annotations
import argparse
import json
from fractions import Fraction
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping
from .bounds import intersection_bounds
from .analysis import MODES, exact_fraction, rate
from .constants import (
    COUNT_ANALYSIS_SCHEMA, COUNT_REPORT_SCHEMA, COUNT_SCHEMA, PAIR_REPORT_SCHEMA, STRICT_CONTRACT,
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


def _required_object(
    value: Any, label: str, fields: Iterable[str], optional: Iterable[str] = (),
) -> Mapping[str, Any]:
    fields = set(fields)
    if not isinstance(value, Mapping):
        raise EvidenceError(f'{label}: expected an object')
    result = checked_object(value, label, fields | set(optional))
    if not fields <= set(result):
        raise EvidenceError(f'{label}: missing fields {sorted(fields - set(result))!r}')
    return result


def _require_json_value(observed: Any, expected: Any, label: str) -> None:
    """Do not let Python's True == 1 or 1 == 1.0 hide a stale report."""
    try:
        equal = json.dumps(observed, sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f'{label}: invalid JSON value') from exc
    if not equal:
        raise EvidenceError(f'{label}: contradicts rederived pair-report facts')


def _pair_stratum(stratum: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _required_object(stratum, 'pair stratum', (
        *STRATA, 'original_rule_id', 'original_threshold', 'original_comparator', 'estimates'))
    for key in STRATA:
        text(value[key], 'stratum.' + key)
    for key in ('original_rule_id', 'original_comparator'):
        if not isinstance(value[key], str):
            raise EvidenceError(f'{key}: expected a string, including an empty unknown')
        text(value[key], key, True)
    if value['original_comparator'] not in ('', *COMPARATORS):
        raise EvidenceError('unknown original comparator')
    number(value['original_threshold'], 'original_threshold')
    _required_object(value['estimates'], 'pair estimates', MODES)
    return value


def _pair_estimate(estimate: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    estimate = _required_object(estimate, 'pair estimate', (
        'analysis_contract', 'mode', 'status', 'attempted_pairs', 'eligible_target_pairs',
        'admitted_pairs', 'cohort_sha256', 'known_target_sha256', 'first_exclusion_counts',
        'n00', 'n01', 'n10', 'n11', 'tpr_before', 'tpr_after', 'net_detection_change',
        'net_detection_change_exact', 'net_detection_loss', 'loss', 'survival',
        'survival_denominator', 'marginal_coverage', 'distinct_originals',
        'functional_preservation', 'missing_decision_change_bounds', 'missing_decision_bounds'))
    contract = STRICT_CONTRACT if mode == 'changed_only' else mode
    if estimate['mode'] != mode or estimate['analysis_contract'] != contract:
        raise EvidenceError('pair export requires the declared mode and analysis contract')
    attempted = count(estimate.get('attempted_pairs'), 'attempted_pairs')
    target = count(estimate.get('eligible_target_pairs'), 'eligible_target_pairs')
    admitted = count(estimate.get('admitted_pairs'), 'admitted_pairs')
    if not 0 <= admitted <= target <= attempted:
        raise EvidenceError('pair cohort sizes contradict strict eligibility')
    marginal = _required_object(estimate['marginal_coverage'], 'marginal coverage', (
        'supported', 'support_unknown', 'changed', 'unchanged', 'change_unknown'))
    coverage = {key: count(marginal.get(key), 'marginal_coverage.' + key)
                for key in ('supported', 'support_unknown', 'changed', 'unchanged', 'change_unknown')}
    if (coverage['changed'] + coverage['unchanged'] + coverage['change_unknown'] != attempted
            or coverage['supported'] + coverage['support_unknown'] > attempted
            or target > coverage['supported']
            or (mode == 'changed_only' and target > coverage['changed'])):
        raise EvidenceError('strict pair cohort contradicts marginal coverage')
    if not isinstance(estimate['first_exclusion_counts'], Mapping):
        raise EvidenceError('first_exclusion_counts: explicit object required')
    exclusions = checked_object(estimate['first_exclusion_counts'], 'first_exclusion_counts', set(EXCLUSIONS))
    exclusions = {key: count(value, 'exclusion.' + key) for key, value in exclusions.items()}
    missing = sum(exclusions.get(key, 0) for key in ('original_decision_unavailable', 'edited_decision_unavailable'))
    if attempted != admitted + sum(exclusions.values()) or target != admitted + missing:
        raise EvidenceError('strict pair cohort contradicts exclusion partition')
    unsupported = attempted - coverage['supported'] - coverage['support_unknown']
    if (exclusions.get('unsupported', 0) != unsupported
            or exclusions.get('support_unknown', 0) != coverage['support_unknown']):
        raise EvidenceError('marginal coverage contradicts support exclusion partition')
    # Support is the first gate. Any unchanged/unknown-change members not excluded
    # at the next gate must fit entirely in the already unsupported population.
    remainder = 0
    for field, active in (('unchanged', mode == 'changed_only'),
                          ('change_unknown', mode != 'all_supported')):
        excluded = exclusions.get(field, 0)
        if (not active and excluded) or excluded > coverage[field]:
            raise EvidenceError('marginal coverage contradicts change exclusion partition')
        if active:
            remainder += coverage[field] - excluded
    if remainder > attempted - coverage['supported']:
        raise EvidenceError('marginal coverage contradicts ordered change gate')
    if mode == 'all_supported' and exclusions.get('source_identity_missing', 0):
        raise EvidenceError('all_supported does not exclude missing source identities')

    joint = {key: count(estimate[key], key) for key in ('n00', 'n01', 'n10', 'n11')}
    if sum(joint.values()) != admitted:
        raise EvidenceError('pair transitions contradict admitted_pairs')
    for key, size in (('cohort_sha256', admitted), ('known_target_sha256', target)):
        value = estimate[key]
        if size:
            _require_json_value(value, sha256(value, key), key)
        elif value is not None:
            raise EvidenceError(f'{key}: empty population requires null fingerprint')
    if target == admitted and estimate['known_target_sha256'] != estimate['cohort_sha256']:
        raise EvidenceError('complete target and admitted cohort fingerprints disagree')
    before = joint['n10'] + joint['n11']
    after = joint['n01'] + joint['n11']
    derived = {
        'status': 'identified' if admitted else 'undefined_empty_cohort',
        'tpr_before': rate(before, admitted), 'tpr_after': rate(after, admitted),
        'net_detection_change': rate(after - before, admitted),
        'net_detection_change_exact': exact_fraction(after - before, admitted),
        'net_detection_loss': rate(before - after, admitted), 'loss': rate(before - after, admitted),
        'survival': rate(joint['n11'], before), 'survival_denominator': before,
    }
    for key, expected in derived.items():
        _require_json_value(estimate[key], expected, key)

    original = _required_object(estimate['distinct_originals'], 'distinct_originals', (
        'count', 'known_check_results', 'passing', 'pass_rate'))
    distinct, known, passing = (count(original[k], 'distinct_originals.' + k)
                                for k in ('count', 'known_check_results', 'passing'))
    # One fixed edit/variant stratum has one deduplicated pair per original key.
    if not target <= passing <= known <= distinct == attempted:
        raise EvidenceError('inconsistent distinct-original counts')
    if (exclusions.get('original_checks_failed', 0) > known - passing
            or exclusions.get('original_validation_unavailable', 0)
            + exclusions.get('original_checks_unknown', 0) > attempted - known):
        raise EvidenceError('original-check exclusions contradict original coverage')
    _require_json_value(original['pass_rate'], rate(passing, known), 'original pass_rate')
    functional = _required_object(estimate['functional_preservation'], 'functional_preservation', (
        'numerator', 'denominator', 'rate'))
    num, den = (count(functional[k], 'functional_preservation.' + k)
                for k in ('numerator', 'denominator'))
    if not target <= num <= den <= min(coverage['supported'], passing):
        raise EvidenceError('functional preservation contradicts eligible target coverage')
    functional_passes = target + sum(exclusions.get(k, 0) for k in (
        'decision_rule_unknown', 'decision_rule_changed'))
    if num != functional_passes or den > num + exclusions.get('edited_checks_failed', 0):
        raise EvidenceError('functional preservation contradicts ordered validation exclusions')
    if mode == 'changed_only' and den > coverage['changed']:
        raise EvidenceError('functional preservation exceeds changed coverage')
    _require_json_value(functional['rate'], rate(num, den), 'functional preservation rate')

    bound_fields = ('target_pairs', 'complete_pairs', 'lower_numerator', 'upper_numerator',
                    'denominator', 'lower', 'upper', 'estimand', 'interpretation')
    loss_bounds = _required_object(estimate['missing_decision_bounds'], 'loss bounds', bound_fields)
    lo, hi = loss_bounds['lower_numerator'], loss_bounds['upper_numerator']
    if type(lo) is not int or type(hi) is not int:
        raise EvidenceError('missing-bound numerators require exact signed integers')
    complete_loss = before - after
    # Edited-only unavailability leaves one observed Boolean decision, so its
    # completion range has width one. Width two is possible only among the
    # original-unavailable bucket, which also contains all both-missing pairs.
    original_missing = exclusions.get('original_decision_unavailable', 0)
    if not (complete_loss - missing <= lo <= complete_loss <= hi <= complete_loss + missing
            and missing <= hi - lo <= missing + original_missing):
        raise EvidenceError('missing-decision bounds contradict complete transitions and missingness')
    for key, lower, upper, estimand in (
        ('missing_decision_bounds', lo, hi, 'net_detection_loss'),
        ('missing_decision_change_bounds', -hi, -lo, 'net_detection_change'),
    ):
        bounds = _required_object(estimate[key], key, bound_fields)
        expected = dict(target_pairs=target, complete_pairs=admitted, denominator=target,
                        lower_numerator=lower, upper_numerator=upper, lower=rate(lower, target),
                        upper=rate(upper, target), estimand=estimand)
        for field, value in expected.items():
            _require_json_value(bounds[field], value, key + '.' + field)
        text(bounds['interpretation'], key + '.interpretation')
    return estimate


def strict_pair_estimate(stratum: Mapping[str, Any]) -> Mapping[str, Any]:
    """Check all strict report facts before reducing them to a count disclosure."""
    stratum = _pair_stratum(stratum)
    estimate = _pair_estimate(stratum['estimates']['changed_only'], 'changed_only')
    if estimate['eligible_target_pairs'] and (
            not stratum['original_rule_id'] or stratum['original_threshold'] is None
            or not stratum['original_comparator']):
        raise EvidenceError('eligible pair targets require a complete original decision rule')
    return estimate


def _nested_pair_estimates(
    narrow: Mapping[str, Any], broad: Mapping[str, Any], removed_gates: set[str],
) -> None:
    """A relaxed gate can add rows, never rewrite an existing observation.

    Check the disclosed partitions and nested sufficient facts, not external
    membership. Unknown Boolean completion intervals must also extend by only
    the newly eligible rows' contributions.
    """
    for key in ('attempted_pairs', 'marginal_coverage', 'distinct_originals'):
        _require_json_value(broad[key], narrow[key], broad['mode'] + '.' + key)
    for key in EXCLUSIONS:
        if key not in removed_gates and (
                broad['first_exclusion_counts'].get(key, 0)
                < narrow['first_exclusion_counts'].get(key, 0)):
            raise EvidenceError('relaxed pair mode loses an existing exclusion observation: ' + key)
    for key in ('n00', 'n01', 'n10', 'n11', 'eligible_target_pairs', 'admitted_pairs'):
        if broad[key] < narrow[key]:
            raise EvidenceError('relaxed pair mode loses an existing cohort observation: ' + key)
    for size, fingerprint in (('admitted_pairs', 'cohort_sha256'),
                              ('eligible_target_pairs', 'known_target_sha256')):
        if broad[size] == narrow[size]:
            _require_json_value(broad[fingerprint], narrow[fingerprint], 'nested ' + fingerprint)
        elif broad[fingerprint] == narrow[fingerprint]:
            raise EvidenceError('nested cohort sizes disagree under one fingerprint')
    nf, bf = narrow['functional_preservation'], broad['functional_preservation']
    if (bf['numerator'] < nf['numerator'] or bf['denominator'] < nf['denominator']
            or bf['denominator'] - bf['numerator'] < nf['denominator'] - nf['numerator']):
        raise EvidenceError('relaxed pair mode rewrites existing functional outcomes')
    added_failures = (broad['first_exclusion_counts'].get('edited_checks_failed', 0)
                      - narrow['first_exclusion_counts'].get('edited_checks_failed', 0))
    if ((bf['denominator'] - bf['numerator']) - (nf['denominator'] - nf['numerator'])
            > added_failures):
        raise EvidenceError('relaxed pair mode invents functional checks on existing rows')
    before_missing = (broad['first_exclusion_counts'].get('original_decision_unavailable', 0)
                      - narrow['first_exclusion_counts'].get('original_decision_unavailable', 0))
    after_missing = (broad['first_exclusion_counts'].get('edited_decision_unavailable', 0)
                     - narrow['first_exclusion_counts'].get('edited_decision_unavailable', 0))
    missing = before_missing + after_missing
    complete_loss = (broad['n10'] - narrow['n10']) - (broad['n01'] - narrow['n01'])
    nb, bb = narrow['missing_decision_bounds'], broad['missing_decision_bounds']
    low = bb['lower_numerator'] - nb['lower_numerator'] - complete_loss
    high = bb['upper_numerator'] - nb['upper_numerator'] - complete_loss
    # Each newly missing row contributes width one, or width two when both
    # decisions are missing (which belongs to the original-unavailable bucket).
    both_missing = high - low - missing
    if (not 0 <= both_missing <= before_missing
            or not -missing <= low <= -both_missing
            or not both_missing <= high <= missing):
        raise EvidenceError('relaxed pair mode rewrites existing missing-decision contributions')


def validate_pair_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Shared producer/basis gate, independent of v1/v2 certificate presentation.

    These are checks on supplied report consistency, not raw membership or truth.
    """
    report = _required_object(report, 'pair report', (
        'schema_version', 'evidence_kind', 'input_pair_count', 'unique_pair_count',
        'identical_duplicates_removed', 'stratum_count', 'strata', 'policy',
        'claim_boundary', 'synthetic_warning'), optional=('input_files',))
    if 'input_files' in report:
        inputs = report['input_files']
        if not isinstance(inputs, list) or not inputs:
            raise EvidenceError('input_files: a nonempty array of file identities is required')
        for item in inputs:
            item = _required_object(item, 'input_files entry', ('name', 'sha256'))
            text(item['name'], 'input_files.name')
            _require_json_value(item['sha256'], sha256(item['sha256'], 'input_files.sha256'),
                                'input_files.sha256')
    if report['schema_version'] != PAIR_REPORT_SCHEMA:
        raise EvidenceError(f'Expected {PAIR_REPORT_SCHEMA!r}')
    total, unique, duplicates, size = (count(report[k], k) for k in (
        'input_pair_count', 'unique_pair_count', 'identical_duplicates_removed', 'stratum_count'))
    strata = report['strata']
    if (not isinstance(strata, list) or size != len(strata) or total != unique + duplicates
            or (not unique and total)):
        raise EvidenceError('pair report cohort/stratum top-level partition disagrees')
    if report['evidence_kind'] not in (KINDS if unique else ('empty',)):
        raise EvidenceError('pair report evidence kind disagrees with population')
    _require_json_value(report['synthetic_warning'],
                        'NOT AN EMPIRICAL RESULT' if report['evidence_kind'] == 'synthetic_test' else None,
                        'synthetic_warning')
    text(report['policy'], 'policy'); text(report['claim_boundary'], 'claim_boundary')
    seen = set()
    attempted = 0
    for stratum in strata:
        stratum = _pair_stratum(stratum)
        identity = tuple(stratum[k] for k in STRATA) + (
            stratum['original_rule_id'], number(stratum['original_threshold'], 'original_threshold'),
            stratum['original_comparator'])
        if identity in seen:
            raise EvidenceError('duplicate pair report cohort/stratum identity')
        seen.add(identity)
        estimate = strict_pair_estimate(stratum)
        if not estimate['attempted_pairs']:
            raise EvidenceError('pair report stratum must retain at least one attempt')
        attempted += estimate['attempted_pairs']
        previous = estimate
        removed = ({'unchanged'}, {'change_unknown', 'source_identity_missing'})
        for mode, removed_gates in zip(MODES[1:], removed):
            diagnostic = _pair_estimate(stratum['estimates'][mode], mode)
            if diagnostic['eligible_target_pairs'] and (
                    not stratum['original_rule_id'] or stratum['original_threshold'] is None
                    or not stratum['original_comparator']):
                raise EvidenceError('eligible targets in every mode require a complete original rule')
            _nested_pair_estimates(previous, diagnostic, removed_gates)
            previous = diagnostic
    if attempted != unique:
        raise EvidenceError('pair report cohort/stratum attempted coverage disagrees with unique_pair_count')
    return report


def from_pair_stratum(stratum: Mapping[str, Any], evidence_kind: str) -> dict[str, Any]:
    """Export sufficient counts from one nonempty strict pair-analysis stratum."""
    estimate = strict_pair_estimate(stratum)
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


def analyze_batch(documents: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate all facts before publication; exact repeats get one cohort vote.

    A cohort/stratum identity cannot name different rules, counts, transitions
    or exclusion partitions. No counts are summed and no missing facts merged.
    """
    unique = {}
    kinds = set()
    for document in documents:
        analysis = analyze(document)
        kinds.add(analysis['evidence_kind'])
        key = (analysis['cohort_sha256'], tuple(analysis['stratum'][k] for k in STRATA))
        canonical = json.dumps(analysis, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if key in unique and canonical != unique[key][0]:
            raise EvidenceError('Conflicting cohort/stratum facts: rule, counts, transitions or partition')
        unique.setdefault(key, (canonical, analysis))
    if len(kinds) != 1:
        raise EvidenceError('A nonempty batch of one evidence kind is required; synthetic and saved counts cannot be mixed')
    return [value[1] for value in unique.values()]


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
        analyses = analyze_batch(documents)
        if digest(args.input) != input_hash:
            raise EvidenceError('Input changed during analysis')
        kinds = {analysis['evidence_kind'] for analysis in analyses}
        if len(kinds) != 1:
            raise EvidenceError('Synthetic and saved counts cannot be mixed')
        if args.require_observations and kinds != {'saved_observation'}:
            raise EvidenceError('Synthetic counts cannot be reported as saved observations')
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
