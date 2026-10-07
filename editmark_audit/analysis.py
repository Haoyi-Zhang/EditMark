"""Same-cohort, same-rule analysis with explicit eligibility and missingness."""
from __future__ import annotations
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from typing import Any, Iterable
from .bounds import decision_difference_range
from .constants import PAIR_REPORT_SCHEMA, STRICT_CONTRACT
from .records import Pair, deduplicate
from .io import EvidenceError

MODES = ('changed_only', 'include_known_noops', 'all_supported')


def precondition(pair: Pair, mode: str = 'changed_only') -> str | None:
    """Return the first failed eligibility condition, before decision availability."""
    if mode not in MODES:
        raise EvidenceError(f'Unknown mode {mode!r}')
    if pair.supported is not True:
        return 'unsupported' if pair.supported is False else 'support_unknown'
    if mode == 'changed_only' and pair.changed is not True:
        return 'unchanged' if pair.changed is False else 'change_unknown'
    if mode == 'include_known_noops' and pair.changed is None:
        return 'change_unknown'
    if mode != 'all_supported' and (
            not pair.original.source_sha256 or not pair.edited.source_sha256):
        return 'source_identity_missing'
    for name, validation in (
        ('original', pair.original.validation), ('edited', pair.edited.validation)
    ):
        if validation.available is not True:
            return name + '_validation_unavailable'
        if validation.passed is not True:
            return name + '_checks_failed' if validation.passed is False else name + '_checks_unknown'
    original_validation = pair.original.validation
    edited_validation = pair.edited.validation
    if not original_validation.contract_id or not edited_validation.contract_id:
        return 'check_contract_unknown'
    if original_validation.contract_id != edited_validation.contract_id:
        return 'check_contract_changed'
    before = pair.original.detection
    after = pair.edited.detection
    if not before.rule_known or not after.rule_known:
        return 'decision_rule_unknown'
    if before.rule != after.rule:
        return 'decision_rule_changed'
    return None


def exclusion(pair: Pair, mode: str = 'changed_only') -> str | None:
    reason = precondition(pair, mode)
    if reason:
        return reason
    for name, detection in (
        ('original', pair.original.detection), ('edited', pair.edited.detection)
    ):
        if not detection.observed:
            return name + '_decision_unavailable'
    return None


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def exact_fraction(numerator: int, denominator: int) -> dict[str, Any] | None:
    if not denominator:
        return None
    value = Fraction(numerator, denominator)
    return {
        'numerator': value.numerator,
        'denominator': value.denominator,
        'decimal': float(value),
    }


def summarize(pairs: Iterable[Pair], mode: str = 'changed_only') -> dict[str, Any]:
    if mode not in MODES:
        raise EvidenceError(f'Unknown mode {mode!r}')
    rows = list(pairs)
    excluded: Counter[str] = Counter()
    joint: Counter[str] = Counter()
    target: list[Pair] = []
    admitted: list[Pair] = []
    for pair in rows:
        if precondition(pair, mode) is None:
            target.append(pair)
        reason = exclusion(pair, mode)
        if reason:
            excluded[reason] += 1
        else:
            admitted.append(pair)
            joint[
                f'n{int(pair.original.detection.detected)}{int(pair.edited.detection.detected)}'
            ] += 1

    admitted.sort(key=lambda pair: pair.key)
    payload = [
        (
            pair.key,
            pair.original.source_sha256,
            pair.edited.source_sha256,
            pair.original.validation.contract_id,
            pair.original.detection.rule,
            pair.original.detection.detected,
            pair.edited.detection.detected,
        )
        for pair in admitted
    ]
    cohort_sha256 = (
        hashlib.sha256(
            json.dumps(payload, separators=(',', ':'), ensure_ascii=True).encode()
        ).hexdigest()
        if admitted else None
    )
    counts = {key: joint[key] for key in ('n00', 'n01', 'n10', 'n11')}
    n = sum(counts.values())

    lower_difference = 0
    upper_difference = 0
    for pair in target:
        before = pair.original.detection
        after = pair.edited.detection
        low, high = decision_difference_range(
            before.detected if before.observed else None,
            after.detected if after.observed else None,
        )
        lower_difference += low
        upper_difference += high

    target_payload = [
        (
            pair.key,
            pair.original.source_sha256,
            pair.edited.source_sha256,
            pair.original.validation.contract_id,
            pair.original.detection.rule,
            pair.original.detection.detected if pair.original.detection.observed else None,
            pair.edited.detection.detected if pair.edited.detection.observed else None,
        )
        for pair in sorted(target, key=lambda pair: pair.key)
    ]
    target_sha256 = (
        hashlib.sha256(
            json.dumps(target_payload, separators=(',', ':'), ensure_ascii=True).encode()
        ).hexdigest()
        if target else None
    )

    originals: dict[tuple[str, ...], Pair] = {pair.original_key: pair for pair in rows}
    known_original_checks = [
        pair for pair in originals.values()
        if pair.original.validation.available is True
        and pair.original.validation.passed is not None
    ]
    functional_candidates: list[Pair] = []
    functional_passes = 0
    for pair in rows:
        change_ok = (
            mode == 'all_supported'
            or (mode == 'include_known_noops' and pair.changed is not None)
            or (mode == 'changed_only' and pair.changed is True)
        )
        hash_ok = (
            mode == 'all_supported'
            or bool(pair.original.source_sha256 and pair.edited.source_sha256)
        )
        before_validation = pair.original.validation
        after_validation = pair.edited.validation
        if (
            pair.supported is True
            and change_ok
            and hash_ok
            and before_validation.available is True
            and before_validation.passed is True
            and after_validation.available is True
            and after_validation.passed is not None
            and before_validation.contract_id
            and before_validation.contract_id == after_validation.contract_id
        ):
            functional_candidates.append(pair)
            functional_passes += int(after_validation.passed)

    before_positive = counts['n10'] + counts['n11']
    after_positive = counts['n01'] + counts['n11']
    difference_sum = after_positive - before_positive
    loss_sum = -difference_sum
    return {
        'analysis_contract': STRICT_CONTRACT if mode == 'changed_only' else mode,
        'mode': mode,
        'status': 'identified' if n else 'undefined_empty_cohort',
        'attempted_pairs': len(rows),
        'eligible_target_pairs': len(target),
        'admitted_pairs': n,
        'cohort_sha256': cohort_sha256,
        'known_target_sha256': target_sha256,
        'first_exclusion_counts': dict(sorted(excluded.items())),
        **counts,
        'tpr_before': rate(before_positive, n),
        'tpr_after': rate(after_positive, n),
        'net_detection_change': rate(difference_sum, n),
        'net_detection_change_exact': exact_fraction(difference_sum, n),
        'net_detection_loss': rate(loss_sum, n),
        'loss': rate(loss_sum, n),  # concise compatibility field; same value as net_detection_loss
        'survival': rate(counts['n11'], before_positive),
        'survival_denominator': before_positive,
        'marginal_coverage': {
            'supported': sum(pair.supported is True for pair in rows),
            'support_unknown': sum(pair.supported is None for pair in rows),
            'changed': sum(pair.changed is True for pair in rows),
            'unchanged': sum(pair.changed is False for pair in rows),
            'change_unknown': sum(pair.changed is None for pair in rows),
        },
        'distinct_originals': {
            'count': len(originals),
            'known_check_results': len(known_original_checks),
            'passing': sum(
                pair.original.validation.passed is True for pair in known_original_checks),
            'pass_rate': rate(
                sum(pair.original.validation.passed is True for pair in known_original_checks),
                len(known_original_checks),
            ),
        },
        'functional_preservation': {
            'numerator': functional_passes,
            'denominator': len(functional_candidates),
            'rate': rate(functional_passes, len(functional_candidates)),
        },
        'missing_decision_change_bounds': {
            'target_pairs': len(target),
            'complete_pairs': n,
            'lower_numerator': -upper_difference,
            'upper_numerator': -lower_difference,
            'denominator': len(target),
            'lower': rate(-upper_difference, len(target)),
            'upper': rate(-lower_difference, len(target)),
            'estimand': 'net_detection_change',
            'interpretation': (
                'Sharp finite-target identification interval, not a confidence interval. '
                'Only jointly valid, changed, same-rule pairs form this target; '
                'unknown validation remains outside it.'
            ),
        },
        'missing_decision_bounds': {
            'target_pairs': len(target),
            'complete_pairs': n,
            'lower_numerator': lower_difference,
            'upper_numerator': upper_difference,
            'denominator': len(target),
            'lower': rate(lower_difference, len(target)),
            'upper': rate(upper_difference, len(target)),
            'estimand': 'net_detection_loss',
            'interpretation': (
                'Sharp finite-target identification interval, not a confidence interval. '
                'This is the sign-reversed form of missing_decision_change_bounds.'
            ),
        },
    }


def report(pairs: list[Pair]) -> dict[str, Any]:
    rows, duplicates = deduplicate(pairs)
    kinds = {pair.evidence_kind for pair in rows}
    if len(kinds) > 1:
        raise EvidenceError('Synthetic fixtures and saved observations cannot share one analysis')
    groups: dict[tuple[Any, ...], list[Pair]] = defaultdict(list)
    for pair in rows:
        groups[pair.stratum].append(pair)
    labels = (
        'experiment_id', 'method', 'model', 'source_group', 'language', 'edit',
        'variant_id', 'original_rule_id', 'original_threshold', 'original_comparator',
    )
    strata = [
        dict(zip(labels, key))
        | {'estimates': {mode: summarize(group, mode) for mode in MODES}}
        for key, group in sorted(groups.items(), key=lambda item: repr(item[0]))
    ]
    return {
        'schema_version': PAIR_REPORT_SCHEMA,
        'evidence_kind': next(iter(kinds), 'empty'),
        'input_pair_count': len(pairs),
        'unique_pair_count': len(rows),
        'identical_duplicates_removed': duplicates,
        'stratum_count': len(strata),
        'strata': strata,
        'policy': (
            'Within-stratum estimates only: no default pooling over methods, models, '
            'datasets, edits, or detector rules.'
        ),
        'claim_boundary': (
            'Positive-pair records can identify same-rule positive-decision change. '
            'They cannot identify false-positive rates, AUROC change, calibration, '
            'source prevalence, or posterior provenance probability without additional cohorts.'
        ),
        'synthetic_warning': 'NOT AN EMPIRICAL RESULT' if kinds == {'synthetic_test'} else None,
    }
