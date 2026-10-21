"""Machine-checkable certificates for claims about edited positive cohorts.

The certificate does not make an empirical claim stronger.  It records whether
a requested statement follows from the supplied cohort, rule, exact counts, and
control populations.  Unsupported requests are returned as non-admissible
rather than being converted into a more favorable metric.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping
from .aggregates import (
    EXCLUSIONS, STRATA, analyze as analyze_counts, analyze_batch, count, from_pair_stratum, sha256,
)
from .analysis import report as analyze_pairs
from .constants import (
    CERTIFICATE_BUNDLE_SCHEMA, CERTIFICATE_SCHEMA, COUNT_ANALYSIS_SCHEMA, COUNT_REPORT_SCHEMA,
    COUNT_SCHEMA, COVERAGE_BUNDLE_SCHEMA, PAIR_REPORT_SCHEMA, PAIR_SCHEMA,
    STRICT_CONTRACT, UNIDENTIFIED_STRATUM_SCHEMA,
)
from .io import EvidenceError, digest, load_rows, loads, write_new_json
from .records import COMPARATORS, KINDS, Pair, checked_object, number, text

CLAIMS = (
    'positive_decision_change',
    'detection_survival',
    'edited_false_positive_rate',
    'auroc_change',
    'post_edit_provenance_probability',
)

UNIDENTIFIED_FACT_FIELDS = (
    'evidence_kind', 'analysis_contract', 'stratum', 'decision_rule',
    'attempted_pairs', 'eligible_target_pairs', 'admitted_pairs',
    'cohort_sha256', 'known_target_sha256', 'first_exclusion_counts',
)


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _fraction_value(value: Mapping[str, Any] | None) -> float | None:
    return None if value is None else float(value['decimal'])


def certify_analysis(
    analysis: Mapping[str, Any],
    requested_claim: str,
    claim_id: str = '',
    claim_text: str = '',
) -> dict[str, Any]:
    if requested_claim not in CLAIMS:
        raise EvidenceError(f'requested_claim must be one of {CLAIMS}')
    required = {
        'schema_version', 'evidence_kind', 'population', 'analysis_contract', 'cohort_sha256',
        'stratum', 'decision_rule', 'counts', 'attempted_pairs',
        'first_exclusion_counts', 'status', 'net_detection_change',
        'survival', 'survival_status', 'survival_bounds', 'membership_verified',
    }
    missing = required - set(analysis)
    if missing:
        raise EvidenceError(f'count analysis missing required fields: {sorted(missing)!r}')

    if analysis['schema_version'] != COUNT_ANALYSIS_SCHEMA:
        raise EvidenceError(f'Expected {COUNT_ANALYSIS_SCHEMA!r}')
    facts = {
        key: analysis[key] for key in (
            'population', 'evidence_kind', 'analysis_contract', 'cohort_sha256',
            'stratum', 'decision_rule', 'counts', 'attempted_pairs',
            'first_exclusion_counts',
        )
    }
    facts['schema_version'] = COUNT_SCHEMA
    facts['transitions'] = analysis.get('transitions')
    derived = analyze_counts(facts)
    mismatched = [
        key for key in derived
        if key not in {'membership_verified', 'claim_boundary', 'synthetic_warning'}
        and key in analysis
        and canonical_hash({'value': analysis[key]}) != canonical_hash({'value': derived[key]})
    ]
    if mismatched:
        raise EvidenceError(f'count analysis contradicts its exact facts: {sorted(mismatched)!r}')
    if analysis['membership_verified'] is not False:
        raise EvidenceError(
            'Supplied counts or pair reports do not authenticate cohort membership; '
            'membership_verified must be false')
    analysis = derived

    checks = {
        'positive_cohort_named': analysis['population'] == 'watermarked_positive',
        'cohort_fingerprint_present': bool(analysis['cohort_sha256']),
        'fixed_decision_rule_present': all(
            analysis['decision_rule'].get(key) not in (None, '')
            for key in ('rule_id', 'threshold', 'comparator')
        ),
        'exact_integer_counts_present': all(
            type(analysis['counts'].get(key)) is int
            for key in ('n', 'before_positive', 'after_positive')
        ),
        'exclusion_partition_present': isinstance(analysis['first_exclusion_counts'], Mapping),
        'nonempty_admitted_cohort': analysis['counts']['n'] > 0,
        'edited_negative_controls_present': False,
        'clean_negative_controls_present': False,
        'cohort_membership_authenticated': bool(analysis['membership_verified']),
    }

    prohibited = [
        'This positive-only certificate does not establish an edited false-positive rate.',
        'It does not establish AUROC change because negative cohorts and score distributions are absent.',
        'It does not establish a posterior probability of provenance because prevalence and calibration are absent.',
        'The cohort fingerprint checks identity consistency, not execution authenticity or authorship.',
        'The generated supported_statement is authoritative; free-form claim_text is unverified metadata.',
    ]
    result: dict[str, Any] = {
        'schema_version': CERTIFICATE_SCHEMA,
        'claim_id': claim_id,
        'claim_text': claim_text,
        'requested_claim': requested_claim,
        'population': analysis['population'],
        'analysis_contract': analysis['analysis_contract'],
        'cohort_sha256': analysis['cohort_sha256'],
        'stratum': analysis['stratum'],
        'decision_rule': analysis['decision_rule'],
        'counts': analysis['counts'],
        'attempted_pairs': analysis['attempted_pairs'],
        'first_exclusion_counts': analysis['first_exclusion_counts'],
        'transitions': analysis.get('transitions'),
        'analysis_status': analysis['status'],
        'membership_verified': bool(analysis['membership_verified']),
        'claim_text_verified': False,
        'requirements': checks,
        'prohibited_interpretations': prohibited,
        'evidence_kind': analysis.get('evidence_kind'),
        'synthetic_warning': analysis.get('synthetic_warning'),
    }

    if requested_claim == 'positive_decision_change':
        if checks['nonempty_admitted_cohort'] and all(
            checks[key] for key in (
                'positive_cohort_named', 'cohort_fingerprint_present',
                'fixed_decision_rule_present', 'exact_integer_counts_present',
                'exclusion_partition_present',
            )
        ):
            estimate = analysis['net_detection_change']
            result.update(
                decision='admissible',
                identification='identified',
                estimand='after-positive rate minus before-positive rate on the admitted cohort',
                estimate=estimate,
                supported_statement=(
                    'On the named jointly valid, changed, same-rule positive cohort, '
                    f'the after-minus-before positive-decision change is {_fraction_value(estimate):.6g}.'
                ),
            )
        else:
            result.update(
                decision='not_admissible', identification='undefined', estimate=None,
                supported_statement='The supplied disclosure does not identify a nonempty same-cohort change.',
            )
    elif requested_claim == 'detection_survival':
        status = analysis['survival_status']
        if status == 'identified':
            estimate = analysis['survival']
            result.update(
                decision='admissible', identification='identified',
                estimand='fraction of initially positive cohort members remaining positive after editing',
                estimate=estimate,
                supported_statement=(
                    'Detection survival is identified on the named cohort as '
                    f'{_fraction_value(estimate):.6g}.'
                ),
            )
        elif status == 'partially_identified':
            bounds = analysis['survival_bounds']
            result.update(
                decision='partially_admissible', identification='sharp_bounds',
                estimand='fraction of initially positive cohort members remaining positive after editing',
                estimate=None, bounds=bounds,
                supported_statement=(
                    'The marginals identify only a sharp finite-cohort interval for detection survival: '
                    f'[{_fraction_value(bounds["lower"]):.6g}, {_fraction_value(bounds["upper"]):.6g}].'
                ),
            )
        else:
            result.update(
                decision='not_admissible', identification=status, estimate=None,
                supported_statement='Detection survival is undefined because the cohort has no initial positives.',
            )
    elif requested_claim == 'edited_false_positive_rate':
        result.update(
            decision='not_admissible', identification='missing_negative_cohort', estimate=None,
            supported_statement=(
                'An edited false-positive rate requires an edited unwatermarked control cohort '
                'evaluated under the same rule; this disclosure contains positive pairs only.'
            ),
        )
    elif requested_claim == 'auroc_change':
        result.update(
            decision='not_admissible', identification='missing_score_distributions_and_controls', estimate=None,
            supported_statement=(
                'AUROC change requires before/after score distributions for both positive and negative cohorts. '
                'Thresholded positive counts are insufficient.'
            ),
        )
    else:
        result.update(
            decision='not_admissible', identification='missing_prevalence_and_calibration', estimate=None,
            supported_statement=(
                'A posterior provenance probability requires a deployment prevalence model and calibrated '
                'positive and negative likelihoods; this certificate supplies neither.'
            ),
        )

    core = dict(result)
    core.pop('certificate_sha256', None)
    result['certificate_sha256'] = canonical_hash(core)
    return result


def certificates_from_count_documents(
    documents: Iterable[Mapping[str, Any]], requested_claim: str
) -> list[dict[str, Any]]:
    return [certify_analysis(analysis, requested_claim) for analysis in analyze_batch(documents)]


def checked_certificate_batch(certificates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Apply the same count batch gate to every producer and verifier route.

    Callers still check each certificate's semantics before using this helper.
    Duplicate normalized count facts keep the first certificate, never pool.
    Unidentified coverage records retain their separate non-count contract.
    """
    certificates = list(certificates)
    kinds = {c.get('evidence_kind') for c in certificates}
    if len(kinds) > 1:
        raise EvidenceError('Synthetic and saved counts cannot be mixed')
    documents = []
    for item in certificates:
        if item.get('schema_version') == CERTIFICATE_SCHEMA:
            document = {key: item[key] for key in (
                'population', 'evidence_kind', 'analysis_contract', 'cohort_sha256',
                'stratum', 'decision_rule', 'counts', 'attempted_pairs', 'first_exclusion_counts')}
            document.update(schema_version=COUNT_SCHEMA, transitions=item.get('transitions'))
            documents.append(document)
    if documents:
        analyze_batch(documents)
    seen = set()
    result = []
    for item in certificates:
        if item.get('schema_version') == CERTIFICATE_SCHEMA:
            key = (item['cohort_sha256'], tuple(item['stratum'][k] for k in STRATA))
            if key in seen:
                continue
            seen.add(key)
        result.append(dict(item))
    return result


def certify_unidentified_stratum(
    facts: Mapping[str, Any], requested_claim: str,
    claim_id: str = '', claim_text: str = '',
) -> dict[str, Any]:
    """Retain an empty admitted stratum without creating a count disclosure.

    Attempted/target counts describe observed coverage, not a metric denominator.
    Neither a target fingerprint nor a missing record is an admitted cohort.
    """
    if requested_claim not in CLAIMS:
        raise EvidenceError(f'requested_claim must be one of {CLAIMS}')
    facts = checked_object(facts, 'unidentified_stratum', set(UNIDENTIFIED_FACT_FIELDS))
    if set(facts) != set(UNIDENTIFIED_FACT_FIELDS):
        raise EvidenceError('unidentified stratum requires all embedded coverage facts')
    if facts['analysis_contract'] != STRICT_CONTRACT or facts['evidence_kind'] not in KINDS:
        raise EvidenceError('unidentified stratum requires a strict, explicitly typed evidence kind')
    if not isinstance(claim_id, str) or not isinstance(claim_text, str):
        raise EvidenceError('claim_id and claim_text must be strings')
    raw_stratum = checked_object(facts['stratum'], 'stratum', set(STRATA))
    stratum = {key: text(raw_stratum.get(key), 'stratum.' + key) for key in STRATA}
    raw_rule = checked_object(
        facts['decision_rule'], 'decision_rule', {'rule_id', 'threshold', 'comparator'})
    if set(raw_rule) != {'rule_id', 'threshold', 'comparator'}:
        raise EvidenceError('unidentified stratum requires the recorded rule fields, including unknowns')
    rule = {
        'rule_id': text(raw_rule['rule_id'], 'rule_id', True),
        'threshold': number(raw_rule['threshold'], 'threshold'),
        'comparator': text(raw_rule['comparator'], 'comparator', True),
    }
    if rule['comparator'] and rule['comparator'] not in COMPARATORS:
        raise EvidenceError('unknown decision comparator')
    attempted = count(facts['attempted_pairs'], 'attempted_pairs')
    target = count(facts['eligible_target_pairs'], 'eligible_target_pairs')
    admitted = count(facts['admitted_pairs'], 'admitted_pairs')
    if attempted == 0 or admitted != 0 or facts['cohort_sha256'] is not None:
        raise EvidenceError('unidentified stratum must have attempts, no admitted pairs and no cohort hash')
    raw_exclusions = checked_object(
        facts['first_exclusion_counts'], 'first_exclusion_counts', set(EXCLUSIONS))
    exclusions = {
        key: count(value, 'exclusion.' + key) for key, value in sorted(raw_exclusions.items())
    }
    if sum(exclusions.values()) != attempted:
        raise EvidenceError('unidentified attempted pairs must equal first-exclusion counts')
    missing = sum(exclusions.get(key, 0) for key in (
        'original_decision_unavailable', 'edited_decision_unavailable'))
    if target != missing:
        raise EvidenceError('eligible empty-cohort targets must match missing-decision exclusions')
    target_hash = facts['known_target_sha256']
    if target:
        target_hash = sha256(target_hash, 'known_target_sha256')
        if not rule['rule_id'] or rule['threshold'] is None or not rule['comparator']:
            raise EvidenceError('eligible targets require a complete fixed decision rule')
    elif target_hash is not None:
        raise EvidenceError('no eligible target means no known-target fingerprint')
    result = {
        'schema_version': UNIDENTIFIED_STRATUM_SCHEMA,
        'claim_id': claim_id, 'claim_text': claim_text, 'requested_claim': requested_claim,
        'population': 'watermarked_positive', 'analysis_contract': STRICT_CONTRACT,
        'evidence_kind': facts['evidence_kind'], 'stratum': stratum, 'decision_rule': rule,
        'attempted_pairs': attempted, 'eligible_target_pairs': target, 'admitted_pairs': 0,
        'cohort_sha256': None, 'known_target_sha256': target_hash,
        'first_exclusion_counts': exclusions, 'analysis_status': 'undefined_empty_cohort',
        'decision': 'not_admissible',
        'identification': ('unidentified_missing_decisions' if target
                           else 'undefined_no_eligible_target'),
        'estimate': None, 'membership_verified': False, 'claim_text_verified': False,
        'supported_statement': (
            'No complete same-rule decision pair is admitted. Missing decisions remain '
            'unobserved; this record supplies no point estimate or metric denominator.'
            if target else
            'No jointly valid changed same-rule target is established. This record retains '
            'the exclusions and supplies no point estimate or metric denominator.'
        ),
        'prohibited_interpretations': [
            'An unavailable decision or eligibility check is not a negative observation.',
            'A known-target fingerprint is not an admitted-cohort fingerprint or execution authentication.',
            'A verified coverage record does not make the requested claim admissible.',
            'The generated supported_statement is authoritative; claim_text is unverified metadata.',
        ],
        'synthetic_warning': ('NOT AN EMPIRICAL RESULT'
                              if facts['evidence_kind'] == 'synthetic_test' else None),
    }
    result['certificate_sha256'] = canonical_hash(result)
    return result


def bundle_schema_for(certificates: Iterable[Mapping[str, Any]]) -> str:
    """Use v2 only when a bundle contains the new, non-count coverage record."""
    return (COVERAGE_BUNDLE_SCHEMA if any(
        item.get('schema_version') == UNIDENTIFIED_STRATUM_SCHEMA for item in certificates
    ) else CERTIFICATE_BUNDLE_SCHEMA)


def certificates_from_pair_report(
    report: Mapping[str, Any], requested_claim: str
) -> list[dict[str, Any]]:
    if report.get('schema_version') != PAIR_REPORT_SCHEMA:
        raise EvidenceError(f'Expected {PAIR_REPORT_SCHEMA!r}')
    evidence_kind = report.get('evidence_kind')
    certificates = []
    for index, stratum in enumerate(report.get('strata', []), 1):
        estimate = stratum.get('estimates', {}).get('changed_only', {})
        admitted = count(estimate.get('admitted_pairs'), 'admitted_pairs')
        if admitted == 0:
            if any(count(estimate.get(key), key) != 0
                   for key in ('n00', 'n01', 'n10', 'n11')):
                raise EvidenceError('empty admitted stratum cannot contain transition observations')
            if estimate.get('analysis_contract') != STRICT_CONTRACT:
                raise EvidenceError('unidentified export requires the strict analysis contract')
            facts = {
                'evidence_kind': evidence_kind, 'analysis_contract': STRICT_CONTRACT,
                'stratum': {key: stratum[key] for key in STRATA},
                'decision_rule': {
                    'rule_id': stratum['original_rule_id'],
                    'threshold': stratum['original_threshold'],
                    'comparator': stratum['original_comparator'],
                },
                **{key: estimate[key] for key in (
                    'attempted_pairs', 'eligible_target_pairs', 'admitted_pairs',
                    'cohort_sha256', 'known_target_sha256', 'first_exclusion_counts',
                )},
            }
            certificates.append(certify_unidentified_stratum(
                facts, requested_claim, claim_id=f'stratum-{index}'))
            continue
        document = from_pair_stratum(stratum, evidence_kind)
        analysis = analyze_counts(document)
        certificate = certify_analysis(
            analysis, requested_claim,
            claim_id=f'stratum-{index}',
        )
        certificates.append(certificate)
    return checked_certificate_batch(certificates)


def certificates_from_raw_pairs(
    documents: Iterable[Mapping[str, Any]], requested_claim: str
) -> list[dict[str, Any]]:
    """Reconstruct membership, source identities and eligibility from supplied rows.

    Parsing does not execute source or authenticate the observations. The same
    strict analysis and certificate arithmetic are used, not a new estimand.
    """
    return certificates_from_pair_report(
        analyze_pairs([Pair.parse(document) for document in documents]), requested_claim)


def _load_input(path: Path) -> tuple[str, list[Mapping[str, Any]] | Mapping[str, Any]]:
    if path.suffix.lower() == '.jsonl':
        rows = load_rows(path)
        kind = 'raw_pairs' if rows[0].get('schema_version') == PAIR_SCHEMA else 'counts'
        return kind, rows
    document = loads(path.read_text(encoding='utf-8-sig'))
    if isinstance(document, list):
        kind = ('raw_pairs' if document and isinstance(document[0], Mapping)
                and document[0].get('schema_version') == PAIR_SCHEMA else 'counts')
        return kind, document
    if not isinstance(document, Mapping):
        raise EvidenceError('Expected a JSON object, a list of count objects, or JSONL count objects')
    schema = document.get('schema_version')
    if schema == PAIR_SCHEMA:
        return 'raw_pairs', [document]
    if schema == PAIR_REPORT_SCHEMA:
        return 'pair_report', document
    if schema == COUNT_REPORT_SCHEMA:
        return 'analyses', document.get('analyses', [])
    if schema == COUNT_SCHEMA:
        return 'counts', [document]
    raise EvidenceError(
        f'Unsupported input schema {schema!r}; expected normalized pairs, pair report or exact cohort counts')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--claim', choices=CLAIMS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-observations', action='store_true')
    args = parser.parse_args(argv)
    try:
        input_hash = digest(args.input)
        kind, value = _load_input(args.input)
        if kind == 'pair_report':
            certificates = certificates_from_pair_report(value, args.claim)  # type: ignore[arg-type]
        elif kind == 'raw_pairs':
            certificates = certificates_from_raw_pairs(value, args.claim)  # type: ignore[arg-type]
        elif kind == 'analyses':
            certificates = [certify_analysis(item, args.claim) for item in value]  # type: ignore[arg-type]
        else:
            certificates = certificates_from_count_documents(value, args.claim)  # type: ignore[arg-type]
        certificates = checked_certificate_batch(certificates)
        if digest(args.input) != input_hash:
            raise EvidenceError('Input changed during certification')
        if not certificates:
            raise EvidenceError('No cohort or stratum coverage record is available for certification')
        if args.require_observations and any(
            certificate.get('evidence_kind') != 'saved_observation'
            for certificate in certificates
        ):
            raise EvidenceError('Synthetic evidence cannot be certified as saved observations')
        bundle = {
            'schema_version': bundle_schema_for(certificates),
            'input_file': {'name': args.input.name, 'sha256': input_hash},
            'requested_claim': args.claim,
            'certificates': certificates,
        }
        write_new_json(args.out, bundle)
        decisions = Counter(certificate['decision'] for certificate in certificates)
        print(
            'CERTIFICATES_CREATED '
            + ' '.join(f'{key}={value}' for key, value in sorted(decisions.items()))
        )
        return 0 if all(c['decision'] == 'admissible' for c in certificates) else 3
    except (EvidenceError, OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        print(f'EVIDENCE_ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
