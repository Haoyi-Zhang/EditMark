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
from .aggregates import analyze as analyze_counts, from_pair_stratum
from .constants import (
    CERTIFICATE_BUNDLE_SCHEMA, CERTIFICATE_SCHEMA, COUNT_REPORT_SCHEMA,
    COUNT_SCHEMA, PAIR_REPORT_SCHEMA,
)
from .io import EvidenceError, digest, load_rows, loads, write_new_json

CLAIMS = (
    'positive_decision_change',
    'detection_survival',
    'edited_false_positive_rate',
    'auroc_change',
    'post_edit_provenance_probability',
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
        'schema_version', 'population', 'analysis_contract', 'cohort_sha256',
        'stratum', 'decision_rule', 'counts', 'attempted_pairs',
        'first_exclusion_counts', 'status', 'net_detection_change',
        'survival', 'survival_status', 'survival_bounds', 'membership_verified',
    }
    missing = required - set(analysis)
    if missing:
        raise EvidenceError(f'count analysis missing required fields: {sorted(missing)!r}')

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
    return [certify_analysis(analyze_counts(document), requested_claim) for document in documents]


def certificates_from_pair_report(
    report: Mapping[str, Any], requested_claim: str
) -> list[dict[str, Any]]:
    if report.get('schema_version') != PAIR_REPORT_SCHEMA:
        raise EvidenceError(f'Expected {PAIR_REPORT_SCHEMA!r}')
    evidence_kind = report.get('evidence_kind')
    certificates = []
    for index, stratum in enumerate(report.get('strata', []), 1):
        estimate = stratum.get('estimates', {}).get('changed_only', {})
        if not estimate.get('admitted_pairs'):
            continue
        document = from_pair_stratum(stratum, evidence_kind)
        analysis = analyze_counts(document)
        analysis['membership_verified'] = True
        certificate = certify_analysis(
            analysis, requested_claim,
            claim_id=f'stratum-{index}',
        )
        certificates.append(certificate)
    return certificates


def _load_input(path: Path) -> tuple[str, list[Mapping[str, Any]] | Mapping[str, Any]]:
    if path.suffix.lower() == '.jsonl':
        rows = load_rows(path)
        return 'counts', rows
    document = loads(path.read_text(encoding='utf-8-sig'))
    if isinstance(document, list):
        return 'counts', document
    if not isinstance(document, Mapping):
        raise EvidenceError('Expected a JSON object, a list of count objects, or JSONL count objects')
    schema = document.get('schema_version')
    if schema == PAIR_REPORT_SCHEMA:
        return 'pair_report', document
    if schema == COUNT_REPORT_SCHEMA:
        return 'analyses', document.get('analyses', [])
    if schema == COUNT_SCHEMA:
        return 'counts', [document]
    raise EvidenceError(
        f'Unsupported input schema {schema!r}; expected pair report or exact cohort counts')


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
        elif kind == 'analyses':
            certificates = [certify_analysis(item, args.claim) for item in value]  # type: ignore[arg-type]
        else:
            certificates = certificates_from_count_documents(value, args.claim)  # type: ignore[arg-type]
        if digest(args.input) != input_hash:
            raise EvidenceError('Input changed during certification')
        if not certificates:
            raise EvidenceError('No nonempty cohort is available for certification')
        if args.require_observations and any(
            certificate.get('evidence_kind') != 'saved_observation'
            for certificate in certificates
        ):
            raise EvidenceError('Synthetic evidence cannot be certified as saved observations')
        bundle = {
            'schema_version': CERTIFICATE_BUNDLE_SCHEMA,
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
