"""Verify EditMark claim certificates and, optionally, their evidence basis.

Standalone verification checks the certificate hash and re-derives every
reported decision, estimate, interval, and generated statement from the exact
facts embedded in the certificate.  Supplying ``--basis`` additionally checks
the recorded input digest and regenerates the certificate records from their
raw-pair, pair-report or exact-count basis. V2 requires full stratum coverage;
legacy v1 retains its cohort contract and explicitly reports omitted strata.

This verifier checks integrity and logical sufficiency.  It does not attest that
an experiment was executed honestly, that task oracles are correct, or that a
cohort fingerprint was produced by a trusted environment.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

from .aggregates import analyze as analyze_counts, sha256
from .certificate import (
    canonical_hash,
    certificates_from_count_documents,
    certificates_from_pair_report,
    certificates_from_raw_pairs,
    certify_analysis,
    certify_unidentified_stratum,
    UNIDENTIFIED_FACT_FIELDS,
    _load_input,
)
from .constants import (
    CERTIFICATE_BUNDLE_SCHEMA,
    CERTIFICATE_SCHEMA,
    COUNT_SCHEMA,
    COVERAGE_BUNDLE_SCHEMA,
    UNIDENTIFIED_STRATUM_SCHEMA,
)
from .io import EvidenceError, digest, loads, write_new_json
from .records import checked_object, text


def _core_hash(certificate: Mapping[str, Any]) -> str:
    core = dict(certificate)
    supplied = core.pop('certificate_sha256', None)
    if not isinstance(supplied, str):
        raise EvidenceError('certificate_sha256 is required')
    actual = canonical_hash(core)
    if supplied != actual:
        raise EvidenceError('certificate hash does not match its canonical content')
    return actual


def _count_document(certificate: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        'population', 'evidence_kind', 'analysis_contract', 'cohort_sha256',
        'stratum', 'decision_rule', 'counts', 'attempted_pairs',
        'first_exclusion_counts',
    )
    missing = [key for key in required if key not in certificate]
    if missing:
        raise EvidenceError(f'certificate missing embedded evidence fields: {missing!r}')
    return {
        'schema_version': COUNT_SCHEMA,
        'population': certificate['population'],
        'evidence_kind': certificate['evidence_kind'],
        'analysis_contract': certificate['analysis_contract'],
        'cohort_sha256': certificate['cohort_sha256'],
        'stratum': certificate['stratum'],
        'decision_rule': certificate['decision_rule'],
        'counts': certificate['counts'],
        'transitions': certificate.get('transitions'),
        'attempted_pairs': certificate['attempted_pairs'],
        'first_exclusion_counts': certificate['first_exclusion_counts'],
    }


def verify_certificate(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Verify one certificate from its embedded sufficient facts.

    The result deliberately reports whether record membership was only asserted
    or was regenerated from a supplied basis.  This function performs the
    former; :func:`verify_bundle_against_basis` performs the latter.
    """
    schema = certificate.get('schema_version')
    if schema not in (CERTIFICATE_SCHEMA, UNIDENTIFIED_STRATUM_SCHEMA):
        raise EvidenceError(f'expected {CERTIFICATE_SCHEMA!r}')
    _core_hash(certificate)
    if certificate.get('membership_verified') is not False:
        raise EvidenceError('Supplied certificates do not authenticate cohort membership')
    if schema == UNIDENTIFIED_STRATUM_SCHEMA:
        missing = set(UNIDENTIFIED_FACT_FIELDS) - set(certificate)
        if missing:
            raise EvidenceError(f'unidentified record missing embedded facts: {sorted(missing)!r}')
        expected = certify_unidentified_stratum(
            {key: certificate[key] for key in UNIDENTIFIED_FACT_FIELDS},
            certificate.get('requested_claim', ''),
            claim_id=certificate.get('claim_id', ''), claim_text=certificate.get('claim_text', ''),
        )
    else:
        analysis = analyze_counts(_count_document(certificate))
        expected = certify_analysis(
            analysis,
            str(certificate.get('requested_claim', '')),
            claim_id=str(certificate.get('claim_id', '')),
            claim_text=str(certificate.get('claim_text', '')),
        )
    if dict(certificate) != expected:
        mismatched = sorted(
            key for key in set(certificate) | set(expected)
            if certificate.get(key) != expected.get(key)
        )
        raise EvidenceError(
            'certificate content is not the deterministic result of its embedded facts; '
            f'mismatched fields: {mismatched!r}'
        )
    return {
        'certificate_sha256': certificate['certificate_sha256'],
        'requested_claim': certificate['requested_claim'],
        'decision': certificate['decision'],
        'embedded_facts_verified': True,
        'basis_verified': False,
    }


def verify_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    bundle = checked_object(
        bundle, 'certificate_bundle',
        {'schema_version', 'input_file', 'requested_claim', 'certificates'},
    )
    schema = bundle.get('schema_version')
    if schema not in (CERTIFICATE_BUNDLE_SCHEMA, COVERAGE_BUNDLE_SCHEMA):
        raise EvidenceError(f'expected {CERTIFICATE_BUNDLE_SCHEMA!r}')
    certificates = bundle.get('certificates')
    if not isinstance(certificates, list) or not certificates:
        raise EvidenceError('certificate bundle must contain a nonempty certificates array')
    requested = bundle.get('requested_claim')
    if not isinstance(requested, str) or not requested:
        raise EvidenceError('requested_claim is required')
    seen: set[str] = set()
    rows = []
    for certificate in certificates:
        if not isinstance(certificate, Mapping):
            raise EvidenceError('each certificate must be an object')
        if certificate.get('requested_claim') != requested:
            raise EvidenceError('bundle and certificate requested_claim values disagree')
        if (schema == CERTIFICATE_BUNDLE_SCHEMA
                and certificate.get('schema_version') != CERTIFICATE_SCHEMA):
            raise EvidenceError('unidentified stratum records require the v2 bundle schema')
        result = verify_certificate(certificate)
        cert_hash = result['certificate_sha256']
        if cert_hash in seen:
            raise EvidenceError('duplicate certificate hash in bundle')
        seen.add(cert_hash)
        rows.append(result)
    input_file = checked_object(
        bundle.get('input_file'), 'input_file', {'name', 'sha256'},
    )
    text(input_file.get('name'), 'input_file.name')
    recorded_digest = sha256(input_file.get('sha256'), 'input_file.sha256')
    if input_file['sha256'] != recorded_digest:
        raise EvidenceError('input_file.sha256: expected lowercase SHA-256')
    return {
        'schema_version': 'editmark-certificate-verification',
        'requested_claim': requested,
        'certificates_verified': len(rows),
        'basis_verified': False,
        'certificates': rows,
    }


def _regenerate_from_basis(basis: Path, requested_claim: str) -> list[dict[str, Any]]:
    kind, value = _load_input(basis)
    if kind == 'pair_report':
        return certificates_from_pair_report(value, requested_claim)  # type: ignore[arg-type]
    if kind == 'raw_pairs':
        return certificates_from_raw_pairs(value, requested_claim)  # type: ignore[arg-type]
    if kind == 'analyses':
        return [certify_analysis(item, requested_claim) for item in value]  # type: ignore[arg-type]
    return certificates_from_count_documents(value, requested_claim)  # type: ignore[arg-type]


def verify_bundle_against_basis(
    bundle: Mapping[str, Any], basis: Path
) -> dict[str, Any]:
    result = verify_bundle(bundle)
    recorded = bundle['input_file']
    actual_digest = digest(basis)
    if actual_digest != recorded['sha256']:
        raise EvidenceError('basis file SHA-256 does not match the bundle input digest')
    regenerated = _regenerate_from_basis(basis, str(bundle['requested_claim']))
    if digest(basis) != actual_digest:
        raise EvidenceError('Input basis changed during verification')
    # v1 promised complete nonempty-cohort certificates, not a record for every
    # stratum. Preserve that historical contract, but expose its omitted strata.
    omitted = []
    expected = regenerated
    if bundle['schema_version'] == CERTIFICATE_BUNDLE_SCHEMA:
        omitted = [item for item in regenerated
                   if item['schema_version'] == UNIDENTIFIED_STRATUM_SCHEMA]
        expected = [item for item in regenerated
                    if item['schema_version'] == CERTIFICATE_SCHEMA]
    if expected != bundle['certificates']:
        raise EvidenceError('bundle certificates do not match regeneration from the supplied basis')
    result['basis_verified'] = True
    result['basis_coverage_complete'] = not omitted
    result['unidentified_strata'] = omitted
    result['basis_file'] = {'name': basis.name, 'sha256': actual_digest}
    kind, _ = _load_input(basis)
    if digest(basis) != actual_digest:
        raise EvidenceError('Input basis changed during verification')
    result['basis_kind'] = kind
    result['raw_pair_membership_rederived'] = kind == 'raw_pairs'
    for row in result['certificates']:
        row['basis_verified'] = True
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--basis', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    try:
        document = loads(args.bundle.read_text(encoding='utf-8-sig'))
        if not isinstance(document, Mapping):
            raise EvidenceError('certificate bundle must be a JSON object')
        result = (
            verify_bundle_against_basis(document, args.basis)
            if args.basis is not None else verify_bundle(document)
        )
        if args.out is not None:
            write_new_json(args.out, result)
        mode = 'basis+certificate' if result['basis_verified'] else 'certificate'
        coverage = (
            f' stratum_coverage=incomplete omitted={len(result["unidentified_strata"])}'
            if result.get('basis_coverage_complete') is False else ''
        )
        print(
            f'CERTIFICATES_VERIFIED count={result["certificates_verified"]} mode={mode}{coverage}'
        )
        return 0
    except (EvidenceError, OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        print(f'EVIDENCE_ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
