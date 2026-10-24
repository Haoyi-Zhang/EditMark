from __future__ import annotations

from dataclasses import dataclass

from .applicability import structural_family_witness
from .semantic_validator import ValidationTrace, stable_code_digest, validate_semantics
from .typed_ast import stable_ast_fingerprint, summarize_typed_ast


@dataclass(frozen=True, slots=True)
class RewriteCertificate:
    family: str
    bit_value: int
    language: str
    original_digest: str
    transformed_digest: str
    original_fingerprint: str
    transformed_fingerprint: str
    witness_preserved: bool
    compile_ok: bool
    tests_ok: bool
    validator_trace: ValidationTrace
    notes: tuple[str, ...] = ()


def certify_rewrite_candidate(
    reference_code: str,
    transformed_code: str,
    *,
    family: str,
    bit_value: int,
    language: str,
    tests: tuple[str, ...],
) -> RewriteCertificate:
    trace = validate_semantics(transformed_code, language, tests)
    witness = structural_family_witness(transformed_code, family, language)
    witness_preserved = witness is not None and witness[0] == f"bit_{bit_value}"
    return RewriteCertificate(
        family=family,
        bit_value=bit_value,
        language=language,
        original_digest=stable_code_digest(reference_code),
        transformed_digest=stable_code_digest(transformed_code),
        original_fingerprint=stable_ast_fingerprint(summarize_typed_ast(reference_code, language)),
        transformed_fingerprint=stable_ast_fingerprint(summarize_typed_ast(transformed_code, language)),
        witness_preserved=witness_preserved,
        compile_ok=trace.compile_ok,
        tests_ok=trace.tests_ok,
        validator_trace=trace,
        notes=(
            "rewrite_certificate",
            f"validator:{trace.validator}",
            "witness_preserved" if witness_preserved else "witness_missing",
        ),
    )


def certificate_is_valid(certificate: RewriteCertificate) -> bool:
    return certificate.compile_ok and certificate.tests_ok and certificate.witness_preserved
