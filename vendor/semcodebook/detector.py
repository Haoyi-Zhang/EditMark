from __future__ import annotations

import ast
import re

from .adaptive_ecc import decode_schedule_block
from .channel import summarize_channel
from .commitments import build_schedule_commitment, stable_structural_fingerprint
from .ecc import anchor_bit_for_nibble
from .pl_analysis import evidence_from_profile
from .negative_controls import assess_negative_control, certify_positive_support
from .protocol import CarrierEvidence, CarrierScheduleEntry, DetectionOutput, WatermarkSpec
from .variant_pool import build_adaptive_carrier_schedule, carrier_applicability_profile, describe_carrier_family


_FAMILY_ORDER: tuple[str, ...] = (
    "early_return_guard_style",
    "iteration_style",
    "accumulator_style",
    "comparison_idiom",
    "helper_extraction_style",
    "temporary_binding_style",
    "initialization_idiom",
    "return_expression_style",
)

_SUPPORTED_ECC_SCHEMES = {"soft_hamming74_audit_v1", "soft_secded84_adaptive_v1"}

_FALLBACK_PATTERNS: dict[str, dict[int, tuple[re.Pattern[str], ...]]] = {
    "iteration_style": {
        0: (re.compile(r"for\s+\w+\s+in\s+\w+"),),
        1: (re.compile(r"for\s+\w+\s+in\s+range\("),),
    },
    "helper_extraction_style": {
        0: (re.compile(r"\breturn\s+\w+\b"),),
        1: (
            re.compile(r"\bhelper_transform\s*\("),
            re.compile(r"\bhelperTransform\s*\("),
            re.compile(r"\bdef\s+helper_transform\s*\("),
            re.compile(r"\bfunc\s+helper_transform\s*\("),
            re.compile(r"\b(?:private\s+static\s+)?(?:int|long|double|boolean|String)\s+helperTransform\s*\("),
        ),
    },
}


def _probabilities(option: str, confidence: float) -> tuple[float, float]:
    bounded_confidence = max(0.0, min(1.0, confidence))
    if option == "bit_1":
        prob_one = 0.5 + 0.5 * bounded_confidence
        return round(1.0 - prob_one, 4), round(prob_one, 4)
    if option == "bit_0":
        prob_zero = 0.5 + 0.5 * bounded_confidence
        return round(prob_zero, 4), round(1.0 - prob_zero, 4)
    return 0.5, 0.5


def _evidence(
    family: str,
    option: str,
    confidence: float,
    *,
    evidence_source: str,
    matches: tuple[str, ...],
    applicable: bool,
    applicability_score: float,
    schedule_priority: float = 0.0,
) -> CarrierEvidence:
    metadata = describe_carrier_family(family)
    prob_zero, prob_one = _probabilities(option, confidence)
    return CarrierEvidence(
        family=family,
        option=option,
        confidence=round(max(0.0, min(confidence, 1.0)), 4),
        evidence_source=evidence_source,
        structural_level=str(metadata["structural_level"]),
        structural_signal=str(metadata["structural_signal"]),
        prob_zero=prob_zero,
        prob_one=prob_one,
        applicable=applicable,
        applicability_score=round(max(0.0, min(applicability_score, 1.0)), 4),
        schedule_priority=round(max(0.0, min(schedule_priority, 1.0)), 4),
        matches=matches[:4],
    )


def _unknown_evidence(family: str, *, applicable: bool, applicability_score: float, schedule_priority: float = 0.0) -> CarrierEvidence:
    return _evidence(
        family,
        "unknown",
        0.0,
        evidence_source="none",
        matches=(),
        applicable=applicable,
        applicability_score=applicability_score,
        schedule_priority=schedule_priority,
    )


def _from_existing(item: CarrierEvidence, *, applicable: bool, applicability_score: float, schedule_priority: float = 0.0) -> CarrierEvidence:
    return _evidence(
        item.family,
        item.option,
        item.confidence,
        evidence_source=item.evidence_source or "profile",
        matches=item.matches,
        applicable=applicable,
        applicability_score=applicability_score,
        schedule_priority=schedule_priority,
    )


def _ast_evidence(tree: ast.AST, family: str, *, applicable: bool, applicability_score: float) -> CarrierEvidence:
    if family == "early_return_guard_style":
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                has_else = bool(node.orelse)
                has_return = any(isinstance(child, ast.Return) for child in node.body + node.orelse)
                if has_return:
                    return _evidence(family, "bit_1" if has_else else "bit_0", 0.9, evidence_source="ast", matches=(ast.dump(node, include_attributes=False)[:80],), applicable=applicable, applicability_score=applicability_score)
    elif family == "iteration_style":
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                iterator = ast.unparse(node.iter) if hasattr(ast, "unparse") else ""
                return _evidence(family, "bit_1" if "range(" in iterator else "bit_0", 0.9, evidence_source="ast", matches=(iterator,), applicable=applicable, applicability_score=applicability_score)
    elif family == "accumulator_style":
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                return _evidence(family, "bit_1", 0.88, evidence_source="ast", matches=(ast.dump(node, include_attributes=False)[:80],), applicable=applicable, applicability_score=applicability_score)
        for node in ast.walk(tree):
            if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
                return _evidence(family, "bit_0", 0.72, evidence_source="ast", matches=(ast.dump(node, include_attributes=False)[:80],), applicable=applicable, applicability_score=applicability_score)
    elif family == "comparison_idiom":
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
                left = ast.unparse(node.left) if hasattr(ast, "unparse") else ""
                right = ast.unparse(node.comparators[0]) if hasattr(ast, "unparse") else ""
                if left == "0" and right != "0" and isinstance(node.ops[0], (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
                    return _evidence(family, "bit_1", 0.88, evidence_source="ast", matches=(ast.dump(node, include_attributes=False)[:80],), applicable=applicable, applicability_score=applicability_score)
                if right == "0" and left != "0" and isinstance(node.ops[0], (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
                    return _evidence(family, "bit_0", 0.82, evidence_source="ast", matches=(ast.dump(node, include_attributes=False)[:80],), applicable=applicable, applicability_score=applicability_score)
    elif family == "helper_extraction_style":
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if any(node.name == "helper_transform" for node in functions):
            return _evidence(family, "bit_1", 0.88, evidence_source="ast", matches=tuple(node.name for node in functions[:3]), applicable=applicable, applicability_score=applicability_score)
        if functions:
            return _evidence(family, "bit_0", 0.66, evidence_source="ast", matches=(functions[0].name,), applicable=applicable, applicability_score=applicability_score)
    elif family == "temporary_binding_style":
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "current_item":
                return _evidence(family, "bit_1", 0.88, evidence_source="ast", matches=(ast.dump(node, include_attributes=False)[:80],), applicable=applicable, applicability_score=applicability_score)
        for node in ast.walk(tree):
            if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
                return _evidence(family, "bit_0", 0.62, evidence_source="ast", matches=(ast.dump(node, include_attributes=False)[:80],), applicable=applicable, applicability_score=applicability_score)
    elif family == "initialization_idiom":
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                value_text = ast.unparse(node.value) if hasattr(ast, "unparse") else ""
                if value_text in {"int(0)", "dict()"}:
                    return _evidence(family, "bit_1", 0.9, evidence_source="ast", matches=(value_text,), applicable=applicable, applicability_score=applicability_score)
                if value_text in {"0", "{}"}:
                    return _evidence(family, "bit_0", 0.82, evidence_source="ast", matches=(value_text,), applicable=applicable, applicability_score=applicability_score)
    elif family == "return_expression_style":
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
                return _evidence(family, "bit_1", 0.86, evidence_source="ast", matches=(node.value.id,), applicable=applicable, applicability_score=applicability_score)
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and node.value is not None:
                return _evidence(family, "bit_0", 0.7, evidence_source="ast", matches=(ast.dump(node.value, include_attributes=False)[:80],), applicable=applicable, applicability_score=applicability_score)
    return _unknown_evidence(family, applicable=applicable, applicability_score=applicability_score)

def _fallback_evidence(code: str, family: str, *, applicable: bool, applicability_score: float) -> CarrierEvidence:
    options = _FALLBACK_PATTERNS.get(family)
    if options is None:
        return _unknown_evidence(family, applicable=applicable, applicability_score=applicability_score)
    scored: list[tuple[float, int, tuple[str, ...]]] = []
    for bit_value, patterns in options.items():
        hits: list[str] = []
        for pattern in patterns:
            hits.extend(match.group(0) for match in pattern.finditer(code))
        confidence = min(1.0, 0.3 + 0.2 * len(hits)) if hits else 0.0
        scored.append((confidence, bit_value, tuple(hits[:3])))
    confidence, bit_value, matches = max(scored, key=lambda item: item[0])
    if confidence <= 0.0:
        return _unknown_evidence(family, applicable=applicable, applicability_score=applicability_score)
    return _evidence(
        family,
        f"bit_{bit_value}",
        confidence,
        evidence_source="regex",
        matches=matches,
        applicable=applicable,
        applicability_score=applicability_score,
    )


def _resolve_schedule(code: str, spec: WatermarkSpec, language: str) -> tuple[CarrierScheduleEntry, ...]:
    if spec.carrier_schedule:
        return spec.carrier_schedule
    if spec.carrier_order:
        schedule: list[CarrierScheduleEntry] = []
        for slot_index, family in enumerate(spec.carrier_order):
            metadata = describe_carrier_family(family)
            applicable, applicability_score, notes = carrier_applicability_profile(code, family, language)
            schedule.append(
                CarrierScheduleEntry(
                    family=family,
                    slot_index=slot_index,
                    role="anchor" if slot_index == len(spec.carrier_order) - 1 else "data",
                    bit_index=slot_index if slot_index < 7 else None,
                    applicable=applicable,
                    applicability_score=applicability_score,
                    schedule_priority=0.0,
                    structural_level=str(metadata["structural_level"]),
                    structural_signal=str(metadata["structural_signal"]),
                    notes=notes + ("legacy_carrier_order",),
                )
            )
        return tuple(schedule)
    return build_adaptive_carrier_schedule(code, spec.carrier_key, language)


class SemCodebookDetector:
    """Snippet-only carrier decoding with typed AST/CFG/SSA-style smoke evidence."""

    def inspect(self, code: str, language: str = "python") -> tuple[CarrierEvidence, ...]:
        tree = None
        if language.lower() == "python":
            try:
                tree = ast.parse(code)
            except SyntaxError:
                tree = None
        evidence: list[CarrierEvidence] = []
        for family in _FAMILY_ORDER:
            applicable, applicability_score, _ = carrier_applicability_profile(code, family, language)
            profile_item = evidence_from_profile(code, family, language)
            if profile_item is not None:
                item = _from_existing(profile_item, applicable=applicable, applicability_score=applicability_score)
            elif tree is not None:
                item = _ast_evidence(tree, family, applicable=applicable, applicability_score=applicability_score)
            else:
                item = _unknown_evidence(family, applicable=applicable, applicability_score=applicability_score)
            if item.confidence == 0.0:
                item = _fallback_evidence(code, family, applicable=applicable, applicability_score=applicability_score)
            if item.confidence == 0.0 and profile_item is not None:
                item = _from_existing(profile_item, applicable=applicable, applicability_score=applicability_score)
            evidence.append(item)
        return tuple(evidence)

    def detect(
        self,
        code: str,
        spec: WatermarkSpec,
        *,
        language: str = "python",
        negative_control: bool = False,
    ) -> DetectionOutput:
        negative_control_declared = negative_control
        if spec.payload_bits != 4:
            raise ValueError("SemCodebook smoke path currently supports exactly 4 payload bits")
        if spec.ecc_scheme not in _SUPPORTED_ECC_SCHEMES:
            raise ValueError("SemCodebook smoke path currently supports only soft_hamming74_audit_v1 or soft_secded84_adaptive_v1")
        evidence_by_family = {item.family: item for item in self.inspect(code, language=language)}
        schedule = _resolve_schedule(code, spec, language)
        ordered_evidence: list[CarrierEvidence] = []
        for entry in schedule:
            item = evidence_by_family.get(entry.family)
            if item is None:
                item = _unknown_evidence(entry.family, applicable=entry.applicable, applicability_score=entry.applicability_score, schedule_priority=entry.schedule_priority)
            else:
                item = _evidence(
                    item.family,
                    item.option,
                    item.confidence,
                    evidence_source=item.evidence_source or "profile",
                    matches=item.matches,
                    applicable=entry.applicable,
                    applicability_score=entry.applicability_score,
                    schedule_priority=entry.schedule_priority,
                )
            ordered_evidence.append(item)
        ordered_evidence_tuple = tuple(ordered_evidence)
        channel_summary = summarize_channel(ordered_evidence_tuple, schedule)
        decode_result = decode_schedule_block(channel_summary, payload_bits=spec.payload_bits)
        current_structural_fingerprint = stable_structural_fingerprint(code, language)
        supported_symbols = sum(1 for item in channel_summary.data_observations if not item.erased)
        support_ratio = round(supported_symbols / len(channel_summary.data_observations), 4) if channel_summary.data_observations else 0.0
        positive_support = certify_positive_support(
            ordered_evidence_tuple,
            schedule,
            decode_result,
            detector_threshold=spec.detector_threshold,
            current_structural_fingerprint=current_structural_fingerprint,
        )
        negative_assessment = assess_negative_control(
            ordered_evidence_tuple,
            schedule,
            decode_result,
            detector_threshold=spec.detector_threshold,
            current_structural_fingerprint=current_structural_fingerprint,
        )
        schedule_commitment = build_schedule_commitment(spec.carrier_key, language, code, schedule)
        recovered = decode_result.value
        decoded_payload_available = recovered is not None
        expected_payload_match = decoded_payload_available and recovered == spec.wm_id
        minimum_decodable_support_ratio = 0.35
        decodable_but_unsupported = decoded_payload_available and support_ratio < minimum_decodable_support_ratio
        meaningful_partial_support = positive_support.score > 0.0 and support_ratio >= minimum_decodable_support_ratio
        if negative_control_declared:
            decision_status = "reject"
            abstain_reason = "declared_negative_control_fail_closed"
            decision_recovered = None
            decision_decoded_candidate = None
            decision_confidence = 0.0
            decision_positive_support_score = 0.0
            decision_positive_support_family_count = 0
            decision_positive_support_level_count = 0
        elif positive_support.accepted and decoded_payload_available:
            decision_status = "watermarked"
            abstain_reason = None
            decision_recovered = recovered
            decision_decoded_candidate = recovered
            decision_confidence = round(
                min(max(decode_result.confidence, 0.0), max(positive_support.score, negative_assessment.score, 0.0)),
                4,
            )
            decision_positive_support_score = positive_support.score
            decision_positive_support_family_count = positive_support.positive_support
            decision_positive_support_level_count = positive_support.rewrite_backed_level_count
        elif decoded_payload_available and not expected_payload_match and (
            meaningful_partial_support or negative_assessment.score >= 0.4
        ):
            decision_status = "abstain"
            abstain_reason = "decoded_payload_mismatch"
            decision_recovered = None
            decision_decoded_candidate = recovered
            decision_confidence = round(
                min(max(decode_result.confidence, 0.0), max(positive_support.score, negative_assessment.score, 0.0)),
                4,
            )
            decision_positive_support_score = positive_support.score
            decision_positive_support_family_count = positive_support.positive_support
            decision_positive_support_level_count = positive_support.rewrite_backed_level_count
        elif (
            (decoded_payload_available and not decodable_but_unsupported)
            or meaningful_partial_support
            or negative_assessment.score >= 0.4
        ):
            decision_status = "abstain"
            abstain_reason = positive_support.primary_failure_reason or (
                "decoded_payload_candidate_missing" if not decoded_payload_available else "positive_support_certificate_incomplete"
            )
            decision_recovered = None
            decision_decoded_candidate = recovered
            decision_confidence = round(
                min(max(decode_result.confidence, 0.0), max(positive_support.score, negative_assessment.score, 0.0)),
                4,
            )
            decision_positive_support_score = positive_support.score
            decision_positive_support_family_count = positive_support.positive_support
            decision_positive_support_level_count = positive_support.rewrite_backed_level_count
        else:
            decision_status = "reject"
            abstain_reason = None
            decision_recovered = None
            decision_decoded_candidate = recovered
            decision_confidence = round(
                min(max(decode_result.confidence, 0.0), max(positive_support.score, negative_assessment.score, 0.0)),
                4,
            )
            decision_positive_support_score = positive_support.score
            decision_positive_support_family_count = positive_support.positive_support
            decision_positive_support_level_count = positive_support.rewrite_backed_level_count
        is_watermarked = decision_status == "watermarked"
        payload_notes: tuple[str, ...] = (
            ("decoded_payload_candidate_available",)
            if decoded_payload_available
            else ("decoded_payload_candidate_missing",)
        )
        if negative_control_declared:
            payload_notes = ("decoded_payload_candidate_vetoed_for_negative_control",)
        if decoded_payload_available and not negative_control_declared:
            payload_notes = payload_notes + (
                f"decoded_payload_matches_expected:{str(expected_payload_match).lower()}",
            )
        decision_notes: tuple[str, ...] = (f"decision_status:{decision_status}",)
        if negative_control_declared:
            decision_notes = decision_notes + ("negative_control_veto:fail_closed",)
        if decision_status in {"abstain", "reject"} and abstain_reason:
            decision_notes = decision_notes + (f"abstain_reason:{abstain_reason}",)
        positive_notes = positive_support.reasons
        if negative_control_declared:
            positive_notes = ("positive_support_suppressed_for_negative_control_veto",)
        return DetectionOutput(
            is_watermarked=is_watermarked,
            wm_id_hat=decision_recovered,
            bit_error_rate=None,
            confidence=decision_confidence,
            corrected_bits=decode_result.corrected_bits,
            decoded_wm_id_candidate=decision_decoded_candidate,
            decoder_status=decode_result.decoder_status,
            erasure_count=decode_result.erasure_count,
            raw_bit_error_count=decode_result.raw_bit_error_count,
            support_count=supported_symbols,
            support_ratio=support_ratio,
            negative_control_score=negative_assessment.score,
            ber_numerator=None,
            ber_denominator=spec.payload_bits,
            carrier_evidence=ordered_evidence_tuple,
            carrier_trace=ordered_evidence_tuple,
            carrier_schedule=schedule,
            implementation_stage=spec.implementation_stage,
            notes=(
                "snippet_only_detection",
                "typed_ast_cfg_ssa_smoke_proxies",
                "adaptive_keyed_schedule_reconstruction",
                f"soft_decision_secded84:{''.join(str(bit) for bit in decode_result.codeword)}",
                f"decoder_status:{decode_result.decoder_status}",
                (
                    "positive_support_gate_fraction_suppressed_for_negative_control"
                    if negative_control_declared
                    else f"positive_support_gate_fraction:{positive_support.score:.4f}"
                ),
                f"negative_control_gate_fraction:{negative_assessment.score:.4f}",
                f"decode_confidence:{decode_result.confidence:.4f}",
                f"erasure_count:{decode_result.erasure_count}",
                f"schedule_root:{schedule_commitment.commitment_root}",
                f"schedule_context:{schedule_commitment.schedule_context_hash}",
                f"anchor_bit:{anchor_bit_for_nibble(recovered)}",
                "confidence_rule:min(decode_confidence,max(positive_support_gate_fraction,negative_control_gate_fraction))",
            )
            + payload_notes
            + decision_notes
            + positive_notes
            + negative_assessment.reasons,
            decision_status=decision_status,
            abstain_reason=abstain_reason if decision_status in {"abstain", "reject"} else None,
            positive_support_score=decision_positive_support_score,
            positive_support_family_count=decision_positive_support_family_count,
            positive_support_level_count=decision_positive_support_level_count,
        )
