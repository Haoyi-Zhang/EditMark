from __future__ import annotations

from dataclasses import dataclass

from .channel import ChannelSummary, soft_anchor_observation, soft_observations
from .ecc import anchor_bit_for_nibble, encode_nibble, soft_decode_nibble_audit


@dataclass(frozen=True, slots=True)
class AdaptiveDecodeResult:
    value: int
    confidence: float
    corrected_bits: int
    corrected_positions: tuple[int, ...]
    decoder_status: str
    erasure_count: int
    raw_bit_error_count: int
    post_correction_bit_error_rate: float | None
    codeword: tuple[int, ...]
    anchor_reliability: float | None = None


def decode_schedule_block(summary: ChannelSummary, payload_bits: int = 4) -> AdaptiveDecodeResult:
    audit = soft_decode_nibble_audit(soft_observations(summary), soft_anchor_observation(summary))
    hard_bits = tuple(1 if item.prob_one >= item.prob_zero else 0 for item in summary.data_observations[:7])
    corrected_positions = tuple(index for index, (left, right) in enumerate(zip(hard_bits, audit.codeword, strict=True)) if left != right)
    raw_bit_error_count = len(corrected_positions)
    status = "decoded_clean"
    if summary.erasure_count > 0:
        status = "decoded_with_erasures"
    if corrected_positions:
        status = "decoded_corrected" if summary.erasure_count == 0 else "decoded_corrected_with_erasures"
    if summary.erasure_count >= 4 and audit.confidence < 0.55:
        status = "uncorrectable_detected"
    return AdaptiveDecodeResult(
        value=audit.value,
        confidence=audit.confidence,
        corrected_bits=audit.corrected_bits,
        corrected_positions=corrected_positions,
        decoder_status=status,
        erasure_count=summary.erasure_count,
        raw_bit_error_count=raw_bit_error_count,
        post_correction_bit_error_rate=None,
        codeword=audit.codeword,
        anchor_reliability=audit.anchor_reliability,
    )


def oracle_post_correction_bit_error_rate(decoded_value: int, expected_value: int, payload_bits: int) -> float:
    mask = (1 << payload_bits) - 1
    return round(bin((decoded_value ^ expected_value) & mask).count("1") / payload_bits, 4)


def expected_anchor_bit(decoded_value: int) -> int:
    return anchor_bit_for_nibble(decoded_value)


def expected_codeword(decoded_value: int) -> tuple[int, ...]:
    return encode_nibble(decoded_value)
