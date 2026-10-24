from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SoftDecodeAudit:
    value: int
    corrected_bits: int
    confidence: float
    codeword: tuple[int, ...]
    margin: float
    erasure_count: int
    symbol_reliability: tuple[float, ...]
    anchor_reliability: float | None = None


def _bit(byte: int, index: int) -> int:
    return (byte >> index) & 1


def _hamming74_encode(nibble: int) -> int:
    d1 = _bit(nibble, 0)
    d2 = _bit(nibble, 1)
    d3 = _bit(nibble, 2)
    d4 = _bit(nibble, 3)
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return (
        (p1 << 0)
        | (p2 << 1)
        | (d1 << 2)
        | (p3 << 3)
        | (d2 << 4)
        | (d3 << 5)
        | (d4 << 6)
    )


def _hamming74_decode(codeword: int) -> tuple[int, bool]:
    b = [_bit(codeword, i) for i in range(7)]
    s1 = b[0] ^ b[2] ^ b[4] ^ b[6]
    s2 = b[1] ^ b[2] ^ b[5] ^ b[6]
    s3 = b[3] ^ b[4] ^ b[5] ^ b[6]
    syndrome = s1 | (s2 << 1) | (s3 << 2)
    corrected = False
    if syndrome:
        corrected = True
        index = syndrome - 1
        codeword ^= 1 << index
        b = [_bit(codeword, i) for i in range(7)]
    nibble = (b[2] << 0) | (b[4] << 1) | (b[5] << 2) | (b[6] << 3)
    return nibble, corrected


def encode_nibble(value: int) -> tuple[int, ...]:
    if not 0 <= value <= 0x0F:
        raise ValueError("value must be an unsigned nibble")
    codeword = _hamming74_encode(value)
    return tuple(_bit(codeword, index) for index in range(7))


def decode_nibble(bits: tuple[int, ...]) -> tuple[int, int]:
    if len(bits) != 7:
        raise ValueError("expected exactly seven codeword bits")
    codeword = sum((bit & 1) << index for index, bit in enumerate(bits))
    value, corrected = _hamming74_decode(codeword)
    return value, int(corrected)


def anchor_bit_for_nibble(value: int) -> int:
    if not 0 <= value <= 0x0F:
        raise ValueError("value must be an unsigned nibble")
    return sum(encode_nibble(value)) & 1


def soft_decode_nibble_audit(
    observations: tuple[tuple[float, float], ...],
    anchor_observation: tuple[float, float] | None = None,
) -> SoftDecodeAudit:
    if len(observations) != 7:
        raise ValueError("expected exactly seven soft observations")

    candidates: list[tuple[float, int, float, tuple[int, ...]]] = []
    for nibble in range(16):
        codeword_bits = encode_nibble(nibble)
        score = 0.0
        match_probs: list[float] = []
        for code_bit, (prob_zero, prob_one) in zip(codeword_bits, observations, strict=True):
            matched = max(1e-9, prob_one if code_bit else prob_zero)
            score += math.log(matched)
            match_probs.append(matched)
        if anchor_observation is not None:
            anchor_bit = anchor_bit_for_nibble(nibble)
            anchor_match = max(1e-9, anchor_observation[anchor_bit])
            score += 0.35 * math.log(anchor_match)
            match_probs.append(anchor_match)
        candidates.append((score, nibble, sum(match_probs) / len(match_probs), codeword_bits))

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_nibble, average_match, best_codeword = candidates[0]
    runner_up_score = candidates[1][0] if len(candidates) > 1 else best_score
    margin = max(0.0, best_score - runner_up_score)
    margin_confidence = 1.0 - math.exp(-margin)
    symbol_reliability = tuple(round(abs(prob_one - prob_zero), 4) for prob_zero, prob_one in observations)
    erasure_count = sum(1 for reliability in symbol_reliability if reliability < 0.12)
    anchor_reliability = None
    if anchor_observation is not None:
        anchor_observation_reliability = abs(anchor_observation[1] - anchor_observation[0])
        if anchor_observation_reliability >= 0.12:
            anchor_reliability = anchor_observation[anchor_bit_for_nibble(best_nibble)]
    confidence = min(
        1.0,
        max(
            0.0,
            min(
                average_match,
                margin_confidence,
                anchor_reliability if anchor_reliability is not None else 1.0,
                1.0 - (erasure_count / len(observations)),
            ),
        ),
    )

    hard_bits = tuple(1 if prob_one >= prob_zero else 0 for prob_zero, prob_one in observations)
    corrected_bits = sum(1 for hard_bit, code_bit in zip(hard_bits, best_codeword, strict=True) if hard_bit != code_bit)
    if anchor_observation is not None:
        anchor_hard = 1 if anchor_observation[1] >= anchor_observation[0] else 0
        corrected_bits += int(anchor_hard != anchor_bit_for_nibble(best_nibble))
    return SoftDecodeAudit(
        value=best_nibble,
        corrected_bits=corrected_bits,
        confidence=round(confidence, 4),
        codeword=best_codeword,
        margin=round(margin, 4),
        erasure_count=erasure_count,
        symbol_reliability=symbol_reliability,
        anchor_reliability=None if anchor_reliability is None else round(anchor_reliability, 4),
    )


def soft_decode_nibble(
    observations: tuple[tuple[float, float], ...],
    anchor_observation: tuple[float, float] | None = None,
) -> tuple[int, int, float, tuple[int, ...]]:
    audit = soft_decode_nibble_audit(observations, anchor_observation)
    return audit.value, audit.corrected_bits, audit.confidence, audit.codeword


def encode_byte(value: int) -> tuple[int, int]:
    if not 0 <= value <= 0xFF:
        raise ValueError("value must be an unsigned byte")
    low = _hamming74_encode(value & 0x0F)
    high = _hamming74_encode((value >> 4) & 0x0F)
    return low, high


def decode_byte(codewords: tuple[int, int]) -> tuple[int, int]:
    if len(codewords) != 2:
        raise ValueError("expected exactly two Hamming(7,4) codewords")
    low, low_fixed = _hamming74_decode(codewords[0])
    high, high_fixed = _hamming74_decode(codewords[1])
    return low | (high << 4), int(low_fixed) + int(high_fixed)
