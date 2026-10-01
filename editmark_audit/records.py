"""Strictly typed before/after observations for edited-code evaluation.

A parsed row is a recorded assertion.  Hashes detect byte inconsistency; they
do not authenticate who produced or executed the program.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping
from .constants import PAIR_SCHEMA
from .io import EvidenceError

SCHEMA = PAIR_SCHEMA
KINDS = ('saved_observation', 'synthetic_test')
COMPARATORS = ('ge', 'gt', 'le', 'lt')


def text(x: Any, name: str, optional: bool = False) -> str:
    if optional and x is None:
        return ''
    if not isinstance(x, str) or (not optional and not x.strip()):
        raise EvidenceError(f'{name}: expected a nonempty string')
    if x != x.strip():
        raise EvidenceError(f'{name}: leading/trailing whitespace in an identifier is not normalized')
    try:
        x.encode('utf-8')
    except UnicodeError as exc:
        raise EvidenceError(f'{name}: invalid Unicode scalar value') from exc
    return x


def flag(x: Any, name: str) -> bool | None:
    if x is None or type(x) is bool:
        return x
    raise EvidenceError(f'{name}: expected JSON true, false or null')


def number(x: Any, name: str) -> float | None:
    if x is None:
        return None
    if type(x) not in (int, float):
        raise EvidenceError(f'{name}: expected a finite numeric value or null')
    try:
        value = float(x)
    except OverflowError as exc:
        raise EvidenceError(f'{name}: numeric value exceeds finite float range') from exc
    if type(x) is int and value != x:
        raise EvidenceError(
            f'{name}: integer cannot be represented exactly as binary64; '
            'supply a saved decision without a score or preserve the intended rule explicitly')
    if not math.isfinite(value):
        raise EvidenceError(f'{name}: expected a finite numeric value or null')
    return value


def obj(x: Any, name: str) -> Mapping[str, Any]:
    if x is None:
        return {}
    if not isinstance(x, Mapping):
        raise EvidenceError(f'{name}: expected an object')
    return x


def checked_object(x: Any, name: str, allowed: set[str]) -> Mapping[str, Any]:
    d = obj(x, name)
    unknown = set(d) - allowed
    if unknown:
        raise EvidenceError(
            f'{name}: unknown fields {sorted(unknown)!r}; '
            'misspellings are not treated as missing observations')
    return d


def compare_score(score: float, threshold: float, comparator: str) -> bool:
    if comparator == 'ge':
        return score >= threshold
    if comparator == 'gt':
        return score > threshold
    if comparator == 'le':
        return score <= threshold
    if comparator == 'lt':
        return score < threshold
    raise EvidenceError(f'Unknown comparator {comparator!r}')


@dataclass(frozen=True, slots=True)
class Validation:
    available: bool | None
    passed: bool | None
    contract_id: str

    @classmethod
    def parse(cls, x: Any, label: str) -> 'Validation':
        d = checked_object(x, label, {'available', 'passed', 'contract_id'})
        value = cls(
            flag(d.get('available'), label + '.available'),
            flag(d.get('passed'), label + '.passed'),
            text(d.get('contract_id'), label + '.contract_id', True),
        )
        if value.available is not True and value.passed is not None:
            raise EvidenceError(
                f'{label}: unavailable or unknown validation cannot contain a pass/fail observation')
        return value


@dataclass(frozen=True, slots=True)
class Detection:
    available: bool | None
    detected: bool | None
    score: float | None
    threshold: float | None
    comparator: str
    rule_id: str

    @classmethod
    def parse(cls, x: Any, label: str) -> 'Detection':
        d = checked_object(
            x, label,
            {'available', 'detected', 'score', 'threshold', 'comparator', 'rule_id'},
        )
        value = cls(
            flag(d.get('available'), label + '.available'),
            flag(d.get('detected'), label + '.detected'),
            number(d.get('score'), label + '.score'),
            number(d.get('threshold'), label + '.threshold'),
            text(d.get('comparator'), label + '.comparator', True),
            text(d.get('rule_id'), label + '.rule_id', True),
        )
        if value.comparator and value.comparator not in COMPARATORS:
            raise EvidenceError(f'{label}: comparator must be ge, gt, le or lt')
        if value.available is not True and (value.detected is not None or value.score is not None):
            raise EvidenceError(
                f'{label}: unavailable or unknown detection cannot contain a score or decision')
        if (value.score is not None and value.threshold is not None and value.comparator
                and value.detected is not None):
            if compare_score(value.score, value.threshold, value.comparator) != value.detected:
                raise EvidenceError(
                    f'{label}: saved decision contradicts score, threshold and comparator')
        return value

    @property
    def rule(self) -> tuple[str, float | None, str]:
        return self.rule_id, self.threshold, self.comparator

    @property
    def rule_known(self) -> bool:
        return bool(self.rule_id and self.comparator and self.threshold is not None)

    @property
    def observed(self) -> bool:
        return self.available is True and self.detected is not None


@dataclass(frozen=True, slots=True)
class Program:
    source_sha256: str
    validation: Validation
    detection: Detection

    @classmethod
    def parse(cls, x: Any, label: str) -> 'Program':
        d = checked_object(x, label, {'source', 'source_sha256', 'validation', 'detection'})
        digest = text(d.get('source_sha256'), label + '.source_sha256', True).lower()
        if digest and (len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
            raise EvidenceError(f'{label}: invalid SHA-256')
        if 'source' in d:
            if not isinstance(d['source'], str):
                raise EvidenceError(f'{label}.source must be a string')
            try:
                actual = hashlib.sha256(d['source'].encode('utf-8')).hexdigest()
            except UnicodeError as exc:
                raise EvidenceError(f'{label}.source: invalid Unicode scalar value') from exc
            if digest and digest != actual:
                raise EvidenceError(f'{label}: source and hash disagree')
            digest = actual
        return cls(
            digest,
            Validation.parse(d.get('validation'), label + '.validation'),
            Detection.parse(d.get('detection'), label + '.detection'),
        )


@dataclass(frozen=True, slots=True)
class Pair:
    evidence_kind: str
    experiment_id: str
    method: str
    model: str
    source_group: str
    language: str
    sample_id: str
    replicate_id: str
    edit: str
    variant_id: str
    cluster_id: str
    supported: bool | None
    changed: bool | None
    original: Program
    edited: Program

    @classmethod
    def parse(cls, document: Mapping[str, Any]) -> 'Pair':
        d = checked_object(
            document,
            'pair',
            {
                'schema_version', 'population', 'evidence_kind', 'experiment_id',
                'method', 'model', 'source_group', 'language', 'sample_id',
                'replicate_id', 'edit', 'variant_id', 'cluster_id', 'supported',
                'changed', 'original', 'edited',
            },
        )
        if d.get('population') != 'watermarked_positive':
            raise EvidenceError(
                'This pair contract requires population=watermarked_positive; controls use a separate cohort')
        if d.get('schema_version') != SCHEMA:
            raise EvidenceError(f'schema_version: expected {SCHEMA!r}')
        fields = {
            key: text(d.get(key), key)
            for key in (
                'evidence_kind', 'experiment_id', 'method', 'model', 'source_group',
                'language', 'sample_id', 'replicate_id', 'edit', 'variant_id',
            )
        }
        if fields['evidence_kind'] not in KINDS:
            raise EvidenceError(f'evidence_kind must be one of {KINDS}')
        original = Program.parse(d.get('original'), 'original')
        edited = Program.parse(d.get('edited'), 'edited')
        changed = flag(d.get('changed'), 'changed')
        if original.source_sha256 and edited.source_sha256:
            by_hash = original.source_sha256 != edited.source_sha256
            if changed is not None and changed != by_hash:
                raise EvidenceError('Changed flag contradicts program hashes')
            changed = by_hash
        return cls(
            **fields,
            cluster_id=text(d.get('cluster_id'), 'cluster_id', True),
            supported=flag(d.get('supported'), 'supported'),
            changed=changed,
            original=original,
            edited=edited,
        )

    @property
    def original_key(self) -> tuple[str, ...]:
        return (
            self.evidence_kind, self.experiment_id, self.method, self.model,
            self.source_group, self.language, self.sample_id, self.replicate_id,
        )

    @property
    def key(self) -> tuple[str, ...]:
        return self.original_key + (self.edit, self.variant_id)

    @property
    def stratum(self) -> tuple[Any, ...]:
        return (
            self.experiment_id, self.method, self.model, self.source_group,
            self.language, self.edit, self.variant_id,
            self.original.detection.rule_id, self.original.detection.threshold,
            self.original.detection.comparator,
        )


def deduplicate(pairs: list[Pair]) -> tuple[list[Pair], int]:
    """Remove byte-identical duplicate records and reject conflicting identities."""
    unique: dict[tuple[str, ...], Pair] = {}
    originals: dict[tuple[str, ...], Program] = {}
    clusters: dict[tuple[str, ...], str] = {}
    removed = 0
    for pair in pairs:
        if pair.original_key in originals and originals[pair.original_key] != pair.original:
            raise EvidenceError('Repeated original identity has inconsistent source, checks or detection')
        if pair.original_key in clusters and clusters[pair.original_key] != pair.cluster_id:
            raise EvidenceError('Repeated original identity has inconsistent cluster identity')
        originals[pair.original_key] = pair.original
        clusters[pair.original_key] = pair.cluster_id
        if pair.key in unique:
            if unique[pair.key] != pair:
                raise EvidenceError('Conflicting duplicate pair identity')
            removed += 1
        else:
            unique[pair.key] = pair
    return list(unique.values()), removed
