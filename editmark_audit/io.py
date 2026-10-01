"""Strict JSON and non-overwriting, atomic report publication."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

class EvidenceError(ValueError):
    """Malformed, contradictory, or insufficiently typed evidence."""


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise EvidenceError(f'Duplicate JSON key: {k!r}')
        out[k] = v
    return out


def _constant(value: str) -> None:
    raise EvidenceError(f'Non-finite JSON constant: {value}')


def _float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceError(f'JSON number exceeds finite float range: {value}')
    # Inspect only the significand: Decimal(value) can itself fail on a
    # syntactically valid literal with an extremely large exponent.
    nonzero_significand = any(c in '123456789' for c in value.lower().split('e', 1)[0])
    if number == 0.0 and nonzero_significand:
        raise EvidenceError(f'Nonzero JSON number underflows binary64: {value}')
    return number


def loads(text: str) -> Any:
    try:
        return json.loads(text.lstrip('\ufeff'), object_pairs_hook=_unique, parse_constant=_constant, parse_float=_float)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise EvidenceError(f'Invalid JSON: {exc}') from exc


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read explicit normalized observations; never discover or execute inputs."""
    path = Path(path)
    if not path.is_file() or path.suffix.lower() not in {'.json', '.jsonl'}:
        raise EvidenceError(f'Expected an existing JSON or JSONL file: {path}')
    if path.suffix.lower() == '.jsonl':
        rows = []
        with path.open(encoding='utf-8-sig') as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    try: rows.append(loads(line))
                    except EvidenceError as exc: raise EvidenceError(f'{path.name}:{i}: {exc}') from exc
    else:
        doc = loads(path.read_text(encoding='utf-8-sig'))
        rows = doc if isinstance(doc, list) else [doc]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise EvidenceError('Expected a nonempty list of record objects; not a summary table')
    return rows


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new_json(path: Path, value: Any) -> None:
    """Create only: fsync a temporary file, then atomically link without replacement.

    A pre-existing output (including a symlink) is never overwritten. On a file
    system without hard-link support the operation fails rather than weakening
    that guarantee. Temporary files are removed even after failure.
    """
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix='.editmark-', suffix='.tmp', delete=False) as f:
            name = f.name
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.link(name, path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)
