from __future__ import annotations

from dataclasses import dataclass

from ..utils import normalize_whitespace, strip_comments
from .base import TextTransform


@dataclass(slots=True)
class NormalizeWhitespaceTransform(TextTransform):
    name: str = "normalize_whitespace"

    def apply(self, source: str) -> str:
        return normalize_whitespace(source)


@dataclass(slots=True)
class StripCommentsTransform(TextTransform):
    name: str = "strip_comments"

    def apply(self, source: str) -> str:
        return normalize_whitespace(strip_comments(source))


@dataclass(slots=True)
class CanonicalizeTextTransform(TextTransform):
    name: str = "canonicalize_text"

    def apply(self, source: str) -> str:
        return normalize_whitespace(strip_comments(source)).lower()
