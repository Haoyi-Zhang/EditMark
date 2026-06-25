from __future__ import annotations

from .base import TransformBundle
from .implementations import CanonicalizeTextTransform, NormalizeWhitespaceTransform, StripCommentsTransform


def available_transforms() -> tuple[str, ...]:
    return ("normalize_whitespace", "strip_comments", "canonicalize_text")


def build_transform_bundle(name: str) -> TransformBundle:
    name = name.lower()
    if name == "normalize_whitespace":
        return TransformBundle(name=name, transform=NormalizeWhitespaceTransform())
    if name == "strip_comments":
        return TransformBundle(name=name, transform=StripCommentsTransform())
    if name == "canonicalize_text":
        return TransformBundle(name=name, transform=CanonicalizeTextTransform())
    raise KeyError(f"unknown transform: {name}")
