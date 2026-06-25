from __future__ import annotations

from pathlib import Path
from typing import Any

from ..baselines.stone_family.common import (
    resolve_checkout,
    runtime_watermark_names as stone_family_runtime_watermark_names,
    stone_family_checkout_available,
    stone_family_checkout_metadata,
)
from .base import WatermarkBundle


_CANONICAL_RUNTIME_NAMES = tuple(stone_family_runtime_watermark_names())


def runtime_watermark_names() -> tuple[str, ...]:
    return _CANONICAL_RUNTIME_NAMES


def is_runtime_watermark(name: str) -> bool:
    normalized = str(name or "").strip().lower()
    return normalized in _CANONICAL_RUNTIME_NAMES


def resolve_upstream_root(name: str = "stone_runtime") -> Path | None:
    normalized = str(name or "").strip().lower()
    checkout = resolve_checkout(normalized)
    return checkout.source_root if checkout is not None else None


def upstream_checkout_available(name: str = "stone_runtime") -> bool:
    normalized = str(name or "").strip().lower()
    return stone_family_checkout_available(normalized)


def upstream_checkout_metadata(name: str = "stone_runtime") -> dict[str, Any]:
    normalized = str(name or "").strip().lower()
    return stone_family_checkout_metadata(normalized)


def build_upstream_runtime_bundle(name: str) -> WatermarkBundle:
    normalized = str(name or "").strip().lower()
    if normalized in _CANONICAL_RUNTIME_NAMES:
        from ..baselines.stone_family.runtime import build_runtime_bundle as build_stone_family_bundle

        return build_stone_family_bundle(normalized)
    raise KeyError(f"unknown runtime watermark scheme: {name}")


__all__ = [
    "build_upstream_runtime_bundle",
    "is_runtime_watermark",
    "resolve_upstream_root",
    "upstream_checkout_available",
    "upstream_checkout_metadata",
    "runtime_watermark_names",
]
