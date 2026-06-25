from .common import (
    CheckoutInfo,
    runtime_watermark_names,
    stone_family_baseline_names,
    stone_family_checkout_available,
    stone_family_checkout_metadata,
    stone_family_checkout_status,
    validate_checkout,
)
from .runtime import build_runtime_bundle


def baseline_family_names() -> tuple[str, ...]:
    return stone_family_baseline_names()


def is_stone_family_baseline(name: str) -> bool:
    return str(name).lower() in stone_family_baseline_names()


def build_stone_family_bundle(name: str):
    normalized = str(name).lower()
    return build_runtime_bundle(normalized)


__all__ = [
    "CheckoutInfo",
    "baseline_family_names",
    "build_stone_family_bundle",
    "is_stone_family_baseline",
    "runtime_watermark_names",
    "stone_family_checkout_available",
    "stone_family_checkout_metadata",
    "stone_family_checkout_status",
    "validate_checkout",
]
