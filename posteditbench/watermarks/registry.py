from __future__ import annotations

from .base import WatermarkBundle
from .kgw import build_kgw_bundle
from .method_stubs import build_comment_bundle, build_identifier_bundle
from .structural import build_structural_flow_bundle
from .upstream_runtime import build_upstream_runtime_bundle

_PUBLIC_WATERMARKS: tuple[str, ...] = (
    "stone_runtime",
    "sweet_runtime",
    "ewd_runtime",
    "kgw_runtime",
)

_INTERNAL_WATERMARKS: tuple[str, ...] = (
    "kgw",
    "comment",
    "identifier",
    "structural_flow",
)

def _allow_internal_watermarks(allow_internal: bool | None = None) -> bool:
    return bool(allow_internal)


def available_watermarks() -> tuple[str, ...]:
    return _PUBLIC_WATERMARKS


def all_watermarks() -> tuple[str, ...]:
    return tuple(dict.fromkeys((*_INTERNAL_WATERMARKS, *_PUBLIC_WATERMARKS)))


def internal_watermarks() -> tuple[str, ...]:
    return _INTERNAL_WATERMARKS


def watermark_origin(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized.endswith("_runtime"):
        return "upstream"
    return "native"


def build_watermark_bundle(name: str, *, allow_internal: bool | None = None) -> WatermarkBundle:
    name = name.lower()
    if name in _INTERNAL_WATERMARKS and not _allow_internal_watermarks(allow_internal):
        raise KeyError(
            f"internal watermark scheme '{name}' is not part of the public canonical roster; "
            "pass allow_internal=True only for internal test/development flows"
        )
    if name == "kgw":
        return build_kgw_bundle()
    if name == "comment":
        return build_comment_bundle()
    if name == "identifier":
        return build_identifier_bundle()
    if name == "structural_flow":
        return build_structural_flow_bundle()
    if name in {"stone_runtime", "sweet_runtime", "ewd_runtime", "kgw_runtime"}:
        return build_upstream_runtime_bundle(name)
    raise KeyError(f"unknown watermark scheme: {name}")
