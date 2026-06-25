from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import BenchmarkExample, DetectionResult, WatermarkSpec, WatermarkedSnippet
from ..utils import language_comment_prefix, stable_hash
from .base import WatermarkBundle, WatermarkDetector, WatermarkEmbedder


@dataclass(slots=True)
class CommentAnchorEmbedder(WatermarkEmbedder):
    name: str = "comment"

    def embed(self, example: BenchmarkExample, spec: WatermarkSpec) -> WatermarkedSnippet:
        marker = stable_hash(example.example_id, secret=spec.secret)[:10]
        prefix = language_comment_prefix(example.language)
        source = "\n".join([f"{prefix} anchor:{marker}", example.reference_solution])
        return WatermarkedSnippet(
            example_id=example.example_id,
            language=example.language,
            source=source,
            watermark=spec,
            metadata={"marker": marker, "channel": "comment"},
        )


@dataclass(slots=True)
class CommentAnchorDetector(WatermarkDetector):
    name: str = "comment"

    def detect(self, source: str | WatermarkedSnippet, spec: WatermarkSpec, *, example_id: str = "") -> DetectionResult:
        text = source.source if isinstance(source, WatermarkedSnippet) else source
        marker = stable_hash(example_id or text, secret=spec.secret)[:10] if example_id else ""
        evidence = []
        score = 0.0
        if marker and f"anchor:{marker}" in text:
            evidence.append(f"anchor:{marker}")
            score = 1.0
        threshold = float(spec.parameters.get("threshold", 0.5))
        return DetectionResult(
            example_id=example_id or "unknown",
            method=self.name,
            score=score,
            detected=score >= threshold,
            threshold=threshold,
            evidence=tuple(evidence),
            metadata={"channel": "comment"},
        )


@dataclass(slots=True)
class IdentifierSaltEmbedder(WatermarkEmbedder):
    name: str = "identifier"

    def embed(self, example: BenchmarkExample, spec: WatermarkSpec) -> WatermarkedSnippet:
        marker = stable_hash(f"{example.example_id}:{spec.payload}", secret=spec.secret)[:8]
        target = next((word for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", example.reference_solution) if len(word) > 3), "value")
        renamed = f"salt_{marker}_{target}"
        source = re.sub(rf"\b{re.escape(target)}\b", renamed, example.reference_solution, count=1)
        return WatermarkedSnippet(
            example_id=example.example_id,
            language=example.language,
            source=source,
            watermark=spec,
            metadata={"marker": marker, "channel": "identifier", "target": target},
        )


@dataclass(slots=True)
class IdentifierSaltDetector(WatermarkDetector):
    name: str = "identifier"

    def detect(self, source: str | WatermarkedSnippet, spec: WatermarkSpec, *, example_id: str = "") -> DetectionResult:
        text = source.source if isinstance(source, WatermarkedSnippet) else source
        evidence = re.findall(r"salt_[a-f0-9]{8}_[A-Za-z_][A-Za-z0-9_]*", text)
        score = 0.8 if evidence else 0.0
        threshold = float(spec.parameters.get("threshold", 0.5))
        return DetectionResult(
            example_id=example_id or "unknown",
            method=self.name,
            score=score,
            detected=score >= threshold,
            threshold=threshold,
            evidence=tuple(evidence),
            metadata={"channel": "identifier"},
        )


def build_comment_bundle() -> WatermarkBundle:
    return WatermarkBundle(name="comment", embedder=CommentAnchorEmbedder(), detector=CommentAnchorDetector())


def build_identifier_bundle() -> WatermarkBundle:
    return WatermarkBundle(name="identifier", embedder=IdentifierSaltEmbedder(), detector=IdentifierSaltDetector())
