from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import BenchmarkExample, DetectionResult, WatermarkSpec, WatermarkedSnippet
from ..utils import language_comment_prefix, stable_hash
from .base import WatermarkBundle, WatermarkDetector, WatermarkEmbedder


_IDENTIFIER_SUFFIXES = {
    "python": "py",
    "py": "py",
    "cpp": "cpp",
    "c++": "cpp",
    "java": "java",
    "javascript": "js",
    "js": "js",
    "go": "go",
}

_IDENTIFIER_PATTERN = re.compile(r"\bwm_[a-f0-9]{12}_(?:py|cpp|java|js|go)\b")
_JS_DIRECTIVE_PATTERN = re.compile(r"""^(?:'[^'\\]*(?:\\.[^'\\]*)*'|"[^"\\]*(?:\\.[^"\\]*)*")\s*;?$""")


def _marker(spec: WatermarkSpec, example: BenchmarkExample) -> str:
    token = stable_hash(f"{example.example_id}:{spec.payload}", secret=spec.secret)
    return token[:12]


def _identifier_name(marker: str, language: str) -> str:
    normalized = language.lower()
    suffix = _IDENTIFIER_SUFFIXES.get(normalized, normalized[:3] or "wm")
    return f"wm_{marker}_{suffix}"


def _anchor_comment(marker: str, language: str) -> str:
    return f"{language_comment_prefix(language)} wm:{marker}"


def _inject_comment(source: str, comment: str, language: str) -> str:
    lines = source.splitlines()
    if not lines:
        return comment

    insert_at = 0
    normalized = language.lower()
    if lines[0].startswith("#!"):
        insert_at = 1

    if normalized in {"javascript", "js"}:
        while insert_at < len(lines) and _JS_DIRECTIVE_PATTERN.fullmatch(lines[insert_at].strip()):
            insert_at += 1

    return "\n".join([*lines[:insert_at], comment, *lines[insert_at:]])


def _already_contains_identifier(source: str, identifier: str) -> bool:
    return bool(re.search(rf"\b{re.escape(identifier)}\b", source))


def _append_identifier(source: str, declaration: str, identifier: str) -> tuple[str, str]:
    if _already_contains_identifier(source, identifier):
        return source, identifier
    if source.strip():
        return "\n".join([source.rstrip(), declaration]), identifier
    return declaration, identifier


def _inject_python_identifier(source: str, identifier: str) -> tuple[str, str]:
    return _append_identifier(source, f"{identifier} = None", identifier)


def _inject_cpp_identifier(source: str, identifier: str) -> tuple[str, str]:
    return _append_identifier(source, f'static const char* {identifier} = "wm";', identifier)


def _inject_js_identifier(source: str, identifier: str) -> tuple[str, str]:
    return _append_identifier(source, f'const {identifier} = "wm";', identifier)


def _inject_go_identifier(source: str, identifier: str) -> tuple[str, str | None]:
    if _already_contains_identifier(source, identifier):
        return source, identifier
    lines = source.splitlines()
    package_index = next((index for index, line in enumerate(lines) if line.strip().startswith("package ")), -1)
    if package_index < 0:
        return source, None
    insert_at = package_index + 1
    if insert_at < len(lines) and not lines[insert_at].strip():
        insert_at += 1
    if insert_at < len(lines) and lines[insert_at].strip() == "import (":
        insert_at += 1
        while insert_at < len(lines) and lines[insert_at].strip() != ")":
            insert_at += 1
        if insert_at < len(lines):
            insert_at += 1
    elif insert_at < len(lines) and lines[insert_at].strip().startswith("import "):
        insert_at += 1
    updated = "\n".join([*lines[:insert_at], f'const {identifier} = "wm"', *lines[insert_at:]])
    return updated, identifier


def _inject_java_identifier(source: str, identifier: str) -> tuple[str, str | None]:
    if _already_contains_identifier(source, identifier):
        return source, identifier
    match = re.search(r"\b(class|interface|enum|record)\b[^{;]*\{", source)
    if not match:
        return source, None
    if match.group(1) == "enum":
        return source, None
    declaration = (
        f'    String {identifier} = "wm";'
        if match.group(1) == "interface"
        else f'    private static final String {identifier} = "wm";'
    )
    updated = f"{source[:match.end()]}\n{declaration}{source[match.end():]}"
    return updated, identifier


def _inject_identifier(source: str, language: str, marker: str) -> tuple[str, str | None]:
    normalized = language.lower()
    identifier = _identifier_name(marker, normalized)
    if normalized in {"python", "py"}:
        return _inject_python_identifier(source, identifier)
    if normalized in {"cpp", "c++"}:
        return _inject_cpp_identifier(source, identifier)
    if normalized in {"javascript", "js"}:
        return _inject_js_identifier(source, identifier)
    if normalized == "go":
        return _inject_go_identifier(source, identifier)
    if normalized == "java":
        return _inject_java_identifier(source, identifier)
    return source, None


def _detect_identifier(text: str) -> tuple[float, tuple[str, ...]]:
    evidence = tuple(match.group(0) for match in re.finditer(_IDENTIFIER_PATTERN, text))
    if not evidence:
        return 0.0, ()
    return 0.4, evidence


def _channels_from_text(text: str, *, marker: str) -> list[str]:
    channels: list[str] = []
    if marker and f"wm:{marker}" in text:
        channels.append("comment")
    if _IDENTIFIER_PATTERN.search(text):
        channels.append("identifier")
    return channels


@dataclass(slots=True)
class KGWEmbedder(WatermarkEmbedder):
    name: str = "kgw"

    def embed(self, example: BenchmarkExample, spec: WatermarkSpec) -> WatermarkedSnippet:
        marker = _marker(spec, example)
        source = _inject_comment(example.reference_solution, _anchor_comment(marker, example.language), example.language)
        source, identifier = _inject_identifier(source, example.language, marker)
        channels = ["comment"]
        if identifier:
            channels.append("identifier")
        return WatermarkedSnippet(
            example_id=example.example_id,
            language=example.language,
            source=source,
            watermark=spec,
            metadata={
                "marker": marker,
                "identifier": identifier,
                "channels": channels,
            },
        )


@dataclass(slots=True)
class KGWDetector(WatermarkDetector):
    name: str = "kgw"

    def detect(self, source: str | WatermarkedSnippet, spec: WatermarkSpec, *, example_id: str = "") -> DetectionResult:
        text = source.source if isinstance(source, WatermarkedSnippet) else source
        marker = stable_hash(f"{example_id}:{spec.payload}", secret=spec.secret)[:12] if example_id else ""
        score = 0.0
        evidence: list[str] = []
        if marker:
            anchor = f"wm:{marker}"
            if anchor in text:
                evidence.append(anchor)
                score += 0.6
        if spec.payload in text:
            evidence.append(f"payload:{spec.payload}")
            score += 0.2
        identifier_score, identifier_evidence = _detect_identifier(text)
        if identifier_evidence:
            evidence.extend(identifier_evidence)
            score += identifier_score
        score = min(score, 1.0)
        threshold = float(spec.parameters.get("threshold", 0.5))
        return DetectionResult(
            example_id=example_id or "unknown",
            method=self.name,
            score=score,
            detected=score >= threshold,
            threshold=threshold,
            evidence=tuple(evidence),
            metadata={"channels": _channels_from_text(text, marker=marker)},
        )


def build_kgw_bundle() -> WatermarkBundle:
    return WatermarkBundle(name="kgw", embedder=KGWEmbedder(), detector=KGWDetector())
