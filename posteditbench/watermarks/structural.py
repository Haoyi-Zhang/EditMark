from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import BenchmarkExample, DetectionResult, WatermarkSpec, WatermarkedSnippet
from ..utils import stable_hash
from .base import WatermarkBundle, WatermarkDetector, WatermarkEmbedder


_PY_SIGNATURE = re.compile(r"^(\s*)(?:async\s+def|def|class)\b.*:\s*$")
_BRACE_OPEN = re.compile(r"^(\s*).*\{\s*$")
_BRACE_IF = re.compile(r"^\s*if\s*(?:\(\s*)?true(?:\s*\))?\s*\{\s*$", re.IGNORECASE)
_BRACE_CLOSE = re.compile(r"^\s*\}\s*$")


def _marker(spec: WatermarkSpec, example: BenchmarkExample) -> str:
    return stable_hash(f"{example.example_id}:{spec.payload}", secret=spec.secret)[:12]


def _depth(marker: str) -> int:
    return 2 + int(marker[:2], 16) % 3


def _python_scaffold(indent: str, depth: int) -> list[str]:
    lines: list[str] = []
    for level in range(depth):
        lines.append(f"{indent}{'    ' * level}if True:")
    lines.append(f"{indent}{'    ' * depth}pass")
    return lines


def _brace_scaffold(indent: str, depth: int, *, style: str) -> list[str]:
    open_stmt = "if (true) {" if style != "rust" else "if true {"
    lines: list[str] = []
    for level in range(depth):
        lines.append(f"{indent}{'  ' * level}{open_stmt}")
    for level in reversed(range(depth)):
        lines.append(f"{indent}{'  ' * level}}}")
    return lines


def _insert_python_scaffold(source: str, depth: int) -> str:
    lines = source.splitlines()
    for index, line in enumerate(lines):
        match = _PY_SIGNATURE.match(line)
        if not match:
            continue
        indent = match.group(1)
        scaffold = _python_scaffold(f"{indent}    ", depth)
        return "\n".join([*lines[: index + 1], *scaffold, *lines[index + 1 :]])
    return source


def _insert_brace_scaffold(source: str, depth: int, *, style: str) -> str:
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if "{" not in line:
            continue
        match = _BRACE_OPEN.match(line)
        if not match:
            continue
        indent = match.group(1)
        scaffold = _brace_scaffold(f"{indent}  ", depth, style=style)
        return "\n".join([*lines[: index + 1], *scaffold, *lines[index + 1 :]])
    return source


def _detect_python_scaffold(text: str, depth: int) -> tuple[bool, tuple[str, ...]]:
    pattern = "\n".join(_python_scaffold("    ", depth))
    if pattern in text:
        return True, (f"structural_depth:{depth}", "channel:python")
    return False, ()


def _detect_brace_scaffold(text: str, depth: int, *, style: str) -> tuple[bool, tuple[str, ...]]:
    pattern = "\n".join(_brace_scaffold("  ", depth, style=style))
    if pattern in text:
        return True, (f"structural_depth:{depth}", f"channel:{style}")
    compact = re.sub(r"\s+", "", text)
    if re.sub(r"\s+", "", pattern) in compact:
        return True, (f"structural_depth:{depth}", f"channel:{style}")
    return False, ()


@dataclass(slots=True)
class StructuralFlowEmbedder(WatermarkEmbedder):
    name: str = "structural_flow"

    def embed(self, example: BenchmarkExample, spec: WatermarkSpec) -> WatermarkedSnippet:
        marker = _marker(spec, example)
        depth = _depth(marker)
        language = example.language.lower()
        if language == "python":
            source = _insert_python_scaffold(example.reference_solution, depth)
            channel = "python"
            shape = "nested-if"
        elif language in {"javascript", "java"}:
            source = _insert_brace_scaffold(example.reference_solution, depth, style=language)
            channel = language
            shape = "nested-if"
        elif language == "rust":
            source = _insert_brace_scaffold(example.reference_solution, depth, style="rust")
            channel = "rust"
            shape = "nested-if"
        else:
            source = example.reference_solution
            channel = language or "unknown"
            shape = "unsupported"
        return WatermarkedSnippet(
            example_id=example.example_id,
            language=example.language,
            source=source,
            watermark=spec,
            metadata={
                "marker": marker,
                "depth": depth,
                "channel": channel,
                "shape": shape,
            },
        )


@dataclass(slots=True)
class StructuralFlowDetector(WatermarkDetector):
    name: str = "structural_flow"

    def detect(self, source: str | WatermarkedSnippet, spec: WatermarkSpec, *, example_id: str = "") -> DetectionResult:
        text = source.source if isinstance(source, WatermarkedSnippet) else source
        marker = stable_hash(f"{example_id}:{spec.payload}", secret=spec.secret)[:12] if example_id else ""
        depth = _depth(marker) if marker else 0
        language = source.language if isinstance(source, WatermarkedSnippet) else ""
        language = language.lower()

        found = False
        evidence: tuple[str, ...] = ()
        if language == "python":
            found, evidence = _detect_python_scaffold(text, depth)
        elif language in {"javascript", "java"}:
            found, evidence = _detect_brace_scaffold(text, depth, style=language)
        elif language == "rust":
            found, evidence = _detect_brace_scaffold(text, depth, style="rust")
        else:
            for candidate_language in ("python", "javascript", "java", "rust"):
                if candidate_language == "python":
                    found, evidence = _detect_python_scaffold(text, depth)
                else:
                    found, evidence = _detect_brace_scaffold(text, depth, style="rust" if candidate_language == "rust" else candidate_language)
                if found:
                    break

        score = 0.95 if found else 0.0
        threshold = float(spec.parameters.get("threshold", 0.5))
        return DetectionResult(
            example_id=example_id or "unknown",
            method=self.name,
            score=score,
            detected=score >= threshold,
            threshold=threshold,
            evidence=evidence,
            metadata={
                "channel": "structural",
                "depth": depth,
                "language": language or "unknown",
            },
        )


def build_structural_flow_bundle() -> WatermarkBundle:
    return WatermarkBundle(name="structural_flow", embedder=StructuralFlowEmbedder(), detector=StructuralFlowDetector())

