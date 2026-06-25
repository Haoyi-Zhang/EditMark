from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from ..models import BenchmarkExample, DetectionResult, WatermarkSpec, WatermarkedSnippet


class WatermarkEmbedder(ABC):
    name: str

    @abstractmethod
    def embed(self, example: BenchmarkExample, spec: WatermarkSpec) -> WatermarkedSnippet:
        raise NotImplementedError


class WatermarkDetector(ABC):
    name: str

    @abstractmethod
    def detect(self, source: str | WatermarkedSnippet, spec: WatermarkSpec, *, example_id: str = "") -> DetectionResult:
        raise NotImplementedError


class WatermarkPreparer(Protocol):
    name: str

    def prepare(self, example: BenchmarkExample, spec: WatermarkSpec) -> BenchmarkExample:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class WatermarkBundle:
    name: str
    embedder: WatermarkEmbedder
    detector: WatermarkDetector
    preparer: WatermarkPreparer | None = None

    @property
    def uses_internal_generation(self) -> bool:
        return self.preparer is not None

    def prepare_example(self, example: BenchmarkExample, spec: WatermarkSpec) -> BenchmarkExample:
        if self.preparer is None:
            return example
        return self.preparer.prepare(example, spec)

    def embed(self, example: BenchmarkExample, spec: WatermarkSpec) -> WatermarkedSnippet:
        return self.embedder.embed(example, spec)

    def detect(self, source: str | WatermarkedSnippet, spec: WatermarkSpec, *, example_id: str = "") -> DetectionResult:
        return self.detector.detect(source, spec, example_id=example_id)
