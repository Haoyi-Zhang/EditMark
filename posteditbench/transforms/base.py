from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class TextTransform(ABC):
    name: str

    @abstractmethod
    def apply(self, source: str) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class TransformBundle:
    name: str
    transform: TextTransform

    def apply(self, source: str) -> str:
        return self.transform.apply(source)
