from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..models import AttackOutcome


class CodeAttack(ABC):
    name: str

    @abstractmethod
    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class AttackBundle:
    name: str
    attack: CodeAttack
    severity: float = 0.0
    description: str = ""

    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        return self.attack.apply(source, seed=seed, metadata=metadata, context=context)
