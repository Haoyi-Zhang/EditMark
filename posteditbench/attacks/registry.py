from __future__ import annotations

from .base import AttackBundle
from .implementations import (
    BlockShuffleAttack,
    BudgetedAdaptiveAttack,
    CommentStripAttack,
    ControlFlowFlattenAttack,
    IdentifierRenameAttack,
    NoiseInsertAttack,
    WhitespaceNormalizeAttack,
)

CORE_ATTACKS: tuple[str, ...] = (
    "whitespace_normalize",
    "comment_strip",
    "identifier_rename",
    "noise_insert",
)
STRESS_ATTACKS: tuple[str, ...] = (
    "control_flow_flatten",
    "block_shuffle",
    "budgeted_adaptive",
)


_ATTACK_INFO: dict[str, dict[str, object]] = {
    "comment_strip": {
        "severity": 0.12,
        "tier": "core",
        "description": "Remove reviewer-visible comments while preserving code tokens.",
    },
    "identifier_rename": {
        "severity": 0.2,
        "tier": "core",
        "description": "Rename safe local identifiers while preserving structure.",
    },
    "whitespace_normalize": {
        "severity": 0.08,
        "tier": "core",
        "description": "Whitespace-only normalization.",
    },
    "noise_insert": {
        "severity": 0.18,
        "tier": "core",
        "description": "Add benign language-correct comment noise.",
    },
    "control_flow_flatten": {
        "severity": 0.42,
        "tier": "stress",
        "description": "Flatten redundant control-flow scaffolds as a structural stress attack.",
    },
    "block_shuffle": {
        "severity": 0.35,
        "tier": "stress",
        "description": "Shuffle code blocks as a stronger structural stress attack.",
    },
    "budgeted_adaptive": {
        "severity": 0.5,
        "tier": "stress",
        "description": "Compose stronger perturbations under a bounded adaptive stress budget.",
    },
}


def available_attacks() -> tuple[str, ...]:
    return tuple(_ATTACK_INFO.keys())


def attack_tier(name: str) -> str:
    info = _ATTACK_INFO.get(str(name).lower(), {})
    return str(info.get("tier", "stress"))


def build_attack_bundle(name: str) -> AttackBundle:
    name = name.lower()
    info = _ATTACK_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown attack: {name}")
    if name == "comment_strip":
        attack = CommentStripAttack()
    elif name == "identifier_rename":
        attack = IdentifierRenameAttack()
    elif name == "whitespace_normalize":
        attack = WhitespaceNormalizeAttack()
    elif name == "noise_insert":
        attack = NoiseInsertAttack()
    elif name == "control_flow_flatten":
        attack = ControlFlowFlattenAttack()
    elif name == "block_shuffle":
        attack = BlockShuffleAttack()
    elif name == "budgeted_adaptive":
        attack = BudgetedAdaptiveAttack()
    else:  # pragma: no cover - guarded by info lookup
        raise KeyError(f"unknown attack: {name}")
    return AttackBundle(
        name=name,
        attack=attack,
        severity=float(info.get("severity", 0.0)),
        description=str(info.get("description", "")),
    )
