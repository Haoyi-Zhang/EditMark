from __future__ import annotations

import os


def token_env_candidates(token_env: str | None) -> tuple[str, ...]:
    primary = str(token_env or "").strip()
    if not primary:
        return ()
    candidates: list[str] = [primary]
    fallback = f"{primary}_FALLBACK"
    if fallback not in candidates:
        candidates.append(fallback)
    if primary == "HF_ACCESS_TOKEN" and "HF_ACCESS_TOKEN_FALLBACK" not in candidates:
        candidates.append("HF_ACCESS_TOKEN_FALLBACK")
    return tuple(candidates)


def resolve_token_env_value(token_env: str | None) -> str:
    for name in token_env_candidates(token_env):
        value = str(os.environ.get(name, "")).strip()
        if value:
            return value
    return ""
