from __future__ import annotations

from pathlib import Path


LANGUAGE_ALIASES: dict[str, str] = {
    "c++": "cpp",
    "cplusplus": "cpp",
    "go/golang": "go",
    "golang": "go",
    "javascript": "javascript",
    "js": "javascript",
    "python3": "python",
}


LANGUAGE_FAMILIES: dict[str, str] = {
    "python": "python",
    "javascript": "ecmascript",
    "typescript": "ecmascript",
    "java": "jvm",
    "cpp": "systems",
    "c": "systems",
    "go": "systems",
    "rust": "systems",
}


RUNNER_IMAGES: dict[str, str] = {
    "python": "python:3.11",
    "cpp": "gcc:13",
    "java": "openjdk:21",
    "javascript": "node:20",
    "go": "golang:1.22",
    "rust": "rust:1.78",
}


LANGUAGE_VERSIONS: dict[str, str] = {
    "python": "3.11",
    "cpp": "c++17",
    "java": "21",
    "javascript": "20",
    "go": "1.22",
    "rust": "1.78",
}


VALIDATION_MODES: dict[str, str] = {
    "python": "python_exec",
    "cpp": "docker_remote",
    "java": "docker_remote",
    "javascript": "docker_remote",
    "go": "docker_remote",
    "rust": "docker_remote",
}


def normalize_language_name(language: str) -> str:
    normalized = str(language or "").strip().lower()
    if not normalized:
        return "unknown"
    return LANGUAGE_ALIASES.get(normalized, normalized)


def language_family(language: str) -> str:
    normalized = normalize_language_name(language)
    return LANGUAGE_FAMILIES.get(normalized, normalized or "unknown")


def validation_mode(language: str) -> str:
    normalized = normalize_language_name(language)
    return VALIDATION_MODES.get(normalized, "unavailable")


def default_evaluation_backend(language: str) -> str:
    mode = validation_mode(language)
    if mode == "python_exec":
        return "python_exec"
    if mode == "docker_remote":
        return "docker_remote"
    return "unavailable"


def runner_image(language: str) -> str:
    return RUNNER_IMAGES.get(normalize_language_name(language), "")


def language_version(language: str) -> str:
    return LANGUAGE_VERSIONS.get(normalize_language_name(language), "")


def supports_execution(language: str, tests: tuple[str, ...] | list[str], *, backend: str | None = None) -> bool:
    if not tests:
        return False
    selected = str(backend or default_evaluation_backend(language)).strip().lower()
    return selected in {"python_exec", "docker_remote", "local_cpp", "mock_multilingual"}


def default_problem_filename(language: str) -> str:
    normalized = normalize_language_name(language)
    suffix = {
        "python": "solution.py",
        "cpp": "solution.cpp",
        "java": "Solution.java",
        "javascript": "solution.js",
        "go": "main.go",
        "rust": "main.rs",
    }.get(normalized, "solution.txt")
    return suffix


def _looks_like_windows_absolute(path_text: str) -> bool:
    normalized = path_text.replace("\\", "/")
    return len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/"


def _normalized_path_parts(path_text: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    normalized = path_text.replace("\\", "/")
    anchor = ""
    if _looks_like_windows_absolute(normalized):
        anchor = normalized[:2].lower()
        normalized = normalized[2:]
    elif normalized.startswith("/"):
        anchor = "/"
        normalized = normalized[1:]
    parts = tuple(part for part in normalized.split("/") if part and part != ".")
    return anchor, parts, tuple(part.lower() for part in parts)


def _path_leaf(path_text: str) -> str:
    normalized = path_text.replace("\\", "/").rstrip("/")
    if not normalized:
        return ""
    return normalized.rsplit("/", 1)[-1]


def source_relative_to(root: Path, path: Path) -> str:
    root_text = str(root)
    path_text = str(path)
    if path.is_absolute() or _looks_like_windows_absolute(path_text):
        candidate_text = path_text
    else:
        root_prefix = root_text.rstrip("/\\")
        path_suffix = path_text.lstrip("/\\")
        candidate_text = f"{root_prefix}/{path_suffix}"

    root_anchor, root_parts, root_compare = _normalized_path_parts(root_text)
    candidate_anchor, candidate_parts, candidate_compare = _normalized_path_parts(candidate_text)
    if (
        root_anchor != candidate_anchor
        or len(candidate_compare) < len(root_compare)
        or candidate_compare[: len(root_compare)] != root_compare
    ):
        return _path_leaf(candidate_text)

    relative_parts = candidate_parts[len(root_parts) :]
    if not relative_parts:
        return _path_leaf(candidate_text)
    if any(part.startswith(".") for part in relative_parts):
        return relative_parts[-1]
    if len(relative_parts) >= 3 and relative_parts[0] == "data" and relative_parts[1] == "public" and relative_parts[2] == "_cache":
        return relative_parts[-1]
    return "/".join(relative_parts)
