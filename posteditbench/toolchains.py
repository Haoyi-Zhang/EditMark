from __future__ import annotations

import re
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import Any

from .language_support import language_version, normalize_language_name, runner_image


TOOLCHAIN_REQUIREMENTS: dict[str, tuple[dict[str, Any], ...]] = {
    "python": (
        {
            "tool": sys.executable,
            "display_name": "python",
            "command": (sys.executable, "--version"),
            "pattern": r"Python\s+(?P<version>\d+(?:\.\d+)+)",
            "minimum_version": "3.10",
            "expected_prefix": "3.11",
        },
    ),
    "javascript": (
        {
            "tool": "node",
            "display_name": "node",
            "command": ("node", "--version"),
            "pattern": r"v(?P<version>\d+(?:\.\d+)+)",
            "minimum_version": "12.22",
            "expected_prefix": "20",
        },
    ),
    "java": (
        {
            "tool": "javac",
            "display_name": "javac",
            "command": ("javac", "-version"),
            "pattern": r"javac\s+(?P<version>\d+(?:\.\d+)+)",
            "minimum_version": "17",
            "expected_prefix": "21",
        },
        {
            "tool": "java",
            "display_name": "java",
            "command": ("java", "-version"),
            "pattern": r'version\s+"(?P<version>\d+(?:\.\d+)+)',
            "minimum_version": "17",
            "expected_prefix": "21",
        },
    ),
    "cpp": (
        {
            "tool": "g++",
            "display_name": "g++",
            "command": ("g++", "--version"),
            "pattern": r"(?P<version>\d+(?:\.\d+)+)",
            "minimum_version": "11",
            "expected_prefix": "13",
        },
    ),
    "go": (
        {
            "tool": "go",
            "display_name": "go",
            "command": ("go", "version"),
            "pattern": r"go version go(?P<version>\d+(?:\.\d+)+)",
            "minimum_version": "1.18",
            "expected_prefix": "1.22",
        },
    ),
}


def toolchain_requirements(language: str) -> tuple[dict[str, Any], ...]:
    normalized = normalize_language_name(language)
    return TOOLCHAIN_REQUIREMENTS.get(normalized, ())


def _extract_version(output: str, pattern: str) -> str:
    match = re.search(pattern, output, flags=re.IGNORECASE | re.MULTILINE)
    if match is None:
        return ""
    return str(match.group("version")).strip()


def _version_matches(actual: str, expected_prefix: str) -> bool:
    return bool(actual and expected_prefix and actual.startswith(expected_prefix))


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", str(value))
    return tuple(int(part) for part in parts)


def _version_at_least(actual: str, minimum_version: str) -> bool:
    actual_parts = _version_tuple(actual)
    minimum_parts = _version_tuple(minimum_version)
    if not actual_parts or not minimum_parts:
        return False
    width = max(len(actual_parts), len(minimum_parts))
    actual_parts = actual_parts + (0,) * (width - len(actual_parts))
    minimum_parts = minimum_parts + (0,) * (width - len(minimum_parts))
    return actual_parts >= minimum_parts


@lru_cache(maxsize=16)
def inspect_local_toolchain(language: str) -> dict[str, Any]:
    normalized = normalize_language_name(language)
    requirements = toolchain_requirements(normalized)
    if not requirements:
        return {
            "language": normalized,
            "status": "not_required",
            "verified": False,
            "runner_image": runner_image(normalized),
            "language_version": language_version(normalized),
            "tools": [],
            "issues": [],
        }

    issues: list[str] = []
    tool_rows: list[dict[str, Any]] = []
    verified = True
    for requirement in requirements:
        executable = str(requirement.get("tool", "")).strip()
        resolved = executable if executable == sys.executable else shutil.which(executable)
        if not resolved:
            issues.append(f"missing_tool:{requirement.get('display_name', executable)}")
            verified = False
            tool_rows.append(
                {
                    "tool": str(requirement.get("display_name", executable)),
                    "resolved_path": "",
                    "version": "",
                    "minimum_version": str(requirement.get("minimum_version", "")),
                    "expected_prefix": str(requirement.get("expected_prefix", "")),
                    "recommended_version": str(requirement.get("expected_prefix", "")),
                    "verified": False,
                    "recommended_match": False,
                }
            )
            continue
        command = list(requirement.get("command", (resolved, "--version")))
        command[0] = resolved
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=10.0,
            )
        except Exception as exc:
            issues.append(f"tool_invocation_failed:{requirement.get('display_name', executable)}:{exc}")
            verified = False
            tool_rows.append(
                {
                    "tool": str(requirement.get("display_name", executable)),
                    "resolved_path": resolved,
                    "version": "",
                    "minimum_version": str(requirement.get("minimum_version", "")),
                    "expected_prefix": str(requirement.get("expected_prefix", "")),
                    "recommended_version": str(requirement.get("expected_prefix", "")),
                    "verified": False,
                    "recommended_match": False,
                }
            )
            continue
        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
        version = _extract_version(output, str(requirement.get("pattern", "")))
        minimum_version = str(requirement.get("minimum_version", requirement.get("expected_prefix", ""))).strip()
        recommended_version = str(requirement.get("expected_prefix", minimum_version)).strip()
        meets_minimum = completed.returncode == 0 and _version_at_least(version, minimum_version)
        recommended_match = _version_matches(version, recommended_version)
        if not meets_minimum:
            verified = False
            if completed.returncode != 0:
                issues.append(f"tool_invocation_failed:{requirement.get('display_name', executable)}:exit_{completed.returncode}")
            elif not version:
                issues.append(f"tool_version_unparseable:{requirement.get('display_name', executable)}")
            else:
                issues.append(
                    f"tool_version_mismatch:{requirement.get('display_name', executable)}:{version}<{minimum_version}"
                )
        tool_rows.append(
            {
                "tool": str(requirement.get("display_name", executable)),
                "resolved_path": resolved,
                "version": version,
                "minimum_version": minimum_version,
                "expected_prefix": recommended_version,
                "recommended_version": recommended_version,
                "verified": meets_minimum,
                "recommended_match": recommended_match,
            }
        )

    return {
        "language": normalized,
        "status": "ok" if verified else "failed",
        "verified": verified,
        "runner_image": runner_image(normalized),
        "language_version": language_version(normalized),
        "tools": tool_rows,
        "issues": issues,
    }
