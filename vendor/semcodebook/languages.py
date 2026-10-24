from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    name: str
    extensions: tuple[str, ...]
    line_comment_prefix: str
    block_comment: tuple[str, str] | None = None


LANGUAGES: dict[str, LanguageSpec] = {
    "python": LanguageSpec("python", (".py",), "#", ('"""', '"""')),
    "javascript": LanguageSpec("javascript", (".js",), "//", ("/*", "*/")),
    "java": LanguageSpec("java", (".java",), "//", ("/*", "*/")),
    "cpp": LanguageSpec("cpp", (".cpp", ".cc", ".hpp", ".h"), "//", ("/*", "*/")),
    "go": LanguageSpec("go", (".go",), "//", ("/*", "*/")),
}


def normalize_language(value: str) -> str:
    key = value.strip().lower()
    aliases = {"c++": "cpp", "js": "javascript", "py": "python"}
    return aliases.get(key, key)


def supported_languages() -> tuple[str, ...]:
    return tuple(LANGUAGES.keys())
