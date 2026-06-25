from __future__ import annotations

import hashlib
import io
import re
import tokenize as py_tokenize
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BLOCK_COMMENT_RE = re.compile(r"(?s)/\*.*?\*/")


def stable_hash(text: str, *, secret: str = "", digest_size: int = 12) -> str:
    hasher = hashlib.blake2b(digest_size=digest_size, key=secret.encode("utf-8"))
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def language_comment_prefix(language: str) -> str:
    language = language.lower()
    if language in {"c", "c++", "cpp", "java", "js", "javascript", "ts", "typescript", "go", "rust"}:
        return "//"
    return "#"


def _trim_comment_stripped_lines(lines: list[str]) -> str:
    trimmed = [line.rstrip() for line in lines if line.rstrip().strip()]
    return "\n".join(trimmed)


def _strip_python_comments_result(text: str) -> tuple[str, str | None]:
    tokens: list[tuple[int, str]] = []
    saw_comment = False
    try:
        for token in py_tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == py_tokenize.COMMENT:
                saw_comment = True
                continue
            tokens.append((token.type, token.string))
    except (IndentationError, SyntaxError, py_tokenize.TokenError):
        return text, "python_comment_strip_parse_failed"
    if not saw_comment:
        return text, None
    return _trim_comment_stripped_lines(py_tokenize.untokenize(tokens).splitlines()), None


def _strip_python_comments(text: str) -> str:
    stripped, _ = _strip_python_comments_result(text)
    return stripped


def _strip_brace_comments(text: str, *, line_markers: tuple[str, ...]) -> str:
    result: list[str] = []
    index = 0
    in_block_comment = False
    quote: str | None = None
    escaped = False
    saw_comment = False
    while index < len(text):
        current = text[index]
        next_two = text[index : index + 2]
        if in_block_comment:
            if next_two == "*/":
                in_block_comment = False
                saw_comment = True
                index += 2
                continue
            index += 1
            continue
        if quote is not None:
            result.append(current)
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif current == quote:
                quote = None
            index += 1
            continue
        if next_two == "/*":
            in_block_comment = True
            saw_comment = True
            index += 2
            continue
        if any(text.startswith(marker, index) for marker in line_markers):
            saw_comment = True
            while index < len(text) and text[index] != "\n":
                index += 1
            continue
        if current in {"'", '"'}:
            quote = current
        result.append(current)
        index += 1
    if not saw_comment:
        return text
    return _trim_comment_stripped_lines("".join(result).splitlines())


def strip_comments_with_reason(text: str, *, language: str = "") -> tuple[str, str | None]:
    normalized_language = str(language or "").strip().lower()
    if normalized_language in {"py", "python"}:
        return _strip_python_comments_result(text)
    if normalized_language in {"c", "c++", "cpp", "go", "java", "javascript", "js", "rust", "ts", "typescript"}:
        return _strip_brace_comments(text, line_markers=("//",)), None
    if "//" in text or "/*" in text:
        return _strip_brace_comments(text, line_markers=("//", "#")), None
    if "#" in text:
        return _strip_python_comments_result(text)
    stripped, reason = _strip_python_comments_result(text)
    if stripped != text or reason is not None:
        return stripped, reason
    return _strip_brace_comments(text, line_markers=("#", "//")), None


def strip_comments(text: str, *, language: str = "") -> str:
    stripped, _ = strip_comments_with_reason(text, language=language)
    return stripped


def normalize_whitespace(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(line for line in lines if line.strip())


def line_count(text: str) -> int:
    return len(text.splitlines())


def edit_distance_ratio(left: str, right: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(a=left, b=right).ratio()


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union)


def scrub_paths(text: str) -> str:
    text = re.sub(r"[A-Za-z]:\\[^\\\s'\"]+(?:\\[^\\\s'\"]+)+", "<path>", text)
    text = re.sub(r"/(?:[^/\s]+/)+[^/\s]+", "<path>", text)
    return text


@dataclass(frozen=True, slots=True)
class StableRandom:
    seed: int

    def choice(self, items: list[str]) -> str:
        if not items:
            raise ValueError("choice requires a non-empty list")
        index = self.seed % len(items)
        return items[index]

    def shuffle(self, items: list[Any]) -> list[Any]:
        result = list(items)
        n = len(result)
        for idx in range(n):
            swap = (self.seed + idx * 17) % n if n else 0
            result[idx], result[swap] = result[swap], result[idx]
        return result


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
