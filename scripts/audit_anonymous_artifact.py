from __future__ import annotations

import argparse
import re
from pathlib import Path

DANGEROUS_PATTERNS = [
    ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("github_token", r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    ("api_token", r"\b(?:sk|ak)-[A-Za-z0-9_-]{20,}\b"),
    ("private_key", r"BEGIN (?:OPENSSH|RSA|DSA|EC) PRIVATE KEY"),
    ("ssh_url", r"\bssh\s+[^@\s]+@[^ \t\r\n]+"),
    ("ipv4_address", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("windows_user_path", r"\b[A-Za-z]:\\Users\\[^\\\s]+\\"),
    ("unix_home_path", r"/home/[^/\s]+/"),
    ("local_data_path", r"/data/[^ \t\r\n]+"),
    (
        "project_identity",
        r"Code" + r"\s*Mark|code" + r"mark|Ha" + r"oyi|Hua" + r"ijin|Tian" + r"feng|"
        r"Ma" + r"ruf|Xun" + r"zhu|Tega" + r"wend",
    ),
]
BANNED_PATH_PARTS = {
    "paper",
    "paper_icse_2027",
    ".reference_guard_cache",
    "model_cache",
    "runs",
    "scratch",
}
BANNED_EXTS = {".pdf", ".tex", ".aux", ".bbl", ".blg", ".synctex.gz"}
SKIP_BINARY = {".png", ".pdf", ".pyc"}
ALLOWED_EMAILS = {"anonymous@example.invalid"}
ALLOWED_EMAIL_DOMAINS = {"example.com", "example.edu", "example.invalid"}


def _skip_match(label: str, value: str, rel: str) -> bool:
    rel_posix = rel.replace("\\", "/")
    if label == "email":
        if value.lower() in ALLOWED_EMAILS:
            return True
        domain = value.rsplit("@", 1)[-1].lower()
        return domain in ALLOWED_EMAIL_DOMAINS
    if label == "ipv4_address":
        return rel_posix.startswith("data/release/") or rel_posix == "results/environment/release_pip_freeze.txt"
    if label == "local_data_path":
        if rel_posix == "scripts/audit_anonymous_artifact.py":
            return True
        return value.startswith("/data/release/") or value.startswith("/data/HumanEval") or value.startswith("/data/{")
    if label == "unix_home_path":
        return rel_posix == "scripts/audit_anonymous_artifact.py"
    if label == "project_identity":
        return rel_posix == "scripts/audit_anonymous_artifact.py"
    return False


def candidate_files(root: Path) -> list[Path]:
    """Scan the release tree itself, including untracked files, while ignoring VCS internals."""
    return [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit anonymous ICSE artifact boundary.")
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    errors: list[str] = []
    compiled = [(label, re.compile(pat, re.IGNORECASE)) for label, pat in DANGEROUS_PATTERNS]
    for path in candidate_files(root):
        if not path.exists() or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        rel_parts = set(Path(rel).parts)
        if rel_parts & BANNED_PATH_PARTS:
            errors.append(f"banned path tracked: {rel}")
        if any(rel.lower().endswith(ext) for ext in BANNED_EXTS):
            errors.append(f"banned TeX/build artifact tracked: {rel}")
        if path.suffix.lower() in SKIP_BINARY or path.stat().st_size > 2_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in compiled:
            for match in pattern.finditer(text):
                value = match.group(0)
                if _skip_match(label, value, rel):
                    continue
                errors.append(f"anonymous-boundary pattern {label!r} in {rel}")
    if errors:
        print("anonymous_artifact_audit=failed")
        for error in errors:
            print(f"- {error}")
        return 1
    print("anonymous_artifact_audit=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
