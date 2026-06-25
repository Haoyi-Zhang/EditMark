from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT.parent / f"{ROOT.name}_review_package.zip"

BANNED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "paper",
    "paper_icse_2027",
    "model_cache",
    "runs",
    "scratch",
}

BANNED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".log",
    ".out",
    ".pdf",
    ".pyc",
    ".pyo",
    ".synctex.gz",
    ".tex",
}


def _tracked_files() -> list[Path]:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"git ls-files failed; export requires the clean anonymous git tree: {exc}") from exc

    files: list[Path] = []
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        rel = Path(raw.decode("utf-8"))
        _validate_release_path(rel)
        files.append(rel)
    return sorted(files, key=lambda p: p.as_posix())


def _validate_release_path(rel: Path) -> None:
    rel_posix = rel.as_posix()
    if rel.is_absolute() or ".." in rel.parts:
        raise SystemExit(f"refusing unsafe path: {rel_posix}")
    if any(part in BANNED_PARTS for part in rel.parts):
        raise SystemExit(f"refusing banned release path: {rel_posix}")
    if rel.name.endswith(".synctex.gz") or rel.suffix.lower() in BANNED_SUFFIXES:
        raise SystemExit(f"refusing banned release suffix: {rel_posix}")


def _write_zip(files: list[Path], output: Path) -> str:
    output = output.resolve()
    if ROOT in output.parents or output == ROOT:
        raise SystemExit("write the review package outside the repository root")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for rel in files:
            archive.write(ROOT / rel, rel.as_posix())

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a tracked-only anonymous review archive.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Zip path outside the repository root.")
    args = parser.parse_args()

    files = _tracked_files()
    digest = _write_zip(files, args.output)
    print("anonymous_review_package=created")
    print(f"path={args.output.resolve()}")
    print(f"files={len(files)}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
