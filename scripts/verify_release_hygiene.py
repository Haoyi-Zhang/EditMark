#!/usr/bin/env python3
"""Verify that the distributable repository contains one active final manuscript tree."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_manuscript import source_graph

FORBIDDEN_SEGMENT = re.compile(
    r"^(?:v\d+(?:[._-]\d+)*|final(?:[-_].*)?|reviewed(?:[-_].*)?|working(?:[-_].*)?|"
    r"draft(?:[-_].*)?|round\d*|history|old|backup|previous|superseded)$",
    re.IGNORECASE,
)
EXPECTED_AUTHORS = ("Huaijin Ran", "Haoyi Zhang", "Xunzhu Tang")


def verify(root: Path) -> list[str]:
    root = root.resolve()
    errors: list[str] = []
    if not root.is_dir():
        return [f"repository root does not exist: {root}"]

    ignored_dirs = {".git", ".pytest_cache", "__pycache__", "build", "dist"}
    for path in sorted(root.rglob("*")):
        if any(part in ignored_dirs for part in path.relative_to(root).parts):
            continue
        for segment in path.relative_to(root).parts:
            stem = Path(segment).stem
            if FORBIDDEN_SEGMENT.fullmatch(stem):
                errors.append(f"superseded-version path is not distributable: {path.relative_to(root)}")
                break

    paper = root.parent / "paper" / "tosem"
    sources, graph_errors = source_graph(paper)
    errors.extend(graph_errors)
    active = {p.resolve() for p in sources}
    all_tex = {
        p.resolve()
        for p in paper.rglob("*.tex")
        if not any(part in {"__pycache__"} for part in p.relative_to(paper).parts)
    }
    orphaned = sorted(all_tex - active)
    if orphaned:
        errors.append(
            "inactive manuscript sources: "
            + ", ".join(str(p.relative_to(paper)) for p in orphaned)
        )

    manuscript_pdfs = [
        p for p in paper.rglob("*.pdf")
        if "figures" not in p.relative_to(paper).parts
    ]
    if manuscript_pdfs != [paper / "main.pdf"]:
        shown = ", ".join(str(p.relative_to(paper)) for p in manuscript_pdfs) or "none"
        errors.append(f"expected exactly paper/tosem/main.pdf as manuscript PDF, found: {shown}")

    authors_path = paper / "authors.tex"
    if not authors_path.is_file():
        errors.append("missing paper/tosem/authors.tex")
    else:
        text = authors_path.read_text(encoding="utf-8")
        found = tuple(re.findall(r"\\author\{([^}]+)\}", text))
        if found != EXPECTED_AUTHORS:
            errors.append(f"author list is {found!r}, expected {EXPECTED_AUTHORS!r}")
        if re.search(r"placeholder|to be confirmed|future author|author slot", text, re.I):
            errors.append("author file contains an unresolved placeholder")

    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    errors = verify(args.root)
    if errors:
        for error in errors:
            print(f"RELEASE_HYGIENE_ERROR: {error}", file=sys.stderr)
        return 2
    print("RELEASE_HYGIENE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
