#!/usr/bin/env python3
"""Deterministic manuscript checks without invoking TeX."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys
from typing import Iterable

INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
CITE_RE = re.compile(r"\\cite(?!style\b)[a-zA-Z]*\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}")
BIB_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)", re.I)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:ref|eqref|autoref|pageref)\{([^}]+)\}")
FIGURE_RE = re.compile(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.S)


def format_errors(active: str) -> list[str]:
    """Check the ACM submission interface and accessibility hooks."""
    clean = strip_comments(active)
    errors: list[str] = []
    if not re.search(
        r"\\documentclass\[(?=[^\]]*\bmanuscript\b)(?=[^\]]*\breview\b)[^\]]*\]\{acmart\}",
        clean,
    ):
        errors.append('main document must use acmart manuscript,review mode')
    if r'\acmJournal{TOSEM}' not in clean:
        errors.append('manuscript must declare TOSEM')
    if 'printacmref=true' not in clean.replace(' ', ''):
        errors.append('ACM reference block must remain enabled')
    forbidden_patterns = {
        'geometry package': r"\\usepackage(?:\[[^\]]*\])?\{geometry\}",
        'manual geometry': r"\\geometry\s*\{",
        'manual text width': r"\\setlength\s*\{\\textwidth\}",
        'manual margins': r"\\(?:oddsidemargin|evensidemargin|topmargin)\b",
        'manual line spacing': r"\\(?:linespread|onehalfspacing|doublespacing)\b",
        'manual baseline stretch': r"\\renewcommand\s*\{\\baselinestretch\}",
    }
    for label, pattern in forbidden_patterns.items():
        if re.search(pattern, clean):
            errors.append(f'forbidden layout override: {label}')
    for index, body in enumerate(FIGURE_RE.findall(clean), 1):
        if r'\Description{' not in body:
            errors.append(f'figure {index} is missing an ACM Description')
    return errors


def strip_comments(text: str) -> str:
    lines=[]
    for line in text.splitlines():
        out=[];escaped=False
        for char in line:
            if char=='%' and not escaped:
                break
            out.append(char)
            escaped=(char=='\\' and not escaped)
            if char!='\\': escaped=False
        lines.append(''.join(out))
    return '\n'.join(lines)


def citation_keys(text: str) -> set[str]:
    return {
        key.strip()
        for match in CITE_RE.finditer(strip_comments(text))
        for key in match.group(1).split(',')
        if key.strip()
    }


def source_graph(root: Path) -> tuple[dict[Path,str], list[str]]:
    root=Path(root).resolve();start=root/'main.tex';seen={};visiting=[];errors=[]
    def visit(path: Path):
        path=path.resolve()
        if root not in path.parents and path!=root:
            errors.append(f'input escapes manuscript root: {path}');return
        if path in visiting:
            errors.append('input cycle: '+' -> '.join(p.name for p in visiting+[path]));return
        if path in seen:return
        if not path.is_file():
            errors.append(f'missing input: {path.relative_to(root) if root in path.parents else path}');return
        try:text=path.read_text(encoding='utf-8')
        except UnicodeError as exc:errors.append(f'cannot decode {path}: {exc}');return
        seen[path]=text;visiting.append(path)
        for match in INPUT_RE.finditer(strip_comments(text)):
            name=match.group(1).strip()
            # TeX resolves manuscript inputs from the compilation working directory.
            # Prefer the manuscript root, while retaining a file-relative fallback for
            # self-contained subtrees.
            candidate=(root/name)
            if candidate.suffix=='':
                candidate=candidate.with_suffix('.tex')
            if not candidate.is_file():
                candidate=(path.parent/name)
                if candidate.suffix=='':
                    candidate=candidate.with_suffix('.tex')
            visit(candidate)
        visiting.pop()
    visit(start)
    return seen,errors


def bib_keys(path: Path) -> set[str]:
    return set(BIB_RE.findall(path.read_text(encoding='utf-8')))


def verify(root: Path) -> list[str]:
    sources,errors=source_graph(root)
    if errors:return errors
    active='\n'.join(sources.values())
    errors.extend(format_errors(active))
    citations=citation_keys(active)
    bibliography=bib_keys(root/'references.bib')
    missing=sorted(citations-bibliography)
    unused=sorted(bibliography-citations)
    if missing:errors.append('missing bibliography keys: '+', '.join(missing))
    if unused:errors.append('uncited bibliography entries: '+', '.join(unused))
    labels=LABEL_RE.findall(strip_comments(active));refs=REF_RE.findall(strip_comments(active))
    duplicates=sorted({x for x in labels if labels.count(x)>1})
    undefined=sorted(set(refs)-set(labels))
    if duplicates:errors.append('duplicate labels: '+', '.join(duplicates))
    if undefined:errors.append('undefined references: '+', '.join(undefined))
    forbidden=('working draft', 'previous revision', 'superseded manuscript')
    low=active.lower()
    for phrase in forbidden:
        if phrase in low:errors.append(f'forbidden draft marker in active manuscript: {phrase}')
    return errors


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',nargs='?',type=Path,default=Path('../paper/tosem'))
    args=parser.parse_args(argv)
    errors=verify(args.root)
    if errors:
        for error in errors:print('MANUSCRIPT_ERROR:',error,file=sys.stderr)
        return 2
    print('MANUSCRIPT_OK')
    return 0

if __name__=='__main__':raise SystemExit(main())
