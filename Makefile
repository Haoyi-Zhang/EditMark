SOURCE_DATE_EPOCH ?= 1767225600
export SOURCE_DATE_EPOCH

PYTHON ?= python
LATEXMK ?= latexmk
BIBTEX ?= $(shell command -v bibtex 2>/dev/null || command -v bibtex.original 2>/dev/null)

.PHONY: help test facts assets lint hygiene paper check full-check clean
.DEFAULT_GOAL := help

help:
	@printf '%s\n' \
	  'test       Run the offline evidence-contract test suite.' \
	  'facts      Regenerate machine-readable facts and synthetic checker outputs.' \
	  'assets     Regenerate facts plus the manuscript figure.' \
	  'lint       Check the active LaTeX source graph, citations, labels, and draft markers.' \
	  'hygiene    Reject superseded manuscript trees and unresolved author metadata.' \
	  'paper      Build the ACM journal manuscript.' \
	  'check      Run all model-free checks; no generated program is executed.' \
	  'full-check Run model-free checks, regenerate assets, and rebuild the paper.'

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s tests/contract -v
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s regressions -v

facts:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/build_contract_artifacts.py --skip-figure

assets:
	MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/build_contract_artifacts.py
	MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/build_paper_assets.py

lint:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/verify_manuscript.py ../paper/tosem

hygiene:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/verify_release_hygiene.py .

paper:
	@test -n "$(BIBTEX)" || { echo 'A BibTeX executable is required.' >&2; exit 2; }
	cd ../paper/tosem && $(LATEXMK) -e '$$bibtex="$(BIBTEX) %O %B"' -xelatex -interaction=nonstopmode -halt-on-error main.tex

check: test facts lint hygiene

full-check: check assets paper

clean:
	cd ../paper/tosem && $(LATEXMK) -C main.tex || true
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '.coverage' \) -delete
