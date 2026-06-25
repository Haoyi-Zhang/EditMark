# Reviewer Artifact Quickstart

## Three-Minute Claim Check

This is the shortest path for checking that the artifact matches the paper's central claim: clean detection is not enough for post-edit provenance unless the edited program, controls, support, utility, and cost remain visible.

```bash
python scripts/verify_release_integrity.py
python scripts/reviewer_workflow.py browse --summary-only
```

Expected review signals:

- `release_integrity=passed`;
- the canonical surface reports `140/140` completed configurations;
- `artifact/CLAIM_TO_EVIDENCE.md` maps the detection-to-robustness gap to the shipped summary tables and figures.

This check does not prove every result from scratch. It verifies that the frozen evidence surface is present, intact, anonymous, and navigable.

## Normal Browse Path

Use the lightweight path first:

```bash
python -m pip install -r requirements.txt
python scripts/verify_release_integrity.py
python scripts/reviewer_workflow.py browse
```

This path checks the canonical manifest digest, shipped summary-table hashes, required figures, source-slice counts, run inventory, and environment-capture files. It does not execute model generation, download model weights, or require private credentials.
