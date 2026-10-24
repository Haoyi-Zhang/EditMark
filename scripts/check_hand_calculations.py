"""Four authored arithmetic examples, not observations of a watermark method.

Expected answers below are fixed hand calculations; the oracle does not call
the analyzer, bound helpers, or certifier. Only the comparison path does so.
No program, detector, model, or third-party package is executed.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from editmark_audit.analysis import summarize
from editmark_audit.certificate import certificates_from_raw_pairs
from editmark_audit.records import Pair
from editmark_audit.verify import verify_certificate

# (name, decisions, complete change, survival, target change interval)
CASES = (
    ("loss", ((1, 1), (1, 0), (0, 0)), "-1/3", "1/2", ("-1/3", "-1/3")),
    ("gain", ((0, 1), (0, 1), (1, 0)), "1/3", "0", ("1/3", "1/3")),
    ("missing-after", ((1, 1), (1, None), (0, 0)), "0", "1", ("-1/3", "0")),
    ("no-initial-positive", ((0, 0), (0, 1)), "1/2", None, ("1/2", "1/2")),
)

def require(condition, message):
    if not condition:
        raise ValueError(message)


def pair(case, index, before, after):
    def program(side, value):
        return {
            "source": f"# authored arithmetic example {case}/{index}/{side}\n",
            "validation": {"available": True, "passed": True,
                           "contract_id": "asserted-example-not-executed"},
            "detection": {"available": value is not None,
                          "detected": None if value is None else bool(value),
                          "score": None, "threshold": 1, "comparator": "ge",
                          "rule_id": "asserted-boolean-example"},
        }
    return dict(schema_version="editmark-pair", population="watermarked_positive",
                evidence_kind="synthetic_test", experiment_id=case,
                method="hand-calculation", model="none", source_group="authored",
                language="text", sample_id=str(index), replicate_id="0",
                edit="example", variant_id="0", supported=True,
                original=program("before", before), edited=program("after", after))


def run():
    result = []
    for name, decisions, change, survival, bounds in CASES:
        rows = [pair(name, i, b, a) for i, (b, a) in enumerate(decisions)]
        got = summarize([Pair.parse(row) for row in rows])
        expected_change = Fraction(change)
        require(Fraction(got["net_detection_change_exact"]["numerator"],
                         got["net_detection_change_exact"]["denominator"]) == expected_change,
                name + ": net-change disagreement")
        require(got["survival"] == (None if survival is None else float(Fraction(survival))),
                name + ": survival disagreement")
        interval = got["missing_decision_change_bounds"]
        actual_bounds = tuple(Fraction(interval[k], interval["denominator"])
                              for k in ("lower_numerator", "upper_numerator"))
        require(actual_bounds == tuple(map(Fraction, bounds)), name + ": interval disagreement")
        change_cert = certificates_from_raw_pairs(rows, "positive_decision_change")[0]
        survival_cert = certificates_from_raw_pairs(rows, "detection_survival")[0]
        for cert in (change_cert, survival_cert):
            require(verify_certificate(cert)["embedded_facts_verified"], name + ": invalid facts")
            require(cert["membership_verified"] is False, name + ": unsupported membership claim")
            require(cert["claim_text_verified"] is False, name + ": unsupported prose claim")
        result.append(dict(case=name, decisions=decisions,
                           complete_pairs=got["admitted_pairs"], target_pairs=len(rows),
                           complete_change=change, survival=survival,
                           target_change_bounds=bounds,
                           survival_decision=survival_cert["decision"]))
    return dict(evidence_kind="synthetic_test",
                meaning="Hand arithmetic and implementation agreement; no source authenticity claim.",
                cases=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run()
    payload = json.dumps(result, indent=2) + "\n"
    if args.out:
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
