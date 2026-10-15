#!/usr/bin/env python3
"""Actual paired measurement on a fixed owner-authored finite carrier.

Opt-in only. Executes the 16 approved Python source texts and no other program,
model, upstream detector or edit/attack workflow. Every state and zero result
is retained. This is a native fixture integration check, not benchmark evidence.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixtures import owned_finite_carrier as owned
from editmark_audit.analysis import report
from editmark_audit.aggregates import analyze, from_pair_stratum
from editmark_audit import certificate, verify
from editmark_audit.io import digest, load_rows, write_new_json
from editmark_audit.records import Pair


def fraction(value):
    return [value.numerator, value.denominator]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def direct_observed(rows):
    """Separately written same-information arithmetic and literal admission."""
    counts = dict(n00=0, n01=0, n10=0, n11=0)
    for row in rows:
        before, after = row["original"], row["edited"]
        require(row["supported"] is True, "owned comparison requires supported pairs")
        require(digest_text(before["source"]) != digest_text(after["source"]), "owned pair must change source")
        require(all(p["source_sha256"] == digest_text(p["source"]) for p in (before, after)), "source fingerprint mismatch")
        require(before["validation"] == after["validation"], "functional contracts differ")
        require(before["validation"]["available"] is True, "functional observation unavailable")
        require(before["validation"]["passed"] is True, "functional validation failed")
        require(bool(before["validation"]["contract_id"]), "functional contract absent")
        require(all(p["detection"]["available"] is True and p["detection"]["rule_id"]
                    for p in (before, after)), "detector observations unavailable")
        require(all(before["detection"][k] == after["detection"][k]
                    for k in ("rule_id", "threshold", "comparator")), "detector rules differ")
        b, a = before["detection"]["detected"], after["detection"]["detected"]
        require(type(b) is bool and type(a) is bool, "detector decisions must be Boolean")
        require(all(p["detection"]["detected"] ==
                    (p["detection"]["score"] >= p["detection"]["threshold"])
                    for p in (before, after)), "detector score and decision disagree")
        counts[f"n{int(b)}{int(a)}"] += 1
    n = sum(counts.values())
    b, a = counts["n10"] + counts["n11"], counts["n01"] + counts["n11"]
    return {"transitions": counts, "n": n, "before_positive": b, "after_positive": a,
            "change": fraction(Fraction(a - b, n)),
            "survival": fraction(Fraction(counts["n11"], b)),
            "marginal_survival_bounds": [fraction(Fraction(max(0, a + b - n), b)),
                                         fraction(Fraction(min(a, b), b))]}


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Output directory must be new")
    started = datetime.now(timezone.utc).isoformat()
    fixture_path = ROOT / "fixtures/owned_finite_carrier.py"
    rule_id = "owned-finite-carrier:" + digest(fixture_path)
    rows, executions = [], []
    for bits in range(8):
        programs, native = [], []
        for edited in (False, True):
            source = owned.source_for(bits, edited)
            passed, calls = owned.measured_validation(source)
            score, decision = owned.measured_detection(source)
            programs.append({"source": source, "source_sha256": digest_text(source),
                "validation": {"available": True, "passed": passed, "contract_id": owned.CONTRACT_ID},
                "detection": {"available": True, "detected": decision, "score": score,
                              "threshold": 2, "comparator": "ge", "rule_id": rule_id}})
            native.append({"side": "edited" if edited else "original", "source_sha256": digest_text(source),
                           "calls": calls, "detector_score": score, "detector_decision": decision})
        rows.append({"schema_version": "editmark-pair", "population": "watermarked_positive",
            "evidence_kind": "synthetic_test", "experiment_id": "OWNED-FINITE-CARRIER",
            "method": "authored-three-bit-carrier", "model": "none",
            "source_group": "authored-finite-integration-fixture", "language": "python",
            "sample_id": format(bits, "03b"), "replicate_id": "one", "cluster_id": format(bits, "03b"),
            "edit": "neutral-review-annotation", "variant_id": "fixed-footer",
            "supported": True, "changed": True, "original": programs[0], "edited": programs[1]})
        executions.append({"sample_id": format(bits, "03b"), "native_observations": native})
    direct = direct_observed(rows)
    paired = report([Pair.parse(row) for row in rows])
    counts = from_pair_stratum(paired["strata"][0], "synthetic_test")
    exact = analyze(counts)
    require(direct["transitions"] == counts["transitions"], "direct transitions disagree with analyzer")
    require(direct["change"] == [exact["net_detection_change"][k] for k in ("numerator", "denominator")], "net change disagrees")
    require(direct["survival"] == [exact["survival"][k] for k in ("numerator", "denominator")], "survival disagrees")
    marginal = dict(counts); marginal.pop("transitions")
    coarse = analyze(marginal)
    require(direct["marginal_survival_bounds"] == [[coarse["survival_bounds"][side][k]
             for k in ("numerator", "denominator")] for side in ("lower", "upper")], "marginal survival bounds disagree")
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "pairs.jsonl"
    with raw.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write_new_json(output / "pair-report.json", paired)
    write_new_json(output / "joint-counts.json", counts)
    write_new_json(output / "marginal-counts.json", marginal)
    outcomes = {}
    for claim in certificate.CLAIMS:
        cert_path = output / (claim + ".json")
        status = certificate.main([str(raw), "--claim", claim, "--out", str(cert_path)])
        bundle = json.loads(cert_path.read_text(encoding="utf-8"))
        checked = verify.verify_bundle_against_basis(bundle, raw)
        write_new_json(output / (claim + "-verification.json"), checked)
        outcomes[claim] = {"exit_code": status, "decision": bundle["certificates"][0]["decision"],
                           "basis_kind": checked["basis_kind"],
                           "raw_pair_membership_rederived": checked["raw_pair_membership_rederived"]}
    reread = load_rows(raw)
    require(reread == rows, "serialized paired observations differ on reread")
    bindings = [{"path": name, "sha256": digest(ROOT / name)} for name in (
        "fixtures/owned_finite_carrier.py", "scripts/measure_owned_carrier.py",
        "editmark_audit/records.py", "editmark_audit/analysis.py",
        "editmark_audit/aggregates.py", "editmark_audit/certificate.py", "editmark_audit/verify.py")]
    value = {"scope": "Owner-authored finite integration fixture; not an independent study or upstream watermark evaluation",
        "evidence_kind": "synthetic_test", "source_origin": "Authored exact eight-state source set in owned_finite_carrier.py",
        "native_python_fixture_execution": True, "upstream_native_or_model_execution": False,
        "attack_workflow_execution": False, "third_party_target_access": False,
        "pair_count": len(rows), "native_source_versions": 16, "finite_contract_calls": 80,
        "same_information_direct": direct, "analyzer_agrees": True, "selection_or_detection_gain": 0,
        "contract_spec": owned.CONTRACT_SPEC, "detector_spec": owned.DETECTOR_SPEC,
        "executions": executions, "claim_outcomes": outcomes, "source_bindings": bindings,
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                        "argv": sys.argv},
        "interpretation": "Observed zero change and joint survival 1 on this authored fixture. Same-cohort marginals give [0,1] survival; direct arithmetic on the same pairs has exactly the same answers. No FPR/AUROC/posterior, security, novelty or deployment claim."}
    write_new_json(output / "measurement.json", value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    result = run(parser.parse_args().out)
    print(json.dumps({k: result[k] for k in ("pair_count", "finite_contract_calls",
                    "same_information_direct", "selection_or_detection_gain")}, indent=2))
