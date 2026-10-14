"""Owner-benign source-basis and current-metadata workflow regressions."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from editmark_audit import certificate, verify
from editmark_audit.analysis import report
from editmark_audit.io import EvidenceError, digest, load_rows
from editmark_audit.records import Pair
from scripts.verify_release_hygiene import verify as hygiene


def paired_bundle(raw, claim):
    values = certificate.certificates_from_pair_report(
        report([Pair.parse(r) for r in load_rows(raw)]), claim)
    return {"schema_version": certificate.CERTIFICATE_BUNDLE_SCHEMA,
            "input_file": {"name": raw.name, "sha256": digest(raw)},
            "requested_claim": claim, "certificates": values}


class RawPairBasisTests(unittest.TestCase):
    def test_raw_jsonl_certificate_preserves_measured_joint_table(self):
        raw = ROOT / "examples/synthetic_count_basis.jsonl"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "certificate.json"
            self.assertEqual(certificate.main([str(raw), "--claim", "detection_survival",
                                               "--out", str(out)]), 0)
            value = json.loads(out.read_text())
            item = value["certificates"][0]
            self.assertEqual(item["transitions"], dict(n00=0, n01=1, n10=3, n11=1))
            self.assertEqual(item["estimate"], dict(numerator=1, denominator=4, decimal=.25))
            self.assertIs(item["membership_verified"], False)

    def test_raw_basis_rederives_records_not_a_supplied_pair_report(self):
        raw = ROOT / "examples/synthetic_count_basis.jsonl"
        result = verify.verify_bundle_against_basis(paired_bundle(raw, "positive_decision_change"), raw)
        self.assertTrue(result["basis_verified"])
        self.assertEqual(result["basis_kind"], "raw_pairs")
        self.assertTrue(result["raw_pair_membership_rederived"])

    def test_changed_basis_during_regeneration_is_rejected(self):
        original = ROOT / "examples/synthetic_cohort_counts.json"
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "counts.json"
            raw.write_bytes(original.read_bytes())
            values = certificate.certificates_from_count_documents(load_rows(raw), "positive_decision_change")
            bundle = {"schema_version": certificate.CERTIFICATE_BUNDLE_SCHEMA,
                      "input_file": {"name": raw.name, "sha256": digest(raw)},
                      "requested_claim": "positive_decision_change", "certificates": values}
            regenerate = verify._regenerate_from_basis
            def change_after_read(path, claim):
                values = regenerate(path, claim)
                path.write_bytes(path.read_bytes() + b"\n")
                return values
            with patch.object(verify, "_regenerate_from_basis", side_effect=change_after_read):
                with self.assertRaisesRegex(EvidenceError, "changed during"):
                    verify.verify_bundle_against_basis(bundle, raw)

    def test_conflicting_raw_identity_is_not_aggregated(self):
        rows = load_rows(ROOT / "examples/synthetic_count_basis.jsonl")
        rows.append(copy.deepcopy(rows[0]))
        rows[-1]["edited"]["source"] += "\n"
        with tempfile.TemporaryDirectory() as td:
            raw, out = Path(td) / "raw.json", Path(td) / "out.json"
            raw.write_text(json.dumps(rows))
            self.assertEqual(certificate.main([str(raw), "--claim", "positive_decision_change",
                                               "--out", str(out)]), 2)
            self.assertFalse(out.exists())

    def test_unsupported_claim_stays_denied_on_raw_pairs(self):
        raw = ROOT / "examples/synthetic_count_basis.jsonl"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "denial.json"
            self.assertEqual(certificate.main([str(raw), "--claim", "auroc_change",
                                               "--out", str(out)]), 3)
            self.assertEqual(json.loads(out.read_text())["certificates"][0]["decision"],
                             "not_admissible")

    def test_require_observations_still_refuses_authored_fixture(self):
        raw = ROOT / "examples/synthetic_count_basis.jsonl"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.json"
            self.assertEqual(certificate.main([str(raw), "--claim", "positive_decision_change",
                                               "--require-observations", "--out", str(out)]), 2)
            self.assertFalse(out.exists())


class CurrentMetadataTests(unittest.TestCase):
    def test_current_author_order_and_nested_accents_are_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifact"
            paper = root.parent / "paper/tosem"
            root.mkdir(); paper.mkdir(parents=True)
            # Literal current-project identity, independent of the validator's
            # constant. Code-only test discovery needs no sibling manuscript.
            (paper / "authors.tex").write_text(
                "\\author{Haoyi Zhang}\n\\author{Huaijin Ran}\n\\author{Xunzhu Tang}\n"
                "\\author{Kisub Kim}\n" + r"\author{Tegawend\'{e} F. Bissyand\'{e}}" + "\n",
                encoding="utf-8")
            (paper / "main.tex").write_text(r"\input{authors}", encoding="utf-8")
            (paper / "main.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            self.assertEqual(hygiene(root), [])


if __name__ == "__main__":
    unittest.main()
