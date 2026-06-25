from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import certify_suite_precheck


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate that the single-host launch still matches the clean post-precheck launch receipt.")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--full-profile", type=str, required=True)
    parser.add_argument("--stage-a-manifest", type=Path, required=True)
    parser.add_argument("--stage-a-profile", type=str, required=True)
    parser.add_argument("--stage-b-manifest", type=Path, required=True)
    parser.add_argument("--stage-b-profile", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--precheck-gate", type=Path, default=Path("results/certifications/suite_precheck_gate.json"))
    parser.add_argument("--skip-hf-access", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = certify_suite_precheck._load_matching_launch_receipt(
        args,
        output_root=certify_suite_precheck._resolve(args.output_root),
        gate_path=certify_suite_precheck._resolve(args.precheck_gate),
    )
    if receipt is None:
        raise SystemExit("launch-time post-precheck receipt validation failed; rerun remote preflight and suite precheck cleanly.")
    print(
        json.dumps(
            {
                "status": "ok",
                "reason": "launch_time_post_precheck_receipt_match",
                "precheck_gate": str(certify_suite_precheck._resolve(args.precheck_gate)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
