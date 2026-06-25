from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from posteditbench.hf_auth import resolve_token_env_value
from posteditbench.suite import SUITE_MODEL_ROSTER, require_pinned_model_revision

DEFAULT_MODELS = SUITE_MODEL_ROSTER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Hugging Face model access for the pinned model roster.")
    parser.add_argument("--token-env", default="HF_ACCESS_TOKEN", help="Environment variable containing the Hugging Face token.")
    parser.add_argument("--model", action="append", dest="models", default=None, help="Model id to probe. Repeat to override the default set.")
    parser.add_argument(
        "--revision",
        action="append",
        dest="revisions",
        default=None,
        help="Pinned revision to probe for each repeated --model. Canonical roster entries reject mismatched revisions; non-roster custom models require an explicit revision.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Network timeout in seconds.")
    parser.add_argument("--require-all", action="store_true", help="Exit with code 1 if any requested model is inaccessible.")
    parser.add_argument(
        "--require-token",
        action="store_true",
        help="Exit with code 1 when the selected token environment variable is empty.",
    )
    return parser.parse_args()


def _resolve_targets(models: list[str] | tuple[str, ...], revisions: list[str] | None) -> list[tuple[str, str]]:
    requested_models = [str(model).strip() for model in models if str(model).strip()]
    if revisions and len(revisions) != len(requested_models):
        raise SystemExit("--revision must be provided exactly once per --model.")
    targets: list[tuple[str, str]] = []
    for index, model_id in enumerate(requested_models):
        revision = ""
        if revisions:
            revision = str(revisions[index]).strip()
        revision = require_pinned_model_revision(model_id, revision)
        targets.append((model_id, revision))
    return targets


def _probe(model_id: str, revision: str, token: str, timeout: float) -> dict[str, object]:
    resolved_revision = str(revision or "").strip()
    if not resolved_revision:
        raise SystemExit(
            f"Missing pinned revision for {model_id}; benchmark access checks require model_name + model_revision."
        )
    url = f"https://huggingface.co/{model_id}/resolve/{resolved_revision}/config.json"
    request = urllib.request.Request(url, method="HEAD")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "model": model_id,
                "requested_revision": resolved_revision,
                "accessible": True,
                "status": getattr(response, "status", 200),
                "reason": "ok",
            }
    except urllib.error.HTTPError as exc:
        return {
            "model": model_id,
            "requested_revision": resolved_revision,
            "accessible": False,
            "status": exc.code,
            "reason": exc.reason,
        }
    except urllib.error.URLError as exc:
        return {
            "model": model_id,
            "requested_revision": resolved_revision,
            "accessible": False,
            "status": None,
            "reason": str(exc.reason),
        }


def main() -> int:
    args = parse_args()
    token = resolve_token_env_value(args.token_env)
    if args.require_token and not token:
        raise SystemExit(f"Missing {args.token_env}")

    models = tuple(args.models or DEFAULT_MODELS)
    targets = _resolve_targets(models, args.revisions)
    results = [_probe(model_id, revision, token, args.timeout) for model_id, revision in targets]
    accessible = [item["model"] for item in results if item["accessible"]]
    blocked = [item["model"] for item in results if not item["accessible"]]
    payload = {
        "token_env": args.token_env,
        "token_present": bool(token),
        "requested_models": list(models),
        "requested_targets": [
            {"model": model_id, "requested_revision": revision}
            for model_id, revision in targets
        ],
        "accessible_models": accessible,
        "blocked_models": blocked,
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_all and blocked:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
