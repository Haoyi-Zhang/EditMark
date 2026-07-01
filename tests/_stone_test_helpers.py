from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_REPO_URL = "https://github.com/inistory/STONE-watermarking.git"
DEFAULT_SWEET_REPO_URL = "https://github.com/hongcheki/sweet-watermark.git"
DEFAULT_EWD_REPO_URL = "https://github.com/luyijian3/EWD.git"
DEFAULT_KGW_REPO_URL = "https://github.com/jwkirchenbauer/lm-watermarking.git"

_RUNTIME_FIXTURES = {
    "stone_runtime": {
        "repo_url": DEFAULT_REPO_URL,
        "relative_path": "third_party/STONE-watermarking",
        "manifest_name": "STONE-watermarking.UPSTREAM.json",
        "external_root": ".coordination/external/STONE-watermarking.gitcheckout",
        "public_external_root": "external_checkout/STONE-watermarking",
        "license_filename": "LICENSE",
        "method_symbol": "STONE",
    },
    "sweet_runtime": {
        "repo_url": DEFAULT_SWEET_REPO_URL,
        "relative_path": "third_party/sweet-watermark",
        "manifest_name": "SWEET-watermark.UPSTREAM.json",
        "external_root": ".coordination/external/sweet-watermark.gitcheckout",
        "public_external_root": "external_checkout/sweet-watermark",
        "license_filename": "LICENSE",
        "method_symbol": "SWEET",
    },
    "ewd_runtime": {
        "repo_url": DEFAULT_EWD_REPO_URL,
        "relative_path": "third_party/EWD",
        "manifest_name": "EWD.UPSTREAM.json",
        "external_root": ".coordination/external/EWD.gitcheckout",
        "public_external_root": "external_checkout/EWD",
        "license_filename": "LICENSE",
        "method_symbol": "EWD",
    },
    "kgw_runtime": {
        "repo_url": DEFAULT_KGW_REPO_URL,
        "relative_path": "third_party/lm-watermarking",
        "manifest_name": "KGW-lm-watermarking.UPSTREAM.json",
        "external_root": ".coordination/external/lm-watermarking.gitcheckout",
        "public_external_root": "external_checkout/lm-watermarking",
        "license_filename": "LICENSE.md",
        "method_symbol": "KGW",
    },
}


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    run_env = os.environ.copy()
    run_env.update(
        {
            "GIT_AUTHOR_NAME": "PostEditBench",
            "GIT_AUTHOR_EMAIL": "posteditbench@example.invalid",
            "GIT_COMMITTER_NAME": "PostEditBench",
            "GIT_COMMITTER_EMAIL": "posteditbench@example.invalid",
        }
    )
    if env is not None:
        run_env.update(env)
    completed = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        env=run_env,
    )
    return completed.stdout.strip()


def create_stone_checkout(
    root: Path,
    *,
    relative_path: str = "third_party/STONE-watermarking",
    manifest_root: Path | None = None,
    repo_url: str = DEFAULT_REPO_URL,
    include_license: bool = True,
    dirty: bool = False,
    license_status: str | None = None,
) -> tuple[Path, Path, str]:
    if license_status is None:
        license_status = "redistributable" if include_license else "unverified"
    checkout = root / relative_path
    if checkout.exists():
        shutil.rmtree(checkout)
    checkout.mkdir(parents=True, exist_ok=True)
    _git(checkout, "init")
    _git(checkout, "config", "user.name", "PostEditBench")
    _git(checkout, "config", "user.email", "posteditbench@example.invalid")
    _git(checkout, "config", "core.fileMode", "false")

    stone_root = checkout / "stone_implementation"
    stone_root.mkdir(parents=True, exist_ok=True)
    (stone_root / "__init__.py").write_text("", encoding="utf-8", newline="\n")
    (stone_root / "README.md").write_text("stone checkout fixture\n", encoding="utf-8", newline="\n")
    (stone_root / "watermark" / "auto_watermark.py").parent.mkdir(parents=True, exist_ok=True)
    (stone_root / "watermark" / "auto_watermark.py").write_text("# fixture\n", encoding="utf-8", newline="\n")
    (stone_root / "utils" / "transformers_config.py").parent.mkdir(parents=True, exist_ok=True)
    (stone_root / "utils" / "transformers_config.py").write_text("# fixture\n", encoding="utf-8", newline="\n")
    if include_license:
        (checkout / "LICENSE").write_text("redistributable", encoding="utf-8", newline="\n")

    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-m", "seed stone checkout")
    commit = _git(checkout, "rev-parse", "HEAD")
    _git(checkout, "remote", "add", "origin", repo_url)

    if dirty:
        (checkout / "dirty.txt").write_text("dirty\n", encoding="utf-8", newline="\n")

    if manifest_root is None:
        manifest_root = root / ".pytest-manifests"
    manifest_path = manifest_root / "STONE-watermarking.UPSTREAM.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo_url": repo_url,
                "pinned_commit": commit,
                "license_status": license_status,
                "checkout_root": "third_party/STONE-watermarking",
                "external_root": ".coordination/external/STONE-watermarking.gitcheckout",
                "notes": "test fixture manifest",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return checkout, manifest_path, commit


def create_runtime_checkout(
    root: Path,
    method: str,
    *,
    relative_path: str | None = None,
    manifest_root: Path | None = None,
    include_license: bool = True,
    dirty: bool = False,
    license_status: str | None = None,
) -> tuple[Path, Path, str]:
    fixture = dict(_RUNTIME_FIXTURES[method])
    if license_status is None:
        license_status = "redistributable" if include_license else "unverified"
    checkout = root / (relative_path or fixture["relative_path"])
    if checkout.exists():
        shutil.rmtree(checkout)
    checkout.mkdir(parents=True, exist_ok=True)
    _git(checkout, "init")
    _git(checkout, "config", "user.name", "PostEditBench")
    _git(checkout, "config", "user.email", "posteditbench@example.invalid")
    _git(checkout, "config", "core.fileMode", "false")

    if method == "sweet_runtime":
        (checkout / "sweet.py").write_text("class SweetLogitsProcessor: pass\nclass SweetDetector: pass\n", encoding="utf-8", newline="\n")
        (checkout / "watermark.py").write_text("class WatermarkDetector: pass\nclass WatermarkLogitsProcessor: pass\n", encoding="utf-8", newline="\n")
    elif method == "ewd_runtime":
        (checkout / "watermark.py").write_text("class WatermarkDetector: pass\nclass WatermarkLogitsProcessor: pass\n", encoding="utf-8", newline="\n")
    elif method == "kgw_runtime":
        (checkout / "extended_watermark_processor.py").write_text("class WatermarkDetector: pass\nclass WatermarkLogitsProcessor: pass\n", encoding="utf-8", newline="\n")
        (checkout / "alternative_prf_schemes.py").write_text("def seeding_scheme_lookup(x): return ('simple',1,False,1)\ndef prf_lookup(*args, **kwargs): return {}\n", encoding="utf-8", newline="\n")
        (checkout / "normalizers.py").write_text("normalization_strategy_lookup = {}\n", encoding="utf-8", newline="\n")
    elif method == "stone_runtime":
        return create_stone_checkout(
            root,
            relative_path=relative_path or fixture["relative_path"],
            manifest_root=manifest_root,
            repo_url=fixture["repo_url"],
            include_license=include_license,
            dirty=dirty,
            license_status=license_status,
        )

    if include_license:
        (checkout / fixture["license_filename"]).write_text("fixture-license", encoding="utf-8", newline="\n")

    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-m", f"seed {method} checkout")
    commit = _git(checkout, "rev-parse", "HEAD")
    _git(checkout, "remote", "add", "origin", fixture["repo_url"])

    if dirty:
        (checkout / "dirty.txt").write_text("dirty\n", encoding="utf-8", newline="\n")

    if manifest_root is None:
        manifest_root = root / ".pytest-manifests"
    manifest_path = manifest_root / fixture["manifest_name"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo_url": fixture["repo_url"],
                "pinned_commit": commit,
                "license_status": license_status,
                "checkout_root": fixture["relative_path"],
                "external_root": fixture["external_root"],
                "source_relative": ".",
                "public_external_root": fixture["public_external_root"],
                "method_symbol": fixture["method_symbol"],
                "notes": "test fixture manifest",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return checkout, manifest_path, commit


def update_manifest(manifest_path: Path, **updates: Any) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(updates)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
