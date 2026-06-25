from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = 1

_RUNTIME_NAMES = (
    "stone_runtime",
    "sweet_runtime",
    "ewd_runtime",
    "kgw_runtime",
)
_BASELINE_NAMES = _RUNTIME_NAMES
_LICENSE_FILENAMES = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "COPYING",
    "COPYING.txt",
    "NOTICE",
    "NOTICE.txt",
)
_IGNORED_CHECKOUT_ARTIFACT_DIRS = (
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
)
_IGNORED_CHECKOUT_ARTIFACT_SUFFIXES = (
    ".pyc",
    ".pyo",
)
_METHOD_SPECS: dict[str, dict[str, str]] = {
    "stone_runtime": {
        "method_symbol": "STONE",
        "repo_url": "https://github.com/inistory/STONE-watermarking.git",
        "pinned_commit": "bb5d809c0c494a219411e861f2313cca2b9fd7b4",
        "manifest_name": "STONE-watermarking.UPSTREAM.json",
        "checkout_root": "third_party/STONE-watermarking",
        "external_root": "external_checkout/STONE-watermarking",
        "source_relative": "stone_implementation",
        "public_external_root": "external_checkout/STONE-watermarking",
        "license_status": "unverified",
        "env_root": "POSTEDITBENCH_STONE_UPSTREAM_ROOT",
        "env_manifest": "POSTEDITBENCH_STONE_UPSTREAM_MANIFEST",
        "notes": "Pinned STONE upstream implementation. The artifact keeps the watermark algorithm logic intact while routing orchestration, local model loading, and decoding policy through the shared benchmark runner.",
    },
    "sweet_runtime": {
        "method_symbol": "SWEET",
        "repo_url": "https://github.com/hongcheki/sweet-watermark.git",
        "pinned_commit": "853b47eb064c180beebd383302d09491fc98a565",
        "manifest_name": "SWEET-watermark.UPSTREAM.json",
        "checkout_root": "third_party/sweet-watermark",
        "external_root": "external_checkout/sweet-watermark",
        "source_relative": ".",
        "public_external_root": "external_checkout/sweet-watermark",
        "license_status": "unverified",
        "env_root": "POSTEDITBENCH_SWEET_UPSTREAM_ROOT",
        "env_manifest": "POSTEDITBENCH_SWEET_UPSTREAM_MANIFEST",
        "notes": "Pinned SWEET upstream implementation. The artifact keeps the watermark algorithm logic intact while routing orchestration, local model loading, and decoding policy through the shared benchmark runner.",
    },
    "ewd_runtime": {
        "method_symbol": "EWD",
        "repo_url": "https://github.com/luyijian3/EWD.git",
        "pinned_commit": "605756acf802528a3df89d95a4661a031eafc79b",
        "manifest_name": "EWD.UPSTREAM.json",
        "checkout_root": "third_party/EWD",
        "external_root": "external_checkout/EWD",
        "source_relative": ".",
        "public_external_root": "external_checkout/EWD",
        "license_status": "unverified",
        "env_root": "POSTEDITBENCH_EWD_UPSTREAM_ROOT",
        "env_manifest": "POSTEDITBENCH_EWD_UPSTREAM_MANIFEST",
        "notes": "Pinned EWD upstream implementation. The artifact keeps the watermark algorithm logic intact while routing orchestration, local model loading, and decoding policy through the shared benchmark runner.",
    },
    "kgw_runtime": {
        "method_symbol": "KGW",
        "repo_url": "https://github.com/jwkirchenbauer/lm-watermarking.git",
        "pinned_commit": "82922516930c02f8aa322765defdb5863d07a00e",
        "manifest_name": "KGW-lm-watermarking.UPSTREAM.json",
        "checkout_root": "third_party/lm-watermarking",
        "external_root": "external_checkout/lm-watermarking",
        "source_relative": ".",
        "public_external_root": "external_checkout/lm-watermarking",
        "license_status": "redistributable",
        "env_root": "POSTEDITBENCH_KGW_UPSTREAM_ROOT",
        "env_manifest": "POSTEDITBENCH_KGW_UPSTREAM_MANIFEST",
        "notes": "Pinned KGW upstream implementation from jwkirchenbauer/lm-watermarking. The artifact keeps the watermark algorithm logic intact while routing orchestration, local model loading, and decoding policy through the shared benchmark runner.",
    },
}
_REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "stone_runtime": ("watermark/auto_watermark.py", "utils/transformers_config.py"),
    "sweet_runtime": ("sweet.py", "watermark.py"),
    "ewd_runtime": ("watermark.py",),
    "kgw_runtime": ("extended_watermark_processor.py", "alternative_prf_schemes.py", "normalizers.py"),
}


@dataclass(frozen=True, slots=True)
class CheckoutInfo:
    method_name: str
    method_symbol: str
    repo_root: Path
    source_root: Path
    origin: str
    source: str
    manifest_path: Path
    manifest_schema_version: int
    manifest_repo_url: str
    manifest_pinned_commit: str
    manifest_license_status: str
    remote_url: str
    upstream_commit: str
    dirty: bool
    license_files: tuple[str, ...]
    license_path: str | None
    issues: tuple[str, ...]
    redistributable: bool
    source_relative: str
    public_source_root: str
    public_external_root: str

    @property
    def is_valid(self) -> bool:
        return not self.issues

    @property
    def bundle_eligible(self) -> bool:
        return self.is_valid and self.origin == "vendored_snapshot" and self.redistributable

    @property
    def stone_root(self) -> Path:
        return self.source_root

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "method_symbol": self.method_symbol,
            "repo_root": str(self.repo_root),
            "source_root": str(self.source_root),
            "stone_root": str(self.source_root),
            "origin": self.origin,
            "source": self.source,
            "manifest_path": str(self.manifest_path),
            "manifest_schema_version": self.manifest_schema_version,
            "manifest_repo_url": self.manifest_repo_url,
            "manifest_pinned_commit": self.manifest_pinned_commit,
            "manifest_license_status": self.manifest_license_status,
            "repo_url": self.manifest_repo_url,
            "pinned_commit": self.manifest_pinned_commit,
            "license_status": self.manifest_license_status,
            "remote_url": self.remote_url,
            "upstream_commit": self.upstream_commit,
            "dirty": self.dirty,
            "license_files": list(self.license_files),
            "license_path": self.license_path,
            "issues": list(self.issues),
            "checkout_present": True,
            "checkout_valid": self.is_valid,
            "bundle_eligible": self.bundle_eligible,
            "redistributable": self.redistributable,
            "source_relative": self.source_relative,
            "source_path": public_source_path(self.origin, self.method_name),
            "external_path": self.public_external_root if self.origin == "external_checkout" else "",
        }


def runtime_watermark_names() -> tuple[str, ...]:
    return _RUNTIME_NAMES


def stone_family_baseline_names() -> tuple[str, ...]:
    return _BASELINE_NAMES


def _normalize_method_name(method: str | None) -> str:
    normalized = str(method or "stone_runtime").strip().lower()
    if normalized not in _METHOD_SPECS:
        raise KeyError(f"unknown runtime watermark method: {method!r}")
    return normalized


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _method_spec(method: str | None) -> dict[str, str]:
    return dict(_METHOD_SPECS[_normalize_method_name(method)])


def _manifest_path(method: str | None) -> Path:
    spec = _method_spec(method)
    override = os.environ.get(spec["env_manifest"], "").strip()
    if override:
        return Path(override)
    return _workspace_root() / "third_party" / spec["manifest_name"]


def _default_manifest(method: str | None) -> dict[str, Any]:
    spec = _method_spec(method)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "repo_url": spec["repo_url"],
        "pinned_commit": spec["pinned_commit"],
        "license_status": spec["license_status"],
        "checkout_root": spec["checkout_root"],
        "external_root": spec["external_root"],
        "source_relative": spec["source_relative"],
        "public_external_root": spec["public_external_root"],
        "method_symbol": spec["method_symbol"],
        "notes": spec["notes"],
    }


def _load_manifest(method: str | None) -> dict[str, Any]:
    path = _manifest_path(method)
    payload = _default_manifest(method)
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload.update(loaded)
    try:
        payload["schema_version"] = int(payload.get("schema_version", MANIFEST_SCHEMA_VERSION))
    except Exception:
        payload["schema_version"] = MANIFEST_SCHEMA_VERSION
    return payload


def load_upstream_manifest(method: str | None = None) -> dict[str, Any]:
    return dict(_load_manifest(method))


def pinned_upstream_commit(method: str | None = None) -> str:
    return str(_load_manifest(method).get("pinned_commit", _method_spec(method)["pinned_commit"]))


def public_source_path(origin: str, method: str | None = None) -> str:
    manifest = _load_manifest(method)
    checkout_root = str(manifest.get("checkout_root", _method_spec(method)["checkout_root"])).strip()
    public_external_root = str(
        manifest.get("public_external_root", _method_spec(method)["public_external_root"])
    ).strip()
    normalized = str(origin).strip().lower()
    if normalized == "vendored_snapshot":
        return checkout_root
    if normalized == "external_checkout":
        return public_external_root
    return ""


def _candidate_roots(method: str | None) -> list[tuple[Path, str]]:
    spec = _method_spec(method)
    root = _workspace_root()
    manifest = _load_manifest(method)
    vendored_root = str(manifest.get("checkout_root", spec["checkout_root"])).strip() or spec["checkout_root"]
    public_external_root = str(
        manifest.get("public_external_root", spec["public_external_root"])
    ).strip() or spec["public_external_root"]
    external_root = str(manifest.get("external_root", public_external_root)).strip() or public_external_root
    local_external_root = str(manifest.get("local_external_root", external_root)).strip() or external_root
    source_relative = str(manifest.get("source_relative", spec["source_relative"])).strip() or spec["source_relative"]
    candidates: list[tuple[Path, str]] = []
    env_root = os.environ.get(spec["env_root"], "").strip()
    if env_root:
        candidates.append((Path(env_root), "env"))
    candidates.extend(
        [
            (root / vendored_root, "vendored"),
            (root / local_external_root, "external"),
        ]
    )
    if external_root != local_external_root:
        candidates.append((root / external_root, "external"))
    if source_relative not in {"", "."}:
        candidates.extend(
            [
                (root / vendored_root / source_relative, "vendored"),
                (root / local_external_root / source_relative, "external"),
            ]
        )
        if external_root != local_external_root:
            candidates.append((root / external_root / source_relative, "external"))
    return candidates


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _normalize_git_status_path(text: str) -> str:
    return text.strip().replace("\\", "/")


def _is_ignored_checkout_artifact(path_text: str) -> bool:
    normalized = _normalize_git_status_path(path_text)
    if not normalized:
        return False
    parts = [part for part in normalized.split("/") if part]
    if any(part in _IGNORED_CHECKOUT_ARTIFACT_DIRS for part in parts):
        return True
    return normalized.endswith(_IGNORED_CHECKOUT_ARTIFACT_SUFFIXES)


def _status_entry_paths(status_line: str) -> tuple[str, ...]:
    body = status_line[3:].strip()
    if not body:
        return ()
    if " -> " in body:
        return tuple(part.strip() for part in body.split(" -> ") if part.strip())
    return (body,)


def _checkout_dirty(repo_root: Path) -> bool:
    porcelain = _git_output(repo_root, "status", "--porcelain")
    if not porcelain:
        return False
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        entry_paths = _status_entry_paths(line)
        if entry_paths and all(_is_ignored_checkout_artifact(path) for path in entry_paths):
            continue
        return True
    return False


def _license_files(path: Path) -> list[str]:
    return [name for name in _LICENSE_FILENAMES if (path / name).exists()]


def _required_paths_exist(source_root: Path, method: str) -> tuple[str, ...]:
    required = _REQUIRED_FILES.get(method, ())
    missing = [relative for relative in required if not (source_root / relative).exists()]
    return tuple(missing)


def resolve_checkout_commit(repo_root: Path | None, method: str | None = None) -> str:
    if repo_root is None:
        return pinned_upstream_commit(method)
    commit = _git_output(repo_root, "rev-parse", "HEAD")
    return commit or pinned_upstream_commit(method)


def _normalize_checkout_root(method: str | None, path: Path, source: str) -> CheckoutInfo | None:
    normalized_method = _normalize_method_name(method)
    candidate = path.resolve()
    if not candidate.exists():
        return None

    manifest = _load_manifest(normalized_method)
    source_relative_text = str(manifest.get("source_relative", _method_spec(normalized_method)["source_relative"])).strip()
    source_relative = Path(source_relative_text or ".")
    if source_relative_text in {"", "."}:
        repo_root = candidate
        source_root = candidate
    elif (candidate / source_relative).exists():
        repo_root = candidate
        source_root = candidate / source_relative
    elif candidate.name == source_relative.name and (candidate.parent / ".git").exists():
        repo_root = candidate.parent
        source_root = candidate
    else:
        return None

    manifest_path = _manifest_path(normalized_method)
    manifest_repo_url = str(manifest.get("repo_url", _method_spec(normalized_method)["repo_url"]))
    manifest_commit = str(manifest.get("pinned_commit", _method_spec(normalized_method)["pinned_commit"]))
    manifest_license_status = str(manifest.get("license_status", _method_spec(normalized_method)["license_status"])).strip().lower()
    method_symbol = str(manifest.get("method_symbol", _method_spec(normalized_method)["method_symbol"])).strip() or normalized_method
    public_external_root = str(
        manifest.get("public_external_root", _method_spec(normalized_method)["public_external_root"])
    ).strip()

    remote_url = _git_output(repo_root, "remote", "get-url", "origin")
    head_commit = _git_output(repo_root, "rev-parse", "HEAD")
    dirty = _checkout_dirty(repo_root)
    license_files = tuple(_license_files(repo_root))
    license_path = license_files[0] if license_files else None
    is_work_tree = _git_output(repo_root, "rev-parse", "--is-inside-work-tree").lower() == "true"

    if source == "external" or "external_checkout" in repo_root.parts or repo_root.name.endswith(".gitcheckout"):
        origin = "external_checkout"
    elif "third_party" in repo_root.parts or source == "vendored":
        origin = "vendored_snapshot"
    else:
        origin = "env_override"

    issues: list[str] = []
    if not is_work_tree:
        issues.append("missing git checkout metadata")
    if not remote_url:
        issues.append("missing origin remote")
    elif remote_url != manifest_repo_url:
        issues.append(f"origin remote mismatch: expected {manifest_repo_url}, found {remote_url}")
    if not head_commit:
        issues.append("missing git HEAD commit")
    elif head_commit != manifest_commit:
        issues.append(f"checkout commit mismatch: expected {manifest_commit}, found {head_commit}")
    if dirty:
        issues.append("checkout has uncommitted changes")
    missing_required_paths = _required_paths_exist(source_root, normalized_method)
    if missing_required_paths:
        issues.append(
            "missing required upstream source files: " + ", ".join(sorted(missing_required_paths))
        )
    if origin == "vendored_snapshot" and not license_files:
        issues.append("vendored checkout is missing LICENSE/COPYING files")
    if manifest_license_status == "redistributable" and not license_files:
        issues.append("manifest marks checkout redistributable but no license files were found")

    return CheckoutInfo(
        method_name=normalized_method,
        method_symbol=method_symbol,
        repo_root=repo_root,
        source_root=source_root,
        origin=origin,
        source=source,
        manifest_path=manifest_path,
        manifest_schema_version=int(manifest.get("schema_version", MANIFEST_SCHEMA_VERSION)),
        manifest_repo_url=manifest_repo_url,
        manifest_pinned_commit=manifest_commit,
        manifest_license_status=manifest_license_status,
        remote_url=remote_url,
        upstream_commit=head_commit or manifest_commit,
        dirty=dirty,
        license_files=license_files,
        license_path=license_path,
        issues=tuple(issues),
        redistributable=bool(license_files),
        source_relative=source_relative_text or ".",
        public_source_root=str(manifest.get("checkout_root", _method_spec(normalized_method)["checkout_root"])).strip(),
        public_external_root=public_external_root,
    )


def stone_family_checkout_status(method: str | None = None) -> CheckoutInfo | None:
    normalized_method = _normalize_method_name(method)
    first_invalid: CheckoutInfo | None = None
    for candidate, source in _candidate_roots(normalized_method):
        checkout = _normalize_checkout_root(normalized_method, candidate, source)
        if checkout is not None:
            if checkout.is_valid:
                return checkout
            if first_invalid is None:
                first_invalid = checkout
    return first_invalid


def resolve_checkout(method: str | None = None) -> CheckoutInfo | None:
    checkout = stone_family_checkout_status(method)
    if checkout is None or not checkout.is_valid:
        return None
    return checkout


def stone_family_checkout_available(method: str | None = None) -> bool:
    return resolve_checkout(method) is not None


def stone_family_checkout_metadata(method: str | None = None) -> dict[str, Any]:
    normalized_method = _normalize_method_name(method)
    manifest = _load_manifest(normalized_method)
    checkout = stone_family_checkout_status(normalized_method)
    payload = {
        "baseline_family": "runtime_official",
        "method_name": normalized_method,
        "method_symbol": str(manifest.get("method_symbol", _method_spec(normalized_method)["method_symbol"])),
        "manifest_path": str(_manifest_path(normalized_method)),
        "manifest_schema_version": int(manifest.get("schema_version", MANIFEST_SCHEMA_VERSION)),
        "manifest_repo_url": str(manifest.get("repo_url", _method_spec(normalized_method)["repo_url"])),
        "manifest_pinned_commit": str(manifest.get("pinned_commit", _method_spec(normalized_method)["pinned_commit"])),
        "manifest_license_status": str(
            manifest.get("license_status", _method_spec(normalized_method)["license_status"])
        ),
        "repo_url": str(manifest.get("repo_url", _method_spec(normalized_method)["repo_url"])),
        "pinned_commit": str(manifest.get("pinned_commit", _method_spec(normalized_method)["pinned_commit"])),
        "license_status": str(manifest.get("license_status", _method_spec(normalized_method)["license_status"])),
        "notes": str(manifest.get("notes", _method_spec(normalized_method)["notes"])),
        "checkout_present": checkout is not None,
        "checkout_valid": bool(checkout.is_valid) if checkout is not None else False,
        "checkout_issues": list(checkout.issues) if checkout is not None else [f"missing local checkout for {normalized_method}"],
    }
    if checkout is not None:
        payload.update(checkout.as_dict())
    else:
        payload.update(
            {
                "repo_root": None,
                "source_root": None,
                "stone_root": None,
                "origin": "missing",
                "source": "missing",
                "remote_url": "",
                "upstream_commit": str(manifest.get("pinned_commit", _method_spec(normalized_method)["pinned_commit"])),
                "dirty": False,
                "license_files": [],
                "license_path": None,
                "issues": [f"missing local checkout for {normalized_method}"],
                "bundle_eligible": False,
                "redistributable": False,
                "source_relative": str(manifest.get("source_relative", _method_spec(normalized_method)["source_relative"])),
                "source_path": "",
                "external_path": "",
            }
        )
    return payload


def validate_checkout(method: str | None = None, *, require_redistributable: bool = False) -> list[str]:
    normalized_method = _normalize_method_name(method)
    checkout = stone_family_checkout_status(normalized_method)
    if checkout is None:
        manifest_name = _method_spec(normalized_method)["manifest_name"]
        return [
            f"runtime watermark '{normalized_method}' requires a local official checkout that matches third_party/{manifest_name}"
        ]
    issues = list(checkout.issues)
    if require_redistributable and not checkout.redistributable:
        issues.append("vendored baseline snapshot is missing an explicit redistributable LICENSE/COPYING file")
    return issues


@contextmanager
def temporary_sys_path(path: Path):
    text = str(path)
    already_present = text in sys.path
    if not already_present:
        sys.path.insert(0, text)
    try:
        yield
    finally:
        if not already_present:
            try:
                sys.path.remove(text)
            except ValueError:
                pass
