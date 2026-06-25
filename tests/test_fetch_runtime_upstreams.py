from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from posteditbench.baselines.stone_family import common as stone_common


BASH_EXE = shutil.which("bash")
PINNED_COMMIT = "bb5d809c0c494a219411e861f2313cca2b9fd7b4"
REMOTE_URL = "https://example.invalid/STONE-watermarking.git"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _wsl_path(path: Path) -> str:
    if BASH_EXE is None:  # pragma: no cover
        raise RuntimeError("bash is required for shell-wrapper tests")
    resolved = path.resolve(strict=False)
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix()[2:] if resolved.drive else resolved.as_posix()
    if drive:
        return f"/mnt/{drive}{tail}"
    return resolved.as_posix()


def _write_fake_git(
    tmp_path: Path,
    *,
    log_path: Path,
    remote_url: str,
    fetch_head: str,
    clone_head: str = "",
) -> Path:
    fake_git = tmp_path / "git"
    fake_git.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from __future__ import annotations",
                "import json",
                "import os",
                "from pathlib import Path",
                "import sys",
                "",
                "argv = sys.argv[1:]",
                "repo = ''",
                "if len(argv) >= 2 and argv[0] == '-C':",
                "    repo = argv[1]",
                "    argv = argv[2:]",
                "while len(argv) >= 2 and argv[0] == '-c':",
                "    argv = argv[2:]",
                "command = argv[0] if argv else ''",
                f"log_path = Path({str(_wsl_path(log_path))!r})",
                "log_path.parent.mkdir(parents=True, exist_ok=True)",
                "with log_path.open('a', encoding='utf-8') as handle:",
                "    handle.write(json.dumps({'repo': repo, 'argv': argv}, ensure_ascii=True) + '\\n')",
                "repo_path = Path(repo) if repo else None",
                "if command == 'clone':",
                "    target = Path(argv[-1])",
                "    (target / '.git').mkdir(parents=True, exist_ok=True)",
                f"    head = {clone_head!r}.strip()",
                "    if head:",
                "        (target / '.fake_head').write_text(head, encoding='utf-8')",
                "    raise SystemExit(0)",
                "if command == 'remote' and argv[1:3] == ['get-url', 'origin']:",
                f"    print({remote_url!r})",
                "    raise SystemExit(0)",
                "if command == 'status' and argv[1:] == ['--porcelain']:",
                "    dirty = repo_path / '.fake_dirty'",
                "    if dirty.exists():",
                "        print(dirty.read_text(encoding='utf-8'), end='')",
                "    raise SystemExit(0)",
                "if command == 'rev-parse' and argv[1:] == ['HEAD']:",
                "    head = (repo_path / '.fake_head').read_text(encoding='utf-8').strip()",
                "    print(head)",
                "    raise SystemExit(0)",
                "if command == 'fetch':",
                "    (repo_path / '.fake_fetched').write_text('1', encoding='utf-8')",
                "    raise SystemExit(0)",
                "if command == 'checkout' and '--detach' in argv:",
                f"    (repo_path / '.fake_head').write_text({fetch_head!r}, encoding='utf-8')",
                "    raise SystemExit(0)",
                "raise SystemExit(f'unsupported fake git argv: {argv!r}')",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run([BASH_EXE, "-lc", f"chmod +x {_wsl_path(fake_git)}"], check=True)
    return fake_git


def _write_test_repo(
    tmp_path: Path,
    *,
    local_external_root: str | None = "external_checkout/STONE-watermarking",
    external_root: str | None = None,
) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "third_party").mkdir(parents=True, exist_ok=True)
    script_src = REPO_ROOT / "scripts" / "fetch_runtime_upstreams.sh"
    (root / "scripts" / "fetch_runtime_upstreams.sh").write_text(script_src.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    manifest = {
        "repo_url": REMOTE_URL,
        "pinned_commit": PINNED_COMMIT,
    }
    if local_external_root is not None:
        manifest["local_external_root"] = local_external_root
    if external_root is not None:
        manifest["external_root"] = external_root
    (root / "third_party" / "STONE-watermarking.UPSTREAM.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    target_relative = local_external_root or external_root or "external_checkout/STONE-watermarking"
    target = root / Path(target_relative)
    (target / ".git").mkdir(parents=True, exist_ok=True)
    return root, target


def _run_fetch_script(
    tmp_path: Path,
    *,
    head: str,
    local_external_root: str | None = "external_checkout/STONE-watermarking",
    external_root: str | None = None,
    dirty_marker: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if BASH_EXE is None:
        pytest.skip("bash is required for fetch_runtime_upstreams.sh tests")
    root, target = _write_test_repo(
        tmp_path,
        local_external_root=local_external_root,
        external_root=external_root,
    )
    (target / ".fake_head").write_text(head, encoding="utf-8")
    if dirty_marker is not None:
        (target / ".fake_dirty").write_text(dirty_marker, encoding="utf-8")
    log_path = tmp_path / "fake_git.log"
    fake_git = _write_fake_git(
        tmp_path,
        log_path=log_path,
        remote_url=REMOTE_URL,
        fetch_head=PINNED_COMMIT,
    )
    env = os.environ.copy()
    env["PYTHON_BIN"] = "python"
    stdout_path = tmp_path / "fetch.stdout.log"
    stderr_path = tmp_path / "fetch.stderr.log"
    launcher_path = tmp_path / "invoke_fetch.sh"
    launcher_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"export PATH={_wsl_path(tmp_path)}:$PATH",
                f"cd {_wsl_path(root)}",
                f"bash {_wsl_path(root / 'scripts' / 'fetch_runtime_upstreams.sh')} stone_runtime > {_wsl_path(stdout_path)} 2> {_wsl_path(stderr_path)}",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run([BASH_EXE, "-lc", f"chmod +x {_wsl_path(launcher_path)}"], check=True)
    raw = subprocess.run([BASH_EXE, "-lc", f"bash {_wsl_path(launcher_path)}"], env=env, check=False)
    completed = subprocess.CompletedProcess(
        args=list(raw.args),
        returncode=raw.returncode,
        stdout=stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else "",
        stderr=stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else "",
    )
    completed.git_log = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()] if log_path.exists() else []  # type: ignore[attr-defined]
    completed.target = target  # type: ignore[attr-defined]
    return completed


def test_fetch_runtime_upstreams_skips_network_for_existing_clean_pinned_checkout(tmp_path: Path) -> None:
    completed = _run_fetch_script(tmp_path, head=PINNED_COMMIT)

    assert completed.returncode == 0
    assert f"ready: stone_runtime @ {PINNED_COMMIT} (existing checkout)" in completed.stdout
    commands = [entry["argv"][0] for entry in completed.git_log]  # type: ignore[attr-defined]
    assert "rev-parse" in commands
    assert "fetch" not in commands
    assert not ((completed.target / ".fake_fetched").exists())  # type: ignore[attr-defined]


def test_fetch_runtime_upstreams_fetches_when_existing_checkout_head_mismatches(tmp_path: Path) -> None:
    completed = _run_fetch_script(tmp_path, head="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")

    assert completed.returncode == 0
    commands = [entry["argv"][0] for entry in completed.git_log]  # type: ignore[attr-defined]
    assert "rev-parse" in commands
    assert "fetch" in commands
    assert "checkout" in commands
    assert (completed.target / ".fake_fetched").exists()  # type: ignore[attr-defined]
    assert (completed.target / ".fake_head").read_text(encoding="utf-8").strip() == PINNED_COMMIT  # type: ignore[attr-defined]


def test_fetch_runtime_upstreams_honors_external_root_when_local_external_root_is_absent(tmp_path: Path) -> None:
    completed = _run_fetch_script(
        tmp_path,
        head=PINNED_COMMIT,
        local_external_root=None,
        external_root="external_checkout/custom-stone",
    )

    assert completed.returncode == 0
    assert "custom-stone" in completed.stdout
    assert "ready: stone_runtime" in completed.stdout


def test_fetch_runtime_upstreams_rejects_escaped_external_root(tmp_path: Path) -> None:
    completed = _run_fetch_script(
        tmp_path,
        head=PINNED_COMMIT,
        local_external_root=None,
        external_root="../escaped_checkout",
    )

    assert completed.returncode != 0
    assert "must stay under external_checkout/" in completed.stderr


def test_fetch_runtime_upstreams_recreates_dirty_checkout_at_manifest_managed_root(tmp_path: Path) -> None:
    completed = _run_fetch_script(
        tmp_path,
        head="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        local_external_root="external_checkout/custom-stone",
        dirty_marker=" M dirty.txt\n",
    )

    assert completed.returncode == 0
    assert "recreating managed external checkout" in completed.stderr
    commands = [entry["argv"][0] for entry in completed.git_log]  # type: ignore[attr-defined]
    assert "clone" in commands
    assert "fetch" in commands


def test_candidate_roots_uses_external_root_as_local_fallback(monkeypatch, tmp_path: Path) -> None:
    spec = stone_common._method_spec("stone_runtime")
    monkeypatch.setattr(stone_common, "_workspace_root", lambda: tmp_path)
    monkeypatch.setattr(
        stone_common,
        "_load_manifest",
        lambda _method: {
            "checkout_root": spec["checkout_root"],
            "external_root": "external_checkout/custom-stone",
            "public_external_root": spec["public_external_root"],
            "source_relative": ".",
        },
    )

    candidates = stone_common._candidate_roots("stone_runtime")
    external_candidates = [path.relative_to(tmp_path).as_posix() for path, source in candidates if source == "external"]

    assert external_candidates[0] == "external_checkout/custom-stone"
