from __future__ import annotations

import subprocess

from posteditbench import toolchains


class _Completed:
    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_inspect_local_toolchain_accepts_minimum_supported_host_versions(monkeypatch) -> None:
    def fake_which(name: str) -> str:
        return f"/usr/bin/{name}"

    def fake_run(command, **kwargs):
        executable = command[0]
        if executable.endswith("node"):
            return _Completed(stdout="v12.22.9\n")
        raise AssertionError(f"unexpected command: {command}")

    toolchains.inspect_local_toolchain.cache_clear()
    monkeypatch.setattr(toolchains.shutil, "which", fake_which)
    monkeypatch.setattr(toolchains.subprocess, "run", fake_run)

    inspection = toolchains.inspect_local_toolchain("javascript")

    assert inspection["status"] == "ok"
    assert inspection["verified"] is True
    assert inspection["issues"] == []
    assert inspection["tools"][0]["minimum_version"] == "12.22"
    assert inspection["tools"][0]["recommended_version"] == "20"
    assert inspection["tools"][0]["recommended_match"] is False


def test_inspect_local_toolchain_rejects_versions_below_minimum(monkeypatch) -> None:
    def fake_which(name: str) -> str:
        return f"/usr/bin/{name}"

    def fake_run(command, **kwargs):
        executable = command[0]
        if executable.endswith("go"):
            return _Completed(stdout="go version go1.17.13 linux/amd64\n")
        raise AssertionError(f"unexpected command: {command}")

    toolchains.inspect_local_toolchain.cache_clear()
    monkeypatch.setattr(toolchains.shutil, "which", fake_which)
    monkeypatch.setattr(toolchains.subprocess, "run", fake_run)

    inspection = toolchains.inspect_local_toolchain("go")

    assert inspection["status"] == "failed"
    assert inspection["verified"] is False
    assert any("tool_version_mismatch:go:1.17.13<1.18" in issue for issue in inspection["issues"])
