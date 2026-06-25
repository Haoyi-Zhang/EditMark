from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from run_full_matrix import _load_matrix_runs, _prevalidate_run_metadata
except ImportError:
    spec = importlib.util.spec_from_file_location("run_full_matrix_monitor_fallback", SCRIPTS_DIR / "run_full_matrix.py")
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _load_matrix_runs = module._load_matrix_runs
    _prevalidate_run_metadata = module._prevalidate_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor an active PostEditBench matrix run.")
    parser.add_argument("--matrix-index", type=Path, required=True)
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0.0,
        help="Refresh interval. Use 0 to print once and exit.",
    )
    parser.add_argument(
        "--max-active-runs",
        type=int,
        default=12,
        help="Maximum number of active runs to show in the dashboard.",
    )
    parser.add_argument(
        "--progress-bar-width",
        type=int,
        default=18,
        help="ASCII progress bar width.",
    )
    parser.add_argument(
        "--stale-seconds",
        type=float,
        default=900.0,
        help="Warn when a running matrix index has not been updated for this many seconds.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _lock_snapshot(matrix_index_path: Path) -> dict[str, Any] | None:
    lock_path = matrix_index_path.parent.parent / ".matrix_runner.lock"
    if not lock_path.exists():
        return None
    try:
        payload = _load_json(lock_path)
    except Exception:
        return {"path": str(lock_path), "state": "unreadable"}
    payload = dict(payload)
    payload["path"] = str(lock_path)
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _format_seconds(value: float | None) -> str:
    if value is None or not math.isfinite(value) or value < 0:
        return "n/a"
    seconds = int(round(value))
    days, rem = divmod(seconds, 24 * 60 * 60)
    hours, rem = divmod(rem, 60 * 60)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}m {secs:02d}s"


def _format_percent(value: float) -> str:
    return f"{max(0.0, min(1.0, value)) * 100:.1f}%"


def _bar(fraction: float, width: int) -> str:
    clamped = max(0.0, min(1.0, fraction))
    filled = int(round(clamped * width))
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _load_progress_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = str(record.get("output_dir", "")).strip()
    if not output_dir:
        return None
    progress_path = Path(output_dir)
    if not progress_path.is_absolute():
        progress_path = ROOT / progress_path
    progress_path = progress_path / "progress.json"
    if not progress_path.exists():
        return None
    try:
        return _load_json(progress_path)
    except Exception:
        return None


def _run_progress_fraction(record: dict[str, Any], progress_payload: dict[str, Any] | None) -> float:
    status = str(record.get("status", "")).strip().lower()
    if status in {"success", "skipped", "failed"}:
        return 1.0
    if progress_payload:
        progress_status = str(progress_payload.get("status", "")).strip().lower()
        total_examples = _safe_int(progress_payload.get("total_examples"), 0)
        example_index = _safe_int(progress_payload.get("example_index"), 0)
        if total_examples > 0:
            fraction = max(0.0, min(1.0, example_index / float(total_examples)))
            if progress_status == "completed":
                return min(0.99, fraction or 0.99)
            return fraction
        if progress_status == "completed":
            return 0.99
    return 0.0


def _run_elapsed_seconds(record: dict[str, Any], *, now: float) -> float:
    status = str(record.get("status", "")).strip().lower()
    if status == "running":
        started_at = _safe_float(record.get("started_at"), 0.0)
        if started_at > 0:
            return max(0.0, now - started_at)
    duration = _safe_float(record.get("duration_seconds"), 0.0)
    if duration > 0:
        return duration
    started_at = _safe_float(record.get("started_at"), 0.0)
    finished_at = _safe_float(record.get("finished_at"), 0.0)
    if started_at > 0 and finished_at > started_at:
        return finished_at - started_at
    return 0.0


def _run_weight(record: dict[str, Any]) -> float:
    return max(1.0, _safe_float(record.get("corpus_size"), 1.0))


def _query_gpu_state() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in (completed.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": _safe_int(parts[0], -1),
                "name": parts[1],
                "utilization_gpu": _safe_int(parts[2], 0),
                "memory_used": _safe_int(parts[3], 0),
                "memory_total": _safe_int(parts[4], 0),
            }
        )
    return rows


def _planned_runs(matrix_index_path: Path, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest_value = str(payload.get("manifest", "")).strip()
    if not manifest_value:
        return {}
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    profile = str(payload.get("profile", "")).strip() or matrix_index_path.parent.name
    output_root = matrix_index_path.parent.parent
    try:
        _, runs = _load_matrix_runs(manifest_path, profile, output_root)
        metadata_map, _ = _prevalidate_run_metadata(runs)
    except Exception:
        return {}
    planned: dict[str, dict[str, Any]] = {}
    for run in runs:
        metadata = metadata_map.get(run.run_id, {})
        planned[run.run_id] = {
            "run_id": run.run_id,
            "status": "pending",
            "resource": run.resource,
            "gpu_pool": run.gpu_pool,
            "output_dir": str(run.output_dir),
            "report_path": str(run.report_path),
            "log_path": str(run.log_path),
            "baseline_eval_path": str(run.output_dir / "baseline_eval.json"),
            "priority": int(run.priority),
            "tags": list(run.tags),
            "config_overrides": dict(run.config_overrides or {}),
            "corpus_size": int(metadata.get("corpus_size", 0) or 0),
            "provider_mode": str(metadata.get("provider_mode", "")),
            "watermark_name": str(metadata.get("watermark_name", "")),
            "provider_model": str(metadata.get("provider_model", "")),
            "runtime_model": str(metadata.get("runtime_model", "")),
            "effective_model": str(metadata.get("effective_model", "")),
            "benchmark_label": str(metadata.get("benchmark_label", "")),
            "benchmark_path": str(metadata.get("benchmark_path", "")),
            "cuda_visible_devices": "",
            "started_at": None,
            "finished_at": None,
            "duration_seconds": 0.0,
        }
    return planned


def _merge_runs(planned: dict[str, dict[str, Any]], current_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {run_id: dict(record) for run_id, record in planned.items()}
    for record in current_runs:
        run_id = str(record.get("run_id", "")).strip()
        if not run_id:
            continue
        base = dict(merged.get(run_id, {}))
        base.update(record)
        merged[run_id] = base
    return [merged[key] for key in sorted(merged)]


def _build_dashboard_from_runs(
    *,
    payload: dict[str, Any],
    runs: list[dict[str, Any]],
    progress_by_run: dict[str, dict[str, Any]] | None,
    now: float,
    gpu_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    model_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    weighted_completed = 0.0
    weighted_total = 0.0
    started_times: list[float] = []

    for record in runs:
        run_id = str(record.get("run_id", "")).strip()
        progress_payload = None
        if progress_by_run is not None:
            progress_payload = dict(progress_by_run.get(run_id, {})) if run_id else None
        else:
            progress_payload = _load_progress_payload(record)
        progress_fraction = _run_progress_fraction(record, progress_payload)
        weight = _run_weight(record)
        weighted_completed += weight * progress_fraction
        weighted_total += weight
        started_at = _safe_float(record.get("started_at"), 0.0)
        if started_at > 0:
            started_times.append(started_at)
        model_name = str(record.get("effective_model", "")).strip() or "unspecified_model"
        stage = ""
        latest = ""
        if progress_payload:
            stage = str(progress_payload.get("stage", "")).strip() or str(progress_payload.get("current_stage", "")).strip()
            latest_example = str(progress_payload.get("latest_example_id", "")).strip()
            latest_attack = str(progress_payload.get("latest_attack", "")).strip()
            if latest_example and latest_attack:
                latest = f"{latest_example} / {latest_attack}"
            elif latest_example:
                latest = latest_example
            elif latest_attack:
                latest = latest_attack
        row = {
            **record,
            "model_name": model_name,
            "method_name": str(record.get("watermark_name", "")).strip() or "unknown_method",
            "benchmark_name": str(record.get("benchmark_label", "")).strip() or "unknown_benchmark",
            "progress_fraction": progress_fraction,
            "progress_payload": progress_payload,
            "elapsed_seconds": _run_elapsed_seconds(record, now=now),
            "stage_label": stage,
            "latest_label": latest,
            "weight": weight,
        }
        run_rows.append(row)
        model_rows[model_name].append(row)

    progress_fraction = (weighted_completed / weighted_total) if weighted_total > 0 else 0.0
    elapsed_seconds = max(0.0, now - min(started_times)) if started_times else 0.0
    eta_seconds: float | None = None
    completed_runs = sum(1 for row in run_rows if row["status"] in {"success", "skipped"})
    if elapsed_seconds > 0 and 0.02 <= progress_fraction < 1.0 and completed_runs >= 2:
        remaining_fraction = 1.0 - progress_fraction
        eta_seconds = elapsed_seconds * remaining_fraction / max(progress_fraction, 1e-9)

    models: list[dict[str, Any]] = []
    completed_models: list[str] = []
    active_models: list[str] = []
    for model_name, records in sorted(model_rows.items()):
        success = sum(1 for row in records if row["status"] == "success")
        skipped = sum(1 for row in records if row["status"] == "skipped")
        failed = sum(1 for row in records if row["status"] == "failed")
        running = sum(1 for row in records if row["status"] == "running")
        pending = sum(1 for row in records if row["status"] not in {"success", "skipped", "failed", "running"})
        model_weight_total = sum(row["weight"] for row in records)
        model_weight_done = sum(row["weight"] * row["progress_fraction"] for row in records)
        model_fraction = (model_weight_done / model_weight_total) if model_weight_total > 0 else 0.0
        is_completed = running == 0 and pending == 0 and failed == 0
        if running > 0:
            active_models.append(model_name)
        if is_completed:
            completed_models.append(model_name)
        models.append(
            {
                "model_name": model_name,
                "run_count": len(records),
                "success_count": success,
                "skipped_count": skipped,
                "failed_count": failed,
                "running_count": running,
                "pending_count": pending,
                "progress_fraction": model_fraction,
                "status": (
                    "completed"
                    if is_completed
                    else "running"
                    if running > 0
                    else "failed"
                    if failed > 0
                    else "pending"
                    if pending > 0
                    else "failed"
                ),
            }
        )

    active_runs = sorted(
        [row for row in run_rows if row["status"] == "running"],
        key=lambda row: (row["elapsed_seconds"], row.get("priority", 0), row["run_id"]),
        reverse=True,
    )
    recent_failures = sorted(
        [row for row in run_rows if row["status"] == "failed"],
        key=lambda row: _safe_float(row.get("finished_at"), 0.0),
        reverse=True,
    )
    longest_tail = active_runs[0] if active_runs else None

    return {
        "matrix_index_path": str(payload.get("_matrix_index_path", "")),
        "profile": str(payload.get("profile", "")).strip() or "unknown_profile",
        "updated_at": _safe_float(payload.get("updated_at"), now),
        "overall": {
            "run_count": max(len(runs), _safe_int(payload.get("run_count"), len(runs))),
            "success_count": _safe_int(payload.get("success_count"), sum(1 for row in run_rows if row["status"] == "success")),
            "skipped_count": _safe_int(payload.get("skipped_count"), sum(1 for row in run_rows if row["status"] == "skipped")),
            "running_count": _safe_int(payload.get("running_count"), sum(1 for row in run_rows if row["status"] == "running")),
            "failed_count": _safe_int(payload.get("failed_count"), sum(1 for row in run_rows if row["status"] == "failed")),
            "pending_count": _safe_int(payload.get("pending_count"), sum(1 for row in run_rows if row["status"] not in {"success", "skipped", "failed", "running"})),
            "progress_fraction": progress_fraction,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
        },
        "models": models,
        "active_models": active_models,
        "completed_models": completed_models,
        "active_runs": active_runs,
        "recent_failures": recent_failures,
        "longest_tail": longest_tail,
        "gpu_rows": list(gpu_rows if gpu_rows is not None else _query_gpu_state()),
    }


def build_dashboard_data(
    matrix_index_path: Path,
    *,
    now: float | None = None,
    gpu_rows: list[dict[str, Any]] | None = None,
    stale_seconds: float = 900.0,
) -> dict[str, Any]:
    now_ts = float(now if now is not None else time.time())
    payload = _load_json(matrix_index_path)
    planned = _planned_runs(matrix_index_path, payload)
    runs = _merge_runs(planned, list(payload.get("runs", [])))
    payload = {**payload, "_matrix_index_path": str(matrix_index_path)}
    dashboard = _build_dashboard_from_runs(
        payload=payload,
        runs=runs,
        progress_by_run=None,
        now=now_ts,
        gpu_rows=gpu_rows,
    )
    updated_at = _safe_float(payload.get("updated_at"), now_ts)
    age_seconds = max(0.0, now_ts - updated_at)
    running_count = int(dashboard.get("overall", {}).get("running_count", 0) or 0)
    dashboard["monitor"] = {
        "updated_age_seconds": age_seconds,
        "stale_threshold_seconds": max(0.0, float(stale_seconds)),
        "stale": running_count > 0 and age_seconds >= max(0.0, float(stale_seconds)),
        "lock": _lock_snapshot(matrix_index_path),
    }
    return dashboard


def load_snapshot(
    matrix_index_path: Path,
    *,
    now: float | None = None,
    gpu_rows: list[dict[str, Any]] | None = None,
    stale_seconds: float = 900.0,
) -> dict[str, Any]:
    """Backward-compatible alias for callers/tests expecting snapshot data."""
    return build_dashboard_data(matrix_index_path, now=now, gpu_rows=gpu_rows, stale_seconds=stale_seconds)


def render_dashboard(data: dict[str, Any], *, progress_bar_width: int = 18, max_active_runs: int = 12) -> str:
    overall = dict(data.get("overall", {}))
    monitor = dict(data.get("monitor", {}))
    lines = [
        f"PostEditBench Monitor  profile={data.get('profile', 'unknown_profile')}",
        f"matrix_index={data.get('matrix_index_path', '')}",
        (
            "overall  "
            f"{_bar(float(overall.get('progress_fraction', 0.0)), progress_bar_width)} "
            f"{_format_percent(float(overall.get('progress_fraction', 0.0)))}  "
            f"success={overall.get('success_count', 0)} "
            f"running={overall.get('running_count', 0)} "
            f"failed={overall.get('failed_count', 0)} "
            f"pending={overall.get('pending_count', 0)} "
            f"elapsed={_format_seconds(_safe_float(overall.get('elapsed_seconds'), 0.0))} "
            f"eta={_format_seconds(_safe_float(overall.get('eta_seconds'), float('nan')))}"
        ),
    ]
    if monitor.get("stale"):
        lock = dict(monitor.get("lock") or {})
        owner_host = str(lock.get("host", "")).strip() or "unknown_host"
        owner_pid = str(lock.get("pid", "")).strip() or "unknown_pid"
        lines.append(
            "warning: matrix index looks stale; "
            f"last_update_age={_format_seconds(_safe_float(monitor.get('updated_age_seconds'), 0.0))} "
            f"lock_owner={owner_host}:{owner_pid}"
        )
    longest_tail = data.get("longest_tail")
    if longest_tail:
        lines.append(
            "longest_tail="
            f"{longest_tail.get('run_id')} "
            f"model={longest_tail.get('model_name')} "
            f"benchmark={longest_tail.get('benchmark_name')} "
            f"method={longest_tail.get('method_name')} "
            f"elapsed={_format_seconds(_safe_float(longest_tail.get('elapsed_seconds'), 0.0))}"
        )

    lines.append("")
    lines.append("Active models")
    active_lines: list[str] = []
    for model in data.get("models", []):
        line = (
            "  - "
            f"{_bar(float(model.get('progress_fraction', 0.0)), progress_bar_width)} "
            f"{_format_percent(float(model.get('progress_fraction', 0.0)))} "
            f"{model.get('model_name')} "
            f"[status={model.get('status')} success={model.get('success_count')} "
            f"running={model.get('running_count')} pending={model.get('pending_count')} failed={model.get('failed_count')}]"
        )
        if str(model.get("status", "")) != "completed":
            active_lines.append(line)
    lines.extend(active_lines or ["  - none"])

    lines.append("")
    lines.append("Running runs")
    if not data.get("active_runs"):
        lines.append("  - none")
    else:
        for row in list(data.get("active_runs", []))[: max(1, int(max_active_runs))]:
            progress_payload = row.get("progress_payload") or {}
            example_index = _safe_int(progress_payload.get("example_index"), 0)
            total_examples = _safe_int(progress_payload.get("total_examples"), 0)
            example_progress = f"{example_index}/{total_examples}" if total_examples > 0 else "n/a"
            lines.append(
                "  - "
                f"{row.get('run_id')} "
                f"[gpu={row.get('cuda_visible_devices', '') or '-'}] "
                f"{row.get('model_name')} | {row.get('method_name')} | {row.get('benchmark_name')} "
                f"| {_bar(float(row.get('progress_fraction', 0.0)), max(8, progress_bar_width // 2))} "
                f"| progress={_format_percent(float(row.get('progress_fraction', 0.0)))} "
                f"| examples={example_progress} "
                f"| stage={row.get('stage_label', '') or 'n/a'} "
                f"| attack={str(progress_payload.get('latest_attack', '')).strip() or 'n/a'} "
                f"| latest={row.get('latest_label', '') or 'n/a'} "
                f"| elapsed={_format_seconds(_safe_float(row.get('elapsed_seconds'), 0.0))}"
            )

    lines.append("")
    lines.append("Completed models")
    completed_models = list(data.get("completed_models", []))
    if completed_models:
        lines.extend(f"  - {model_name}" for model_name in completed_models)
    else:
        lines.append("  - none")

    lines.append("")
    lines.append("Recent failures")
    recent_failures = list(data.get("recent_failures", []))
    if recent_failures:
        for row in recent_failures[:5]:
            reason = str(row.get("reason", "")).strip() or "unknown"
            reason_detail = str(row.get("reason_detail", "")).strip()
            error = str(row.get("error", "")).strip() or "n/a"
            log_path = str(row.get("log_path", "")).strip()
            message = (
                "  - "
                f"{row.get('run_id')} "
                f"{row.get('model_name')} | {row.get('method_name')} | {row.get('benchmark_name')} "
                f"| reason={reason}"
            )
            if reason_detail:
                message += f" | detail={reason_detail[:180]}"
            if error and error != "n/a":
                message += f" | error={error[:180]}"
            if log_path:
                message += f" | log={log_path}"
            lines.append(message)
    else:
        lines.append("  - none")

    lines.append("")
    lines.append("GPU state")
    gpu_rows = list(data.get("gpu_rows", []))
    if not gpu_rows:
        lines.append("  - nvidia-smi unavailable")
    else:
        for row in gpu_rows:
            lines.append(
                "  - "
                f"gpu{row.get('index')} {row.get('name')} "
                f"util={row.get('utilization_gpu', 0)}% "
                f"mem={row.get('memory_used', 0)}/{row.get('memory_total', 0)} MiB"
            )
    return "\n".join(lines) + "\n"


def _render_dashboard(
    index_payload: dict[str, Any],
    runs: list[dict[str, Any]],
    progress_by_run: dict[str, dict[str, Any]],
    gpu_rows: list[dict[str, Any]],
    *,
    now: float | None = None,
    progress_bar_width: int = 18,
    max_active_runs: int = 12,
) -> str:
    now_ts = float(now if now is not None else time.time())
    payload = dict(index_payload)
    payload["_matrix_index_path"] = str(payload.get("_matrix_index_path", "synthetic://matrix_index.json"))
    dashboard = _build_dashboard_from_runs(
        payload=payload,
        runs=[dict(run) for run in runs],
        progress_by_run=progress_by_run,
        now=now_ts,
        gpu_rows=gpu_rows,
    )
    return render_dashboard(
        dashboard,
        progress_bar_width=progress_bar_width,
        max_active_runs=max_active_runs,
    )


def render_snapshot(
    matrix_index_path: Path,
    *,
    gpu_rows: list[dict[str, Any]] | None = None,
    now: float | None = None,
    progress_bar_width: int = 18,
    max_active_runs: int = 12,
    stale_seconds: float = 900.0,
) -> str:
    return render_dashboard(
        build_dashboard_data(matrix_index_path, now=now, gpu_rows=gpu_rows, stale_seconds=stale_seconds),
        progress_bar_width=progress_bar_width,
        max_active_runs=max_active_runs,
    )


def _clear_screen() -> None:
    print("\033[2J\033[H", end="", flush=True)


def main() -> int:
    args = parse_args()
    matrix_index_path = args.matrix_index.resolve()
    watch_seconds = max(0.0, float(args.watch_seconds))
    while True:
        try:
            output = render_snapshot(
                matrix_index_path,
                progress_bar_width=max(8, int(args.progress_bar_width)),
                max_active_runs=max(1, int(args.max_active_runs)),
                stale_seconds=max(0.0, float(args.stale_seconds)),
            )
        except FileNotFoundError:
            output = f"matrix index not found: {matrix_index_path}\n"
        except Exception as exc:  # pragma: no cover - operator-facing fallback
            output = f"monitor error: {type(exc).__name__}: {exc}\n"
        if watch_seconds > 0:
            _clear_screen()
        print(output, end="", flush=True)
        if watch_seconds <= 0:
            return 0
        time.sleep(watch_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
