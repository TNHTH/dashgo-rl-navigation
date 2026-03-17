from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Iterable

from .io_utils import ensure_dir, read_json, write_json
from .types import LineageRecord, RunLayout


LINEAGE_SCHEMA_VERSION = 1


def resolve_project_root(start: str | Path | None = None) -> Path:
    if start is None:
        return Path(__file__).resolve().parent.parent
    path = Path(start).resolve()
    if path.is_file():
        return path.parent
    return path


def default_autopilot_root(project_root: str | Path | None = None) -> Path:
    return resolve_project_root(project_root) / "autopilot"


def default_lineage_file(project_root: str | Path | None = None) -> Path:
    return default_autopilot_root(project_root) / "lineage.json"


def sanitize_name(raw: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip())
    normalized = normalized.strip("._-")
    return normalized or "run"


def build_run_layout(
    *,
    project_root: str | Path | None = None,
    generation: str,
    run_name: str,
    timestamp: datetime | None = None,
    create: bool = True,
) -> RunLayout:
    project = resolve_project_root(project_root)
    autopilot_root = default_autopilot_root(project)
    runs_root = autopilot_root / "runs"
    metrics_root = autopilot_root / "metrics"
    generation_root = runs_root / sanitize_name(generation)
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    run_root = generation_root / f"{stamp}_{sanitize_name(run_name)}"
    checkpoints_dir = run_root / "checkpoints"
    artifacts_dir = run_root / "artifacts"
    tensorboard_dir = run_root / "tensorboard"
    metrics_dir = run_root / "metrics"
    layout = RunLayout(
        project_root=project,
        autopilot_root=autopilot_root,
        runs_root=runs_root,
        metrics_root=metrics_root,
        lineage_file=default_lineage_file(project),
        generation_root=generation_root,
        run_root=run_root,
        checkpoints_dir=checkpoints_dir,
        artifacts_dir=artifacts_dir,
        tensorboard_dir=tensorboard_dir,
        metrics_dir=metrics_dir,
    )
    if create:
        for path in (
            autopilot_root,
            runs_root,
            metrics_root,
            generation_root,
            run_root,
            checkpoints_dir,
            artifacts_dir,
            tensorboard_dir,
            metrics_dir,
        ):
            ensure_dir(path)
        bootstrap_lineage_file(layout.lineage_file)
    return layout


def bootstrap_lineage_file(path: str | Path) -> Path:
    target = Path(path)
    if not target.exists():
        write_json(
            target,
            {"schema_version": LINEAGE_SCHEMA_VERSION, "records": []},
        )
    return target


def load_lineage_records(path: str | Path | None = None) -> list[LineageRecord]:
    target = Path(path) if path is not None else default_lineage_file()
    payload = read_json(target, default={"schema_version": LINEAGE_SCHEMA_VERSION, "records": []})
    records = payload.get("records", []) if isinstance(payload, dict) else []
    loaded: list[LineageRecord] = []
    for item in records:
        loaded.append(
            LineageRecord(
                record_id=item["record_id"],
                generation=item["generation"],
                run_name=item["run_name"],
                run_dir=Path(item["run_dir"]),
                checkpoint_path=Path(item["checkpoint_path"]),
                checkpoint_iteration=item.get("checkpoint_iteration"),
                seed=item.get("seed"),
                stage=item.get("stage"),
                parent_checkpoint=item.get("parent_checkpoint"),
                warm_start_source=item.get("warm_start_source"),
                metrics_file=item.get("metrics_file"),
                tags=list(item.get("tags", [])),
                notes=list(item.get("notes", [])),
                created_at=item.get("created_at"),
            )
        )
    return loaded


def save_lineage_records(path: str | Path, records: Iterable[LineageRecord]) -> Path:
    target = Path(path)
    payload = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "records": [record.to_dict() for record in records],
    }
    return write_json(target, payload)


def append_lineage_record(record: LineageRecord, path: str | Path | None = None) -> Path:
    target = Path(path) if path is not None else default_lineage_file()
    bootstrap_lineage_file(target)
    records = load_lineage_records(target)
    records.append(record)
    return save_lineage_records(target, records)
