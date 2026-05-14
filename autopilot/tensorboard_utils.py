from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .types import ScalarPoint, ScalarSeries

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:  # pragma: no cover - 运行环境缺 tensorboard 时走回退
    EventAccumulator = None  # type: ignore[assignment]


def is_tensorboard_available() -> bool:
    return EventAccumulator is not None


def find_event_files(log_dir: str | Path) -> list[Path]:
    root = Path(log_dir)
    if not root.exists():
        return []
    return sorted(root.rglob("events.out.tfevents.*"))


def read_scalar_series(
    log_dir: str | Path,
    tags: Sequence[str],
    *,
    size_guidance: dict[str, int] | None = None,
) -> dict[str, ScalarSeries]:
    if EventAccumulator is None:
        raise RuntimeError("当前 Python 环境不可用 tensorboard.backend.event_processing.event_accumulator")
    event_files = find_event_files(log_dir)
    if not event_files:
        return {tag: ScalarSeries(tag=tag, source_files=[]) for tag in tags}

    accumulator = EventAccumulator(
        str(Path(log_dir)),
        size_guidance=size_guidance or {"scalars": 0},
    )
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    result: dict[str, ScalarSeries] = {}
    for tag in tags:
        if tag not in available:
            result[tag] = ScalarSeries(tag=tag, source_files=[str(path) for path in event_files])
            continue
        scalars = accumulator.Scalars(tag)
        result[tag] = ScalarSeries(
            tag=tag,
            source_files=[str(path) for path in event_files],
            points=[
                ScalarPoint(step=point.step, value=point.value, wall_time=point.wall_time)
                for point in scalars
            ],
        )
    return result


def summarize_latest_scalars(
    log_dir: str | Path,
    tags: Sequence[str],
) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for tag, series in read_scalar_series(log_dir, tags).items():
        latest = series.latest
        summary[tag] = latest.value if latest is not None else None
    return summary
