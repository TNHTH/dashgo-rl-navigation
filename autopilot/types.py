from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def iso_now() -> str:
    """返回 UTC ISO 时间戳。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


@dataclass
class JsonModel:
    """为 dataclass 提供稳定的 JSON 导出。"""

    def to_dict(self) -> dict[str, Any]:
        return _normalize(asdict(self))


@dataclass
class RunLayout(JsonModel):
    project_root: Path
    autopilot_root: Path
    runs_root: Path
    metrics_root: Path
    lineage_file: Path
    generation_root: Path
    run_root: Path
    checkpoints_dir: Path
    artifacts_dir: Path
    tensorboard_dir: Path
    metrics_dir: Path


@dataclass
class ScalarPoint(JsonModel):
    step: int
    value: float
    wall_time: float


@dataclass
class ScalarSeries(JsonModel):
    tag: str
    source_files: list[str] = field(default_factory=list)
    points: list[ScalarPoint] = field(default_factory=list)

    @property
    def latest(self) -> ScalarPoint | None:
        return self.points[-1] if self.points else None


@dataclass
class EvalMetrics(JsonModel):
    success_rate: float = 0.0
    collision_rate: float = 0.0
    timeout_rate: float = 0.0
    mean_steps: float = 0.0
    reverse_case_success_rate: float = 0.0
    spin_proxy_rate: float = 0.0
    progress_stall_rate: float = 0.0
    high_clip_ratio: float = 0.0
    path_efficiency: float = 0.0
    net_progress_ratio: float = 0.0
    orbit_score: float = 0.0
    near_obstacle_dwell: float = 0.0
    sensor_health_score: float = 0.0
    log_anomaly_count: float = 0.0
    score: float = 0.0
    total_episodes: int = 0
    completed_episodes: int = 0


@dataclass
class EvalRequest(JsonModel):
    checkpoint: Path
    suite: str
    project_root: Path
    notes: list[str] = field(default_factory=list)
    requested_episodes: int | None = None
    created_at: str = field(default_factory=iso_now)


@dataclass
class EvalResult(JsonModel):
    status: str
    request: EvalRequest
    metrics: EvalMetrics | None = None
    scenes: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=iso_now)


@dataclass
class DoctorCheck(JsonModel):
    name: str
    status: str
    message: str
    severity: str = "info"
    source: str = "preflight"
    evidence_paths: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DoctorResult(JsonModel):
    status: str
    checks: list[DoctorCheck] = field(default_factory=list)
    recommended_action: str = "continue"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=iso_now)


@dataclass
class AnomalyReport(JsonModel):
    anomaly_type: str
    severity: str
    suspected_layer: str
    trigger_run_name: str | None = None
    message: str = ""
    evidence_paths: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    recommended_job: str = "debug_job"
    created_at: str = field(default_factory=iso_now)


@dataclass
class CodexRouteDecision(JsonModel):
    job_type: str
    route_tier: str
    requested_profile: str
    requested_model: str
    effective_model: str
    requested_reasoning_effort: str
    effective_reasoning_effort: str
    resolution_mode: str = "exact"
    fallback_reason: str = ""
    catalog_source: str = ""
    created_at: str = field(default_factory=iso_now)


@dataclass
class CodexJobSpec(JsonModel):
    job_type: str
    prompt: str
    project_root: Path
    allowed_paths: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    expected_artifacts: list[str] = field(default_factory=list)
    route: CodexRouteDecision | None = None
    created_at: str = field(default_factory=iso_now)


@dataclass
class LineageRecord(JsonModel):
    record_id: str
    generation: str
    run_name: str
    run_dir: Path
    checkpoint_path: Path
    checkpoint_iteration: int | None = None
    seed: int | None = None
    stage: str | None = None
    parent_checkpoint: str | None = None
    warm_start_source: str | None = None
    metrics_file: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=iso_now)
