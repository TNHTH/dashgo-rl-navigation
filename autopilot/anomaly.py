from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import math
import re

from .types import DoctorCheck, DoctorResult, EvalMetrics


SCALAR_PREFILTER_TAGS = (
    "Curriculum/target_adaptive",
    "Episode_Termination/reach_goal",
    "Episode_Termination/object_collision",
    "Episode_Termination/time_out",
    "Metrics/target_pose/position_error",
)


def _status_rank(status: str) -> int:
    return {
        "ok": 0,
        "warning": 1,
        "soft_fail": 2,
        "failed": 3,
        "hard_fail": 4,
    }.get(status, 0)


def _severity_rank(severity: str) -> int:
    return {
        "info": 0,
        "warning": 1,
        "soft_fail": 2,
        "hard_fail": 3,
    }.get(severity, 0)


def _result_status(checks: list[DoctorCheck]) -> str:
    if any(check.status in {"failed", "hard_fail"} for check in checks):
        return "failed"
    if any(check.status == "soft_fail" for check in checks):
        return "soft_fail"
    if any(check.status == "warning" for check in checks):
        return "warning"
    return "ok"


def _recommended_action(checks: list[DoctorCheck]) -> str:
    if any(check.severity == "hard_fail" for check in checks):
        return "codex_debug"
    if any(check.status == "soft_fail" for check in checks):
        return "rollback"
    if any(check.status == "warning" for check in checks):
        return "retry"
    return "continue"


def make_check(
    *,
    name: str,
    status: str,
    message: str,
    severity: str,
    source: str,
    details: dict | None = None,
    evidence_paths: list[str] | None = None,
) -> DoctorCheck:
    return DoctorCheck(
        name=name,
        status=status,
        message=message,
        severity=severity,
        source=source,
        details=details or {},
        evidence_paths=evidence_paths or [],
    )


def merge_doctor_results(*results: DoctorResult) -> DoctorResult:
    checks: list[DoctorCheck] = []
    metadata: dict = {}
    for result in results:
        checks.extend(result.checks)
        metadata.update(result.metadata)
    return DoctorResult(
        status=_result_status(checks),
        checks=checks,
        recommended_action=_recommended_action(checks),
        metadata=metadata,
    )


def build_doctor_result(checks: list[DoctorCheck], *, metadata: dict | None = None) -> DoctorResult:
    return DoctorResult(
        status=_result_status(checks),
        checks=checks,
        recommended_action=_recommended_action(checks),
        metadata=metadata or {},
    )


def prefilter_training_summary(summary: dict, *, base_curriculum: float = 3.75) -> bool:
    scalars = summary.get("latest_scalars", {}) or {}
    curriculum = scalars.get("Curriculum/target_adaptive")
    reach_goal = scalars.get("Episode_Termination/reach_goal")
    collision = scalars.get("Episode_Termination/object_collision")
    timeout = scalars.get("Episode_Termination/time_out")
    position_error = scalars.get("Metrics/target_pose/position_error")
    return (
        curriculum is not None
        and curriculum >= base_curriculum
        and reach_goal is not None
        and reach_goal >= 0.95
        and collision == 0.0
        and timeout == 0.0
        and position_error is not None
        and position_error <= 0.40
    )


def analyze_log_text(
    log_text: str,
    *,
    log_path: str | None = None,
    summary: dict | None = None,
) -> DoctorResult:
    checks: list[DoctorCheck] = []
    evidence = [log_path] if log_path else []

    patterns = [
        (
            "python_traceback",
            r"Traceback \(most recent call last\):",
            "训练日志出现 Python traceback",
            "hard_fail",
            "trainer",
        ),
        (
            "key_error",
            r"\bKeyError\b",
            "训练日志出现 KeyError，疑似观测/配置合同回归",
            "hard_fail",
            "observation_contract",
        ),
        (
            "runtime_error",
            r"\bRuntimeError\b",
            "训练日志出现 RuntimeError",
            "hard_fail",
            "trainer",
        ),
        (
            "shape_mismatch",
            r"(shape mismatch|size mismatch|mat1 and mat2 shapes cannot be multiplied)",
            "训练日志出现维度或张量形状不匹配",
            "hard_fail",
            "observation_contract",
        ),
        (
            "camera_disabled",
            r"(enable_cameras.*缺失|No cameras found|camera_.*not found)",
            "训练日志显示相机未启用或相机实体缺失",
            "hard_fail",
            "sensor",
        ),
        (
            "nan_detected",
            r"(\bnan\b|NaN|non-finite|inf detected)",
            "训练日志出现 NaN/Inf 迹象",
            "soft_fail",
            "trainer",
        ),
    ]

    for name, pattern, message, severity, layer in patterns:
        match = re.search(pattern, log_text, flags=re.IGNORECASE)
        if not match:
            continue
        checks.append(
            make_check(
                name=name,
                status="failed" if severity == "hard_fail" else "soft_fail",
                message=message,
                severity=severity,
                source="log",
                details={"match": match.group(0), "suspected_layer": layer},
                evidence_paths=evidence,
            )
        )

    if summary is not None:
        latest_scalars = summary.get("latest_scalars", {}) or {}
        null_tags = [tag for tag in SCALAR_PREFILTER_TAGS if latest_scalars.get(tag) is None]
        if len(null_tags) == len(SCALAR_PREFILTER_TAGS):
            checks.append(
                make_check(
                    name="all_core_scalars_missing",
                    status="soft_fail",
                    message="核心训练标量全部缺失，训练可能异常启动或日志链已断流",
                    severity="soft_fail",
                    source="runtime",
                    details={"missing_tags": null_tags},
                    evidence_paths=evidence,
                )
            )
        seconds_since_update = float(summary.get("seconds_since_update") or 0.0)
        if seconds_since_update > 180.0:
            checks.append(
                make_check(
                    name="stale_run_update",
                    status="warning",
                    message="训练长时间没有新 checkpoint 或 run_meta 更新",
                    severity="warning",
                    source="runtime",
                    details={"seconds_since_update": seconds_since_update},
                    evidence_paths=evidence,
                )
            )

    if not checks:
        checks.append(
            make_check(
                name="runtime_log_ok",
                status="ok",
                message="未从日志中检测到已知硬错误模式",
                severity="info",
                source="log",
                evidence_paths=evidence,
            )
        )

    return build_doctor_result(checks, metadata={"log_path": log_path})


def analyze_live_sensor_payload(payload: dict, *, evidence_path: str | None = None) -> DoctorResult:
    checks: list[DoctorCheck] = []
    evidence = [evidence_path] if evidence_path else []
    samples = payload.get("samples", []) or []
    lidar_normalized = bool(payload.get("lidar_normalized", False))
    lidar_max_range_m = payload.get("lidar_max_range_m")
    lidar_scale = 1.0
    if isinstance(lidar_max_range_m, (int, float)) and float(lidar_max_range_m) > 0.0:
        if lidar_normalized:
            lidar_scale = float(lidar_max_range_m)

    if not samples:
        checks.append(
            make_check(
                name="live_probe_missing",
                status="failed",
                message="未拿到活体传感器采样数据",
                severity="hard_fail",
                source="preflight",
                evidence_paths=evidence,
            )
        )
        return build_doctor_result(checks, metadata={"profile": payload.get("profile")})

    def _all_stats(key: str, field: str) -> list[float]:
        values: list[float] = []
        for sample in samples:
            stats = sample.get(key) or {}
            value = stats.get(field)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    lidar_means = _all_stats("lidar", "mean")
    lidar_mins = _all_stats("lidar", "min")
    min_obstacle_means = _all_stats("min_obstacle_distance", "mean")

    if not lidar_means or not lidar_mins or not min_obstacle_means:
        checks.append(
            make_check(
                name="live_probe_incomplete",
                status="failed",
                message="活体传感器采样缺少关键字段",
                severity="hard_fail",
                source="preflight",
                evidence_paths=evidence,
            )
        )
        return build_doctor_result(checks, metadata={"profile": payload.get("profile")})

    if any(math.isnan(value) or math.isinf(value) for value in lidar_means + lidar_mins + min_obstacle_means):
        checks.append(
            make_check(
                name="sensor_non_finite",
                status="failed",
                message="活体传感器采样出现 NaN/Inf",
                severity="hard_fail",
                source="preflight",
                evidence_paths=evidence,
            )
        )

    if max(lidar_means) - min(lidar_means) < 1.0e-6 and max(lidar_mins) - min(lidar_mins) < 1.0e-6:
        checks.append(
            make_check(
                name="stitched_lidar_flatline",
                status="failed",
                message="stitched lidar 在多个采样步几乎完全不变，疑似传感器死数据",
                severity="hard_fail",
                source="preflight",
                evidence_paths=evidence,
            )
        )

    mismatch_count = 0
    for sample in samples:
        lidar_stats = sample.get("lidar") or {}
        min_obstacle_stats = sample.get("min_obstacle_distance") or {}
        lidar_min = lidar_stats.get("min")
        obstacle_min = min_obstacle_stats.get("min")
        if not isinstance(lidar_min, (int, float)) or not isinstance(obstacle_min, (int, float)):
            continue
        effective_lidar_min = float(lidar_min) * lidar_scale
        if abs(effective_lidar_min - float(obstacle_min)) > 0.25:
            mismatch_count += 1
    if samples and mismatch_count / len(samples) > 0.20:
        checks.append(
            make_check(
                name="sensor_contract_mismatch",
                status="soft_fail",
                message="最小障碍距离与 stitched lidar 最近值长期不一致，疑似观测合同漂移",
                severity="soft_fail",
                source="preflight",
                details={
                    "mismatch_ratio": mismatch_count / len(samples),
                    "lidar_normalized": lidar_normalized,
                    "lidar_scale": lidar_scale,
                },
                evidence_paths=evidence,
            )
        )

    if not checks:
        checks.append(
            make_check(
                name="live_sensor_ok",
                status="ok",
                message="活体传感器探针未发现异常",
                severity="info",
                source="preflight",
                evidence_paths=evidence,
            )
        )

    return build_doctor_result(checks, metadata={"profile": payload.get("profile"), "samples": len(samples)})


def summarize_eval_episodes(episodes: list[dict], *, suite: str, log_anomaly_count: int = 0) -> EvalMetrics:
    total = len(episodes)
    if total == 0:
        return EvalMetrics(log_anomaly_count=float(log_anomaly_count))

    successes = sum(1 for item in episodes if item.get("termination_reason") == "reach_goal")
    collisions = sum(1 for item in episodes if item.get("termination_reason") == "object_collision")
    timeouts = sum(1 for item in episodes if item.get("termination_reason") == "time_out")
    reverse_successes = sum(1 for item in episodes if item.get("reverse_case") and item.get("termination_reason") == "reach_goal")

    mean_steps = sum(float(item.get("steps", 0.0)) for item in episodes) / total
    spin_proxy_rate = sum(float(item.get("spin_proxy_ratio", 0.0)) for item in episodes) / total
    progress_stall_rate = sum(1.0 for item in episodes if item.get("progress_stall")) / total
    high_clip_ratio = sum(float(item.get("high_clip_ratio", 0.0)) for item in episodes) / total
    path_efficiency = sum(float(item.get("path_efficiency", 0.0)) for item in episodes) / total
    net_progress_ratio = sum(float(item.get("net_progress_ratio", 0.0)) for item in episodes) / total
    orbit_score = sum(1.0 for item in episodes if item.get("orbit_detected")) / total
    near_obstacle_dwell = sum(float(item.get("near_obstacle_dwell_ratio", 0.0)) for item in episodes) / total
    sensor_health_score = sum(float(item.get("sensor_health_score", 1.0)) for item in episodes) / total

    score = (
        successes / total * 100.0
        - collisions / total * 40.0
        - timeouts / total * 25.0
        - orbit_score * 25.0
        - progress_stall_rate * 20.0
        + path_efficiency * 15.0
        + net_progress_ratio * 10.0
        - high_clip_ratio * 10.0
    )
    if suite == "main":
        score -= near_obstacle_dwell * 5.0

    return EvalMetrics(
        success_rate=successes / total,
        collision_rate=collisions / total,
        timeout_rate=timeouts / total,
        mean_steps=mean_steps,
        reverse_case_success_rate=(reverse_successes / max(1, sum(1 for item in episodes if item.get("reverse_case")))),
        spin_proxy_rate=spin_proxy_rate,
        progress_stall_rate=progress_stall_rate,
        high_clip_ratio=high_clip_ratio,
        path_efficiency=path_efficiency,
        net_progress_ratio=net_progress_ratio,
        orbit_score=orbit_score,
        near_obstacle_dwell=near_obstacle_dwell,
        sensor_health_score=sensor_health_score,
        log_anomaly_count=float(log_anomaly_count),
        score=score,
        total_episodes=total,
        completed_episodes=total,
    )


def behavior_gate_violations(metrics: EvalMetrics, *, suite: str) -> list[str]:
    violations: list[str] = []
    if suite == "quick":
        if metrics.success_rate < 0.75:
            violations.append("success_rate<0.75")
        if metrics.collision_rate > 0.10:
            violations.append("collision_rate>0.10")
        if metrics.orbit_score > 0.10:
            violations.append("orbit_score>0.10")
        if metrics.progress_stall_rate > 0.25:
            violations.append("progress_stall_rate>0.25")
    else:
        if metrics.success_rate < 0.85:
            violations.append("success_rate<0.85")
        if metrics.collision_rate > 0.05:
            violations.append("collision_rate>0.05")
        if metrics.orbit_score > 0.05:
            violations.append("orbit_score>0.05")
        if metrics.spin_proxy_rate > 0.35 and metrics.net_progress_ratio < 0.25:
            violations.append("spin_proxy_rate>0.35_and_net_progress_ratio<0.25")
        if metrics.high_clip_ratio > 0.60 and metrics.path_efficiency < 0.45:
            violations.append("high_clip_ratio>0.60_and_path_efficiency<0.45")
        if metrics.progress_stall_rate > 0.20:
            violations.append("progress_stall_rate>0.20")
    if metrics.sensor_health_score < 0.80:
        violations.append("sensor_health_score<0.80")
    if metrics.log_anomaly_count > 0.0:
        violations.append("log_anomaly_count>0")
    return violations


def metrics_to_metadata(metrics: EvalMetrics) -> dict:
    return asdict(metrics)
