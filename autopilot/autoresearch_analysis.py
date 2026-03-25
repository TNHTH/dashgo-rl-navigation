from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, read_json, write_json


ITERATION_STATES_WITH_FATAL_FAILURE = {"failed", "blocked_runtime"}
PARAMETER_SCOPE = "parameter"
STRUCTURE_SCOPE = "structure"
CODE_SCOPE = "code"


def iso_now_local() -> str:
    return datetime.now().astimezone().isoformat()


def compute_score(metrics: dict[str, Any]) -> float:
    success_rate = float(metrics.get("success_rate", 0.0))
    collision_rate = float(metrics.get("collision_rate", 0.0))
    progress_stall_rate = float(metrics.get("progress_stall_rate", 0.0))
    orbit_score = float(metrics.get("orbit_score", 0.0))
    timeout_rate = float(metrics.get("timeout_rate", 0.0))
    cmd_saturation_rate = float(metrics.get("cmd_saturation_rate", 0.0))
    path_efficiency = float(metrics.get("path_efficiency", 0.0))
    net_progress_ratio = float(metrics.get("net_progress_ratio", 0.0))
    reverse_case_success_rate = float(metrics.get("reverse_case_success_rate", 0.0))
    return (
        100.0 * success_rate
        - 40.0 * collision_rate
        - 25.0 * progress_stall_rate
        - 20.0 * orbit_score
        - 10.0 * timeout_rate
        - 8.0 * cmd_saturation_rate
        - 5.0 * (1.0 - path_efficiency)
        - 5.0 * (1.0 - net_progress_ratio)
        + 3.0 * reverse_case_success_rate
    )


def guard_violations(*, run_status: str, metrics: dict[str, Any] | None) -> list[str]:
    violations: list[str] = []
    if run_status in ITERATION_STATES_WITH_FATAL_FAILURE:
        violations.append(f"run_status={run_status}")
    if not isinstance(metrics, dict):
        violations.append("metrics_missing")
        return violations
    if float(metrics.get("plan_invalid_ratio", 1.0)) != 0.0:
        violations.append("plan_invalid_ratio!=0")
    if float(metrics.get("collision_rate", 1.0)) > 0.05:
        violations.append("collision_rate>0.05")
    return violations


def make_idea(
    *,
    idea_id: str,
    family: str,
    hypothesis: str,
    target_metric: str,
    expected_direction: str,
    change_scope: str,
    priority: int,
    source: str,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "idea_id": idea_id,
        "family": family,
        "hypothesis": hypothesis,
        "target_metric": target_metric,
        "expected_direction": expected_direction,
        "change_scope": change_scope,
        "priority": priority,
        "source": source,
        "profile": profile or {"env": {}, "config": {}},
    }


def default_ideas() -> list[dict[str, Any]]:
    return [
        make_idea(
            idea_id="reward.orbit_weight.up_4_0",
            family="reward",
            hypothesis="提高 orbit 惩罚权重，压制绕圈倾向。",
            target_metric="orbit_score",
            expected_direction="down",
            change_scope=PARAMETER_SCOPE,
            priority=100,
            source="manual_seeded",
            profile={"env": {"DASHGO_ORBIT_WEIGHT": "4.0"}, "config": {}},
        ),
        make_idea(
            idea_id="reward.progress_stall_weight.up_4_5",
            family="reward",
            hypothesis="提高 progress stall 惩罚，降低原地停滞。",
            target_metric="progress_stall_rate",
            expected_direction="down",
            change_scope=PARAMETER_SCOPE,
            priority=95,
            source="manual_seeded",
            profile={"env": {"DASHGO_PROGRESS_STALL_WEIGHT": "4.5"}, "config": {}},
        ),
        make_idea(
            idea_id="optimizer.learning_rate.down_1e4",
            family="optimizer",
            hypothesis="降低学习率，减少训练后段发散和 NaN。",
            target_metric="cmd_saturation_rate",
            expected_direction="down",
            change_scope=PARAMETER_SCOPE,
            priority=88,
            source="manual_seeded",
            profile={"env": {}, "config": {"algorithm": {"learning_rate": 1.0e-4}}},
        ),
        make_idea(
            idea_id="optimizer.entropy.down_0_003",
            family="optimizer",
            hypothesis="适度降低 entropy，减少随机抖动和大角速度抽搐。",
            target_metric="orbit_score",
            expected_direction="down",
            change_scope=PARAMETER_SCOPE,
            priority=82,
            source="manual_seeded",
            profile={"env": {}, "config": {"algorithm": {"entropy_coef": 0.003}}},
        ),
        make_idea(
            idea_id="optimizer.desired_kl.down_0_003",
            family="optimizer",
            hypothesis="更早触发自适应降速，压低过激更新。",
            target_metric="cmd_saturation_rate",
            expected_direction="down",
            change_scope=PARAMETER_SCOPE,
            priority=78,
            source="manual_seeded",
            profile={"env": {}, "config": {"algorithm": {"desired_kl": 0.003}}},
        ),
        make_idea(
            idea_id="policy.actor_hidden_dims.up_160_96",
            family="policy",
            hypothesis="适度增大 actor head，给局部避障策略更多容量。",
            target_metric="success_rate",
            expected_direction="up",
            change_scope=STRUCTURE_SCOPE,
            priority=60,
            source="manual_seeded",
            profile={"env": {}, "config": {"policy": {"actor_hidden_dims": [160, 96]}}},
        ),
        make_idea(
            idea_id="policy.init_noise_std.down_0_7",
            family="policy",
            hypothesis="降低初始噪声强度，避免训练初期和恢复阶段过度抖动。",
            target_metric="cmd_saturation_rate",
            expected_direction="down",
            change_scope=STRUCTURE_SCOPE,
            priority=58,
            source="manual_seeded",
            profile={"env": {}, "config": {"policy": {"init_noise_std": 0.7}}},
        ),
        make_idea(
            idea_id="code.nan_guard.policy_distribution",
            family="policy",
            hypothesis="在策略分布前增加有限值保护，阻断 NaN 扩散。",
            target_metric="success_rate",
            expected_direction="up",
            change_scope=CODE_SCOPE,
            priority=50,
            source="manual_seeded",
            profile={"env": {}, "config": {}},
        ),
    ]


def dedupe_ideas(ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in sorted(ideas, key=lambda payload: (-int(payload.get("priority", 0)), str(payload.get("idea_id", "")))):
        idea_id = str(item.get("idea_id", "")).strip()
        if not idea_id or idea_id in seen:
            continue
        seen.add(idea_id)
        unique.append(item)
    return unique


def ensure_ideas_queue(path: Path) -> list[dict[str, Any]]:
    existing = read_json(path, default=None)
    if isinstance(existing, list) and existing:
        ideas = dedupe_ideas(existing)
    else:
        ideas = dedupe_ideas(default_ideas())
    write_json(path, ideas)
    return ideas


def allowed_scopes(*, iteration_index: int, no_improve_streak: int) -> set[str]:
    if iteration_index < 6:
        return {PARAMETER_SCOPE}
    if no_improve_streak >= 2:
        return {PARAMETER_SCOPE, STRUCTURE_SCOPE, CODE_SCOPE}
    return {PARAMETER_SCOPE, STRUCTURE_SCOPE}


def choose_next_idea(
    ideas: list[dict[str, Any]],
    *,
    iteration_index: int,
    tried_idea_ids: set[str],
    no_improve_streak: int,
) -> dict[str, Any]:
    scopes = allowed_scopes(iteration_index=iteration_index, no_improve_streak=no_improve_streak)
    for item in dedupe_ideas(ideas):
        if item.get("idea_id") in tried_idea_ids:
            continue
        if item.get("change_scope") not in scopes:
            continue
        return item
    return dedupe_ideas(ideas)[0]


def extend_ideas_from_failure(summary: dict[str, Any], metrics: dict[str, Any] | None) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    runs = summary.get("runs") or []
    run = runs[0] if runs else {}
    attempts = run.get("attempts") or []
    latest_attempt = attempts[-1] if attempts else {}
    train_log_raw = str((latest_attempt.get("train") or {}).get("log_path") or "").strip()
    train_log = Path(train_log_raw) if train_log_raw else None
    log_text = ""
    if train_log is not None and train_log.exists() and train_log.is_file():
        log_text = train_log.read_text(encoding="utf-8", errors="ignore")
    lowered = log_text.lower()
    if "nan" in lowered or "expected parameter loc" in lowered:
        generated.extend(
            [
                make_idea(
                    idea_id="stability.learning_rate.down_8e5",
                    family="optimizer",
                    hypothesis="NaN 失败，进一步下调学习率观察是否稳定。",
                    target_metric="success_rate",
                    expected_direction="up",
                    change_scope=PARAMETER_SCOPE,
                    priority=110,
                    source="analysis_generated",
                    profile={"env": {}, "config": {"algorithm": {"learning_rate": 8.0e-5}}},
                ),
                make_idea(
                    idea_id="stability.entropy.down_0_002",
                    family="optimizer",
                    hypothesis="NaN 失败，进一步降低 entropy 减少激进探索。",
                    target_metric="cmd_saturation_rate",
                    expected_direction="down",
                    change_scope=PARAMETER_SCOPE,
                    priority=108,
                    source="analysis_generated",
                    profile={"env": {}, "config": {"algorithm": {"entropy_coef": 0.002}}},
                ),
                make_idea(
                    idea_id="stability.init_noise_std.down_0_5",
                    family="policy",
                    hypothesis="NaN 失败，进一步降低策略初始噪声。",
                    target_metric="cmd_saturation_rate",
                    expected_direction="down",
                    change_scope=STRUCTURE_SCOPE,
                    priority=104,
                    source="analysis_generated",
                    profile={"env": {}, "config": {"policy": {"init_noise_std": 0.5}}},
                ),
            ]
        )
    if isinstance(metrics, dict):
        if float(metrics.get("orbit_score", 0.0)) > 0.10:
            generated.append(
                make_idea(
                    idea_id="reward.orbit_trigger_steps.down_8",
                    family="reward",
                    hypothesis="更早识别 orbit 模式，提前施加惩罚。",
                    target_metric="orbit_score",
                    expected_direction="down",
                    change_scope=PARAMETER_SCOPE,
                    priority=92,
                    source="analysis_generated",
                    profile={"env": {"DASHGO_ORBIT_TRIGGER_STEPS": "8"}, "config": {}},
                )
            )
        if float(metrics.get("progress_stall_rate", 0.0)) > 0.25:
            generated.append(
                make_idea(
                    idea_id="reward.progress_stall_weight.up_5_0",
                    family="reward",
                    hypothesis="停滞率过高，继续提高 stall 惩罚。",
                    target_metric="progress_stall_rate",
                    expected_direction="down",
                    change_scope=PARAMETER_SCOPE,
                    priority=91,
                    source="analysis_generated",
                    profile={"env": {"DASHGO_PROGRESS_STALL_WEIGHT": "5.0"}, "config": {}},
                )
            )
        if float(metrics.get("cmd_saturation_rate", 0.0)) > 0.20:
            generated.append(
                make_idea(
                    idea_id="optimizer.desired_kl.down_0_0025",
                    family="optimizer",
                    hypothesis="动作饱和率偏高，提前触发自适应学习率降速。",
                    target_metric="cmd_saturation_rate",
                    expected_direction="down",
                    change_scope=PARAMETER_SCOPE,
                    priority=89,
                    source="analysis_generated",
                    profile={"env": {}, "config": {"algorithm": {"desired_kl": 0.0025}}},
                )
            )
    return dedupe_ideas(generated)


def extract_primary_run(summary: dict[str, Any]) -> dict[str, Any]:
    runs = summary.get("runs") or []
    if not runs:
        return {}
    return runs[0]


def extract_metrics(summary: dict[str, Any]) -> dict[str, Any] | None:
    run = extract_primary_run(summary)
    payload = run.get("eval_payload")
    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        return payload["metrics"]
    attempts = run.get("attempts") or []
    if attempts:
        latest = attempts[-1]
        payload = latest.get("eval_payload")
        if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
            return payload["metrics"]
    return None


def update_insights(insights_path: Path, *, message: str) -> None:
    ensure_dir(insights_path.parent)
    existing = insights_path.read_text(encoding="utf-8") if insights_path.exists() else "# DashGo Autoresearch Insights\n\n"
    if message in existing:
        return
    with insights_path.open("a", encoding="utf-8") as handle:
        if existing and not existing.endswith("\n"):
            handle.write("\n")
        handle.write(f"- {message}\n")


def write_iteration_archive(iteration_root: Path, payloads: dict[str, Any]) -> None:
    ensure_dir(iteration_root)
    for name, payload in payloads.items():
        target = iteration_root / name
        if target.suffix == ".md":
            text = payload if isinstance(payload, str) else str(payload)
            target.write_text(text if text.endswith("\n") else f"{text}\n", encoding="utf-8")
        else:
            write_json(target, payload)


def analyze_iteration(
    *,
    iteration_id: str,
    iteration_root: Path,
    idea: dict[str, Any],
    summary: dict[str, Any],
    best_candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    run = extract_primary_run(summary)
    metrics = extract_metrics(summary)
    summary_status = str(summary.get("status") or run.get("status") or "failed")
    score = compute_score(metrics) if isinstance(metrics, dict) else None
    violations = guard_violations(run_status=summary_status, metrics=metrics)
    best_score = None
    if isinstance(best_candidate, dict) and isinstance(best_candidate.get("score"), (float, int)):
        best_score = float(best_candidate["score"])
    keep = False
    promotion_candidate = False
    if not violations and score is not None:
        if best_score is None or score >= best_score + 3.0:
            keep = True
        promotion_candidate = best_score is not None and score >= best_score + 8.0
    next_ideas = extend_ideas_from_failure(summary, metrics)
    reason_parts = []
    if violations:
        reason_parts.append("hard_guard 未通过")
    if score is None:
        reason_parts.append("没有可比较的评估指标")
    elif best_score is None:
        reason_parts.append("首次形成可用候选")
    else:
        delta = score - best_score
        reason_parts.append(f"score Δ={delta:.3f}")
    decision = "keep" if keep else "discard"
    analysis_lines = [
        f"# Iteration {iteration_id}",
        "",
        f"- idea_id: `{idea.get('idea_id')}`",
        f"- family: `{idea.get('family')}`",
        f"- decision: `{decision}`",
        f"- summary_status: `{summary_status}`",
        f"- score: `{score}`",
        f"- best_score_before: `{best_score}`",
        f"- guard_violations: `{violations}`",
        f"- reason: {'; '.join(reason_parts) if reason_parts else 'n/a'}",
    ]
    if isinstance(metrics, dict):
        analysis_lines.extend(
            [
                "",
                "## Metrics",
                "",
                *[f"- {key}: {value}" for key, value in sorted(metrics.items())],
            ]
        )
    payload = {
        "iteration_id": iteration_id,
        "created_at": iso_now_local(),
        "idea": idea,
        "decision": decision,
        "summary_status": summary_status,
        "score": score,
        "best_score_before": best_score,
        "guard_violations": violations,
        "promotion_candidate": promotion_candidate,
        "next_ideas": next_ideas,
        "metrics": metrics,
        "reason": "; ".join(reason_parts),
    }
    write_iteration_archive(
        iteration_root,
        {
            "analysis.md": "\n".join(analysis_lines) + "\n",
            "decision.json": payload,
            "eval_quick.json": run.get("eval_payload") or {},
            "train_summary.json": summary,
        },
    )
    return payload
