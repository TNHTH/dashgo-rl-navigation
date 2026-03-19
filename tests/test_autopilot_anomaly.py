from __future__ import annotations

from autopilot.anomaly import (
    analyze_live_sensor_payload,
    analyze_log_text,
    behavior_gate_violations,
    prefilter_training_summary,
    summarize_eval_episodes,
)


def test_analyze_log_text_detects_traceback_and_camera_issue() -> None:
    text = """
Traceback (most recent call last):
RuntimeError: boom
[INFO] enable_cameras 缺失
"""
    result = analyze_log_text(text, log_path="/tmp/train.log")
    names = {check.name for check in result.checks}
    assert result.status == "failed"
    assert "python_traceback" in names
    assert "runtime_error" in names
    assert "camera_disabled" in names
    assert result.recommended_action == "codex_debug"


def test_analyze_live_sensor_payload_detects_flatline() -> None:
    payload = {
        "profile": "gen2",
        "samples": [
            {
                "lidar": {"min": 0.0, "max": 0.0, "mean": 0.0},
                "min_obstacle_distance": {"min": 0.0, "max": 0.0, "mean": 0.0},
            }
            for _ in range(12)
        ],
    }
    result = analyze_live_sensor_payload(payload, evidence_path="/tmp/live.json")
    names = {check.name for check in result.checks}
    assert result.status == "failed"
    assert "stitched_lidar_flatline" in names


def test_behavior_gate_blocks_orbiting_policy() -> None:
    episodes = [
        {
            "termination_reason": "time_out",
            "reverse_case": False,
            "steps": 120,
            "path_efficiency": 0.15,
            "net_progress_ratio": 0.10,
            "near_obstacle_dwell_ratio": 0.40,
            "spin_proxy_ratio": 0.70,
            "high_clip_ratio": 0.85,
            "progress_stall": True,
            "orbit_detected": True,
            "sensor_health_score": 1.0,
        }
        for _ in range(12)
    ]
    metrics = summarize_eval_episodes(episodes, suite="quick")
    violations = behavior_gate_violations(metrics, suite="quick")
    assert "orbit_score>0.10" in violations
    assert "progress_stall_rate>0.25" in violations


def test_prefilter_training_summary_requires_clean_scalars() -> None:
    summary = {
        "latest_scalars": {
            "Curriculum/target_adaptive": 3.75,
            "Episode_Termination/reach_goal": 1.0,
            "Episode_Termination/object_collision": 0.0,
            "Episode_Termination/time_out": 0.0,
            "Metrics/target_pose/position_error": 0.22,
        }
    }
    assert prefilter_training_summary(summary, base_curriculum=3.75) is True
    summary["latest_scalars"]["Episode_Termination/time_out"] = 0.5
    assert prefilter_training_summary(summary, base_curriculum=3.75) is False
