from __future__ import annotations

from pathlib import Path

from autopilot import continuous_gen2_supervisor as supervisor
from autopilot.types import EvalMetrics, EvalRequest, EvalResult


def test_evaluate_checkpoint_for_promotion_blocks_failed_eval(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "wave70.log"
    log_file.write_text("[INFO] no traceback\n", encoding="utf-8")

    def fake_log_file_for_run_name(run_name: str) -> Path:
        return log_file

    def fake_build_eval_result(**kwargs):
        return EvalResult(
            status="failed",
            request=EvalRequest(
                checkpoint=Path(kwargs["checkpoint"]),
                suite="quick",
                project_root=Path(kwargs["project_root"]),
            ),
            metrics=EvalMetrics(
                success_rate=0.4,
                collision_rate=0.0,
                timeout_rate=0.6,
                orbit_score=0.2,
                progress_stall_rate=0.6,
                total_episodes=12,
                completed_episodes=12,
            ),
            notes=["behavior_gate_veto"],
        )

    monkeypatch.setattr(supervisor, "log_file_for_run_name", fake_log_file_for_run_name)
    monkeypatch.setattr(supervisor, "build_eval_result", fake_build_eval_result)

    passed, evidence = supervisor.evaluate_checkpoint_for_promotion(
        checkpoint_path=tmp_path / "model_1.pt",
        run_name="wave70",
        summary={
            "latest_scalars": {
                "Curriculum/target_adaptive": 3.75,
                "Episode_Termination/reach_goal": 1.0,
                "Episode_Termination/object_collision": 0.0,
                "Episode_Termination/time_out": 0.0,
                "Metrics/target_pose/position_error": 0.2,
            },
            "run_dir": str(tmp_path),
            "seconds_since_update": 10.0,
        },
    )
    assert passed is False
    assert evidence["eval"]["status"] == "failed"


def test_evaluate_checkpoint_for_promotion_escalates_doctor_failure(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "wave71.log"
    log_file.write_text("Traceback (most recent call last):\nKeyError: obs_groups\n", encoding="utf-8")

    calls = {}

    def fake_log_file_for_run_name(run_name: str) -> Path:
        return log_file

    def fake_escalate(job_type: str, **kwargs):
        calls["job_type"] = job_type
        calls["kwargs"] = kwargs
        return {"status": "queued_only", "job_type": job_type}

    monkeypatch.setattr(supervisor, "log_file_for_run_name", fake_log_file_for_run_name)
    monkeypatch.setattr(supervisor, "escalate_codex", fake_escalate)

    passed, evidence = supervisor.evaluate_checkpoint_for_promotion(
        checkpoint_path=tmp_path / "model_2.pt",
        run_name="wave71",
        summary={"latest_scalars": {}, "run_dir": str(tmp_path)},
    )
    assert passed is False
    assert calls["job_type"] == "debug_job"
    assert evidence["doctor"]["status"] == "failed"


def test_pause_requested_reads_state(monkeypatch) -> None:
    monkeypatch.setattr(
        supervisor,
        "read_state",
        lambda: {"desired_state": supervisor.PAUSE_AFTER_CURRENT_RUN, "pause_scope": supervisor.PAUSE_SCOPE_ALL},
    )

    assert supervisor.pause_requested() is True
    assert supervisor.pause_scope() == "all"


def test_active_regression_state_detects_running_regression(monkeypatch, tmp_path) -> None:
    state_path = tmp_path / "regression_state.json"
    state_path.write_text('{"status": "train_running", "current_run_name": "bounded_tanh_regression_seed41"}', encoding="utf-8")
    monkeypatch.setattr(supervisor, "REGRESSION_STATE_PATH", state_path)

    payload = supervisor.active_regression_state()

    assert payload is not None
    assert payload["current_run_name"] == "bounded_tanh_regression_seed41"


def test_active_autoresearch_state_detects_running_supervisor(monkeypatch, tmp_path) -> None:
    state_path = tmp_path / "autoresearch_state.json"
    state_path.write_text('{"supervisor_status": "train_running", "next_trial": "reward.orbit_weight.up_4_0"}', encoding="utf-8")
    monkeypatch.setattr(supervisor, "AUTORESEARCH_STATE_PATH", state_path)

    payload = supervisor.active_autoresearch_state()

    assert payload is not None
    assert payload["next_trial"] == "reward.orbit_weight.up_4_0"
