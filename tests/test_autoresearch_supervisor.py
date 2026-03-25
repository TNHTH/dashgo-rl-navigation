from __future__ import annotations

import json
import os
from pathlib import Path

from autopilot import autoresearch_supervisor as supervisor


def test_regression_runner_snapshot_detects_active_process(tmp_path: Path) -> None:
    state_path = tmp_path / "regression_state.json"
    pid_path = tmp_path / "training_regression.pid"
    state_path.write_text(json.dumps({"status": "train_running", "current_run_name": "run41"}), encoding="utf-8")
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    payload = supervisor.regression_runner_snapshot(state_path, pid_path)

    assert payload is not None
    assert payload["active"] is True
    assert payload["current_run_name"] == "run41"


def test_resolve_baseline_checkpoint_prefers_best_candidate(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / ".git").mkdir()
    (project_root / "README.md").write_text("repo\n", encoding="utf-8")
    best_candidate_path = tmp_path / "best_candidate.json"
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    best_candidate_path.write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint),
                "sensor_contract": supervisor.current_sensor_contract(),
            }
        ),
        encoding="utf-8",
    )

    payload = supervisor.resolve_baseline_checkpoint(
        project_root,
        best_candidate_path,
        explicit=None,
        required_sensor_contract=supervisor.current_sensor_contract(),
    )

    assert payload["source"] == "best_candidate"
    assert payload["checkpoint_path"] == str(checkpoint)


def test_resolve_baseline_checkpoint_rejects_stale_contract(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / ".git").mkdir()
    (project_root / "README.md").write_text("repo\n", encoding="utf-8")
    best_candidate_path = tmp_path / "best_candidate.json"
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    best_candidate_path.write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint),
                "sensor_contract": {"contract_id": "legacy_360"},
            }
        ),
        encoding="utf-8",
    )

    payload = supervisor.resolve_baseline_checkpoint(
        project_root,
        best_candidate_path,
        explicit=None,
        required_sensor_contract=supervisor.current_sensor_contract(),
    )

    assert payload is None


def test_safe_pause_requested_reads_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"desired_state": "pause_after_current_run"}), encoding="utf-8")
    assert supervisor.safe_pause_requested(state_path) is True
