from __future__ import annotations

import json
from pathlib import Path

from tools.diagnostics import run_training_regression as regression


def test_detect_runtime_failure_matches_known_patterns() -> None:
    assert regression.detect_runtime_failure("PhysX failed to allocate GPU memory") is True
    assert regression.detect_runtime_failure("vkCreateRayTracingPipelinesKHR failed") is True
    assert regression.detect_runtime_failure("plain python traceback") is False


def test_resume_from_state_reuses_existing_summary_and_skips_completed_seed(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path
    metrics_root = project_root / ".artifacts" / "autopilot" / "metrics"
    generation_root = project_root / ".artifacts" / "autopilot" / "runs" / "gen2"
    metrics_root.mkdir(parents=True, exist_ok=True)
    generation_root.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_root / "existing_summary.json"
    state_path = metrics_root / "regression_state.json"
    state_path.write_text(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "runs": {
                    "41": {
                        "seed": 41,
                        "run_name": "bounded_tanh_regression_seed41",
                        "status": "completed",
                        "attempts": [],
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fake_find_latest_run_root(_generation_root: Path, run_name: str) -> Path:
        run_root = generation_root / f"20260325_000000_{run_name}"
        run_root.mkdir(parents=True, exist_ok=True)
        checkpoint = run_root / "checkpoints" / "model_1.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        (run_root / "run_meta.json").write_text(
            json.dumps({"run_name": run_name, "latest_checkpoint": str(checkpoint)}, ensure_ascii=False),
            encoding="utf-8",
        )
        return run_root

    monkeypatch.setattr(regression, "find_latest_run_root", fake_find_latest_run_root)
    monkeypatch.setattr(regression, "evaluation_passed", lambda **_: True)

    exit_code = regression.main(
        [
            "--project-root",
            str(project_root),
            "--seeds",
            "41,42",
            "--dry-run",
            "--resume-from-state",
            "--staging-export",
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["summary_path"] == str(summary_path)
    assert [item["seed"] for item in summary["runs"]] == [41, 42]
    assert summary["runs"][0]["status"] == "completed"
    assert summary["runs"][1]["status"] == "completed"


def test_evaluation_passed_supports_metrics_only_policy() -> None:
    eval_payload = {"status": "failed", "metrics": {"success_rate": 0.0}}

    assert regression.evaluation_passed(
        eval_result={"returncode": 1},
        eval_payload=eval_payload,
        evaluation_policy="metrics_only",
    ) is True
    assert regression.evaluation_passed(
        eval_result={"returncode": 1},
        eval_payload=eval_payload,
        evaluation_policy="completed",
    ) is False


def test_resolve_runtime_paths_uses_dry_run_suffix_for_dry_run(tmp_path: Path) -> None:
    state_path, events_path = regression.resolve_runtime_paths(tmp_path, dry_run=True)
    assert state_path.name == "regression_state.dry_run.json"
    assert events_path.name == "regression_events.dry_run.jsonl"

    state_path, events_path = regression.resolve_runtime_paths(tmp_path, dry_run=False)
    assert state_path.name == "regression_state.json"
    assert events_path.name == "regression_events.jsonl"


def test_parse_env_assignments_accepts_multiple_items() -> None:
    payload = regression.parse_env_assignments(["FOO=bar", "BAR=baz=qux"])
    assert payload == {"FOO": "bar", "BAR": "baz=qux"}
