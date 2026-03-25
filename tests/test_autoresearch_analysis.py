from __future__ import annotations

from pathlib import Path

from autopilot import autoresearch_analysis as analysis


def test_compute_score_matches_contract() -> None:
    metrics = {
        "success_rate": 0.8,
        "collision_rate": 0.05,
        "progress_stall_rate": 0.10,
        "orbit_score": 0.05,
        "timeout_rate": 0.10,
        "cmd_saturation_rate": 0.20,
        "path_efficiency": 0.60,
        "net_progress_ratio": 0.70,
        "reverse_case_success_rate": 0.40,
    }
    score = analysis.compute_score(metrics)
    assert round(score, 3) == 69.6


def test_choose_next_idea_prefers_parameter_scope_in_first_six_rounds() -> None:
    ideas = analysis.default_ideas()
    picked = analysis.choose_next_idea(
        ideas,
        iteration_index=0,
        tried_idea_ids={"reward.orbit_weight.up_4_0"},
        no_improve_streak=0,
    )
    assert picked["change_scope"] == analysis.PARAMETER_SCOPE
    assert picked["idea_id"] != "reward.orbit_weight.up_4_0"


def test_analyze_iteration_keeps_candidate_when_score_improves(tmp_path: Path) -> None:
    iteration_root = tmp_path / "iter"
    summary = {
        "status": "completed",
        "runs": [
            {
                "seed": 141,
                "run_name": "autoresearch_seed141",
                "run_root": str(tmp_path / "run"),
                "latest_checkpoint": str(tmp_path / "run" / "checkpoints" / "model_10.pt"),
                "eval_payload": {
                    "metrics": {
                        "success_rate": 0.85,
                        "collision_rate": 0.0,
                        "progress_stall_rate": 0.10,
                        "orbit_score": 0.05,
                        "timeout_rate": 0.10,
                        "cmd_saturation_rate": 0.10,
                        "path_efficiency": 0.65,
                        "net_progress_ratio": 0.75,
                        "reverse_case_success_rate": 0.30,
                        "plan_invalid_ratio": 0.0,
                    }
                },
            }
        ],
    }
    idea = analysis.default_ideas()[0]
    payload = analysis.analyze_iteration(
        iteration_id="20260325_120000_iter0000",
        iteration_root=iteration_root,
        idea=idea,
        summary=summary,
        best_candidate={"score": 50.0},
    )
    assert payload["decision"] == "keep"
    assert payload["guard_violations"] == []
    assert (iteration_root / "analysis.md").exists()


def test_analyze_iteration_discards_and_generates_followup_ideas(tmp_path: Path) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text("ValueError: Expected parameter loc ... nan", encoding="utf-8")
    iteration_root = tmp_path / "iter"
    summary = {
        "status": "failed",
        "runs": [
            {
                "status": "failed",
                "attempts": [
                    {
                        "train": {"log_path": str(train_log)},
                    }
                ],
            }
        ],
    }
    idea = analysis.default_ideas()[0]
    payload = analysis.analyze_iteration(
        iteration_id="20260325_120000_iter0001",
        iteration_root=iteration_root,
        idea=idea,
        summary=summary,
        best_candidate={"score": 50.0},
    )
    assert payload["decision"] == "discard"
    assert any(item["idea_id"].startswith("stability.") for item in payload["next_ideas"])
