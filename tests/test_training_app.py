from __future__ import annotations

from types import SimpleNamespace


def _args(**overrides):
    values = {
        "gen": "Gen 2",
        "num_envs": None,
        "checkpoint": None,
        "resume": False,
        "run_name": None,
        "seed": None,
        "max_iterations": None,
        "save_interval": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_training_app_prepares_flat_agent_config() -> None:
    from dashgo_rl.training_app import DashGoTrainingApp

    app = DashGoTrainingApp(
        _args(run_name=" My Run ", seed=7, max_iterations=12, save_interval=3),
        script_dir="/tmp/project",
    )

    agent_cfg = app.prepare_agent_config(
        {
            "runner": {"num_steps_per_env": 24, "max_iterations": 100},
            "run_name": "from-yaml",
        },
        autoresearch_overrides={"config": {"runner": {"save_interval": 50}}},
    )

    assert agent_cfg["num_steps_per_env"] == 24
    assert agent_cfg["seed"] == 7
    assert agent_cfg["max_iterations"] == 12
    assert agent_cfg["save_interval"] == 3
    assert agent_cfg["run_name"] == "My_Run"
    assert agent_cfg["obs_groups"] == {"policy": ["policy"], "critic": ["policy"]}
    assert agent_cfg["device"] == "cuda:0"


def test_training_app_resolves_resume_checkpoint(tmp_path) -> None:
    from dashgo_rl.training_app import DashGoTrainingApp

    low = tmp_path / "model_1.pt"
    high = tmp_path / "model_10.pt"
    low.write_text("low", encoding="utf-8")
    high.write_text("high", encoding="utf-8")

    app = DashGoTrainingApp(_args(resume=True), script_dir=str(tmp_path))

    assert app.resolve_resume_checkpoint([tmp_path]) == str(high)


def test_training_app_curriculum_sidecar_path() -> None:
    from dashgo_rl.training_app import curriculum_state_sidecar_path

    assert curriculum_state_sidecar_path("/tmp/model_42.pt").name == "model_42.curriculum.json"
