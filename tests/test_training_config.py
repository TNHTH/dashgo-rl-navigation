from __future__ import annotations

from dashgo_rl.training_config import resolve_num_envs


def test_resolve_num_envs_prefers_cli() -> None:
    assert resolve_num_envs(8, {"num_envs": 32}, 16) == 8


def test_resolve_num_envs_uses_yaml_before_env_default() -> None:
    assert resolve_num_envs(None, {"num_envs": 32}, 16) == 32


def test_resolve_num_envs_falls_back_to_env_default() -> None:
    assert resolve_num_envs(None, {}, 16) == 16
