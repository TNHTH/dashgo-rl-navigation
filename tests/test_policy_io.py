from __future__ import annotations

from pathlib import Path

from dashgo_rl.deployment.policy_io import (
    extract_checkpoint_iteration,
    find_model_checkpoints,
    split_policy_and_normalizer_state,
)


def test_split_policy_and_normalizer_state_separates_legacy_normalizers() -> None:
    state = {
        "actor.0.weight": "policy-weight",
        "actor_obs_normalizer.mean": "actor-mean",
        "actor_obs_normalizer.var": "actor-var",
        "critic_obs_normalizer.mean": "critic-mean",
    }

    policy_state, normalizer_state = split_policy_and_normalizer_state(state)

    assert policy_state == {"actor.0.weight": "policy-weight"}
    assert normalizer_state == {"mean": "actor-mean", "var": "actor-var"}


def test_find_model_checkpoints_sorts_by_iteration_then_mtime(tmp_path: Path) -> None:
    low = tmp_path / "run_a" / "model_10.pt"
    high = tmp_path / "run_b" / "model_20.pt"
    final = tmp_path / "run_c" / "model_final.pt"
    for path in (low, high, final):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("checkpoint", encoding="utf-8")

    checkpoints = find_model_checkpoints([tmp_path])

    assert checkpoints[0] == high
    assert checkpoints[1] == low
    assert final not in checkpoints
    assert extract_checkpoint_iteration(high) == 20
    assert extract_checkpoint_iteration(final) == -1
