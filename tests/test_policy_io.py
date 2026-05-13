from __future__ import annotations

import ast
from pathlib import Path

from dashgo_rl.deployment.policy_io import (
    PolicyNormalizerBundle,
    extract_checkpoint_iteration,
    find_model_checkpoints,
    prepend_manual_checkpoint,
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


def test_prepend_manual_checkpoint_validates_and_deduplicates(tmp_path: Path) -> None:
    manual = tmp_path / "manual.pt"
    existing = tmp_path / "model_1.pt"
    manual.write_text("manual", encoding="utf-8")
    existing.write_text("existing", encoding="utf-8")

    checkpoints = prepend_manual_checkpoint([manual, existing], manual)

    assert checkpoints == [manual, existing]


def test_policy_normalizer_bundle_prefers_checkpoint_normalizer_state() -> None:
    checkpoint = {
        "model_state_dict": {
            "actor.weight": "policy",
            "actor_obs_normalizer.mean": "legacy",
        },
        "obs_norm_state_dict": {"mean": "external"},
    }

    bundle = PolicyNormalizerBundle.from_checkpoint(checkpoint)

    assert bundle.policy_state == {"actor.weight": "policy"}
    assert bundle.normalizer_state == {"mean": "external"}


def test_policy_io_has_no_top_level_torch_or_isaac_imports() -> None:
    module_path = Path(__file__).resolve().parents[1] / "src" / "dashgo_rl" / "deployment" / "policy_io.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    assert "torch" not in imports
    assert all(not name.startswith("isaac") for name in imports)
