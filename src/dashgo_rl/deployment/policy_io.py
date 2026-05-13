"""策略 checkpoint 与 normalizer 的轻量 I/O 工具。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping


def extract_checkpoint_iteration(path: str | Path) -> int:
    """从 `model_<iteration>.pt` 文件名提取迭代数。"""
    match = re.search(r"model_(\d+)\.pt$", Path(path).name)
    return int(match.group(1)) if match else -1


def find_model_checkpoints(search_roots: Iterable[str | Path]) -> list[Path]:
    """递归查找并按迭代数、mtime 降序排序 `model_*.pt`。"""
    candidates: list[Path] = []
    for root in search_roots:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            continue
        candidates.extend(
            path
            for path in root_path.glob("**/model_*.pt")
            if extract_checkpoint_iteration(path) >= 0
        )
    return sorted(
        candidates,
        key=lambda path: (extract_checkpoint_iteration(path), path.stat().st_mtime),
        reverse=True,
    )


def split_policy_and_normalizer_state(
    state_dict: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """拆分旧 checkpoint 中混在模型权重里的 actor normalizer 状态。"""
    policy_state = dict(state_dict)
    normalizer_state: dict[str, Any] = {}

    for key in list(policy_state.keys()):
        if key.startswith("actor_obs_normalizer."):
            normalizer_state[key.replace("actor_obs_normalizer.", "", 1)] = policy_state.pop(key)
        elif key.startswith("critic_obs_normalizer."):
            policy_state.pop(key)

    return policy_state, normalizer_state or None


class PolicyCheckpointLoader:
    """无框架依赖的 checkpoint 候选选择器。"""

    def __init__(self, search_roots: Iterable[str | Path]) -> None:
        self.search_roots = tuple(Path(root).expanduser() for root in search_roots)

    def candidates(self) -> list[Path]:
        return find_model_checkpoints(self.search_roots)
