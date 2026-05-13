"""策略 checkpoint 与 normalizer 的轻量 I/O 工具。"""

from __future__ import annotations

import re
from dataclasses import dataclass
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


def prepend_manual_checkpoint(candidates: Iterable[str | Path], manual_checkpoint: str | Path | None) -> list[Path]:
    """把显式 checkpoint 放到候选列表首位，并去重。"""
    normalized = [Path(path).expanduser().resolve() for path in candidates]
    if manual_checkpoint is None or str(manual_checkpoint).strip() == "":
        return normalized

    manual = Path(manual_checkpoint).expanduser().resolve()
    if not manual.exists():
        raise FileNotFoundError(f"显式指定的 checkpoint 不存在: {manual}")
    return [manual, *[candidate for candidate in normalized if candidate != manual]]


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


@dataclass(frozen=True)
class PolicyNormalizerBundle:
    """checkpoint 中策略权重和观测 normalizer 状态的组合。"""

    policy_state: dict[str, Any]
    normalizer_state: dict[str, Any] | None = None

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> "PolicyNormalizerBundle":
        raw_state = checkpoint.get("model_state_dict", checkpoint)
        policy_state, fallback_state = split_policy_and_normalizer_state(raw_state)
        normalizer_state = checkpoint.get("obs_norm_state_dict", fallback_state)
        return cls(policy_state=policy_state, normalizer_state=normalizer_state)

    def build_normalizer(self, obs_dim: int, normalizer_cls: Any, device: Any = None) -> Any:
        """按需构造 RSL-RL EmpiricalNormalization；无状态时返回 None。"""
        if self.normalizer_state is None:
            return None
        normalizer = normalizer_cls(shape=[obs_dim])
        if device is not None and hasattr(normalizer, "to"):
            normalizer = normalizer.to(device)
        normalizer.load_state_dict(self.normalizer_state, strict=True)
        normalizer.eval()
        return normalizer


class PolicyCheckpointLoader:
    """无框架依赖的 checkpoint 候选选择器。"""

    def __init__(self, search_roots: Iterable[str | Path]) -> None:
        self.search_roots = tuple(Path(root).expanduser() for root in search_roots)

    def candidates(self) -> list[Path]:
        return find_model_checkpoints(self.search_roots)


class GeoNavPolicyFactory:
    """集中保存 GeoNavPolicy 默认构造参数。"""

    def __init__(
        self,
        policy_cls: Any,
        num_actions: int = 2,
        actor_hidden_dims: Iterable[int] = (128, 64),
        critic_hidden_dims: Iterable[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
    ) -> None:
        self.policy_cls = policy_cls
        self.num_actions = int(num_actions)
        self.actor_hidden_dims = list(actor_hidden_dims)
        self.critic_hidden_dims = list(critic_hidden_dims)
        self.activation = activation
        self.init_noise_std = float(init_noise_std)

    def create(self, obs: Any, obs_groups: Any = None, device: Any = None) -> Any:
        policy = self.policy_cls(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=self.num_actions,
            actor_hidden_dims=self.actor_hidden_dims,
            critic_hidden_dims=self.critic_hidden_dims,
            activation=self.activation,
            init_noise_std=self.init_noise_std,
        )
        return policy.to(device) if device is not None and hasattr(policy, "to") else policy


def load_policy_and_normalizer(
    policy: Any,
    checkpoint_path: str | Path,
    torch_module: Any,
    normalizer_cls: Any,
    device: Any = None,
) -> tuple[Any, Any | None]:
    """加载策略权重和 normalizer，torch 由调用方在 Isaac 启动后传入。"""
    checkpoint = torch_module.load(str(checkpoint_path), map_location=device)
    bundle = PolicyNormalizerBundle.from_checkpoint(checkpoint)
    policy.load_state_dict(bundle.policy_state, strict=True)
    normalizer = bundle.build_normalizer(policy.num_actor_obs, normalizer_cls, device=device)
    return policy, normalizer
