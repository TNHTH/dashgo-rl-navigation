"""训练配置解析工具。"""

from __future__ import annotations


def resolve_num_envs(cli_num_envs: int | None, agent_cfg: dict, env_default: int) -> int:
    """环境数量优先级：CLI > YAML > 环境默认值。"""
    if cli_num_envs is not None:
        return int(cli_num_envs)
    yaml_num_envs = agent_cfg.get("num_envs")
    if yaml_num_envs is not None:
        return int(yaml_num_envs)
    return int(env_default)
