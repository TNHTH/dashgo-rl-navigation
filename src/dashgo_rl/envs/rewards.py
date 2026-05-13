"""奖励项兼容入口。

保持导入轻量，不在模块导入时加载 Isaac Lab。
"""

from __future__ import annotations


def reward_distance_tracking_potential(*args, **kwargs):
    from dashgo_rl.dashgo_env_v2 import reward_distance_tracking_potential as _impl

    return _impl(*args, **kwargs)


__all__ = ["reward_distance_tracking_potential"]
