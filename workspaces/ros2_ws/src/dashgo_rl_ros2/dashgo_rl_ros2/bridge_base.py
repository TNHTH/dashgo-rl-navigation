"""ROS2 bridge 的共享轻量工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def is_stale(now_sec: float, stamp_sec: float | None, timeout_sec: float) -> bool:
    """判断数据时间戳是否超过有效期；timeout<=0 表示不启用超时。"""
    if stamp_sec is None:
        return True
    return float(timeout_sec) > 0.0 and float(now_sec) - float(stamp_sec) > float(timeout_sec)


@dataclass
class BridgeCommandPublisher:
    """统一处理 debug cmd 与 shadow mode。"""

    cmd_pub: Any
    debug_cmd_pub: Any
    shadow_mode: bool = False

    def publish(self, twist: Any) -> None:
        self.debug_cmd_pub.publish(twist)
        if not self.shadow_mode:
            self.cmd_pub.publish(twist)
