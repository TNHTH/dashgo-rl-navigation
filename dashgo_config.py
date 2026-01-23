"""
DashGo机器人配置中心

开发基准: Isaac Sim 4.5 + Ubuntu 20.04
参数来源: ROS配置文件 (dashgo/EAI驱动/dashgo_bringup/config/)

用途:
    统一管理DashGo机器人的所有物理参数，确保仿真与实物对齐。
    所有参数从ROS配置文件读取，避免硬编码。
"""

import yaml
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DashGoROSParams:
    """
    DashGo ROS参数配置类

    属性:
        wheel_diameter: 轮子直径（米）
        wheel_track: 轮距（左右轮中心距，米）
        encoder_resolution: 编码器线数（ticks/转）
    """
    wheel_diameter: float = 0.1264
    wheel_track: float = 0.3420
    encoder_resolution: int = 1200

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> 'DashGoROSParams':
        """
        从ROS YAML配置文件加载参数

        开发基准: Isaac Sim 4.5 + Ubuntu 20.04
        参考文档: ROS官方YAML配置规范

        Args:
            yaml_path: YAML文件路径。如果为None，使用默认路径。

        Returns:
            DashGoROSParams: 参数配置对象

        说明:
            如果YAML文件不存在，返回默认值（与原硬编码值一致）。
            默认路径: dashgo/EAI驱动/dashgo_bringup/config/my_dashgo_params.yaml
        """
        if yaml_path is None:
            yaml_path = "dashgo/EAI驱动/dashgo_bringup/config/my_dashgo_params.yaml"

        if not os.path.exists(yaml_path):
            print(f"[DashGoROSParams] YAML文件不存在: {yaml_path}，使用默认值")
            print(f"[DashGoROSParams] 默认值: wheel_diameter={cls.wheel_diameter}, "
                  f"wheel_track={cls.wheel_track}")
            return cls()

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                params = yaml.safe_load(f)

            # 提取关键参数
            wheel_diameter = params.get("wheel_diameter", cls.wheel_diameter)
            wheel_track = params.get("wheel_track", cls.wheel_track)
            encoder_resolution = params.get("encoder_resolution", cls.encoder_resolution)

            print(f"[DashGoROSParams] 从YAML加载参数: {yaml_path}")
            print(f"[DashGoROSParams] wheel_diameter={wheel_diameter}, "
                  f"wheel_track={wheel_track}, encoder_resolution={encoder_resolution}")

            return cls(
                wheel_diameter=wheel_diameter,
                wheel_track=wheel_track,
                encoder_resolution=encoder_resolution
            )

        except Exception as e:
            print(f"[DashGoROSParams] 读取YAML文件失败: {e}，使用默认值")
            return cls()

    @property
    def wheel_radius(self) -> float:
        """获取轮子半径（米）"""
        return self.wheel_diameter / 2.0


# 测试代码
if __name__ == "__main__":
    # 测试参数加载
    params = DashGoROSParams.from_yaml()
    print(f"\n[测试] DashGo ROS参数:")
    print(f"  轮子直径: {params.wheel_diameter} m")
    print(f"  轮子半径: {params.wheel_radius} m")
    print(f"  轮距: {params.wheel_track} m")
    print(f"  编码器: {params.encoder_resolution} ticks/转")
