# dashgo_assets.py
# 2026-01-24: Isaac Sim Architect Edition - Added Actuators
# [架构师修复 2026-01-24] 修复机器人"瘫痪"问题 - 添加正确的actuators配置

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import os

# 获取当前文件所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 优先查找 config 目录下的 urdf
URDF_PATH = os.path.join(CURRENT_DIR, "config", "dashgo.urdf")
if not os.path.exists(URDF_PATH):
    # 回退到当前目录
    URDF_PATH = os.path.join(CURRENT_DIR, "dashgo.urdf")

# [验证] URDF 关节名称确认：
# - left_wheel_joint (左轮)
# - right_wheel_joint (右轮)
# - front_caster_joint (前脚轮)
# - back_caster_joint (后脚轮)
# - lidar_joint (雷达固定)

DASHGO_D1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=False,
        make_instanceable=False,  # 防止幽灵碰撞
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.1,
            angular_damping=0.1,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),  # 稍微抬高一点防止卡地（原0.2米太高）
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            ".*": 0.0,  # 所有关节归零（正则表达式匹配所有关节）
        },
    ),

    # [架构师修复 2026-01-24] 核心部分：添加电机！
    # 问题：机器人速度为 0 m/s（Episode_Reward/target_speed: 0.0021）
    # 原因：actuators 配置不正确，机器人"瘫痪"了
    # 解决：调整 damping、effort_limit、velocity_limit
    actuators={
        "diff_drive": ImplicitActuatorCfg(
            # [验证] URDF 关节名称：left_wheel_joint, right_wheel_joint
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],

            # [架构师修正] 速度控制模式配置
            # 修改历史：
            # - damping: 5.0 → 100.0 (提高20倍，增强速度跟踪能力)
            # - effort_limit_sim: 20.0 → 20.0 (保持不变，已足够)
            # - velocity_limit_sim: 5.0 → 10.0 (提高2倍，允许更快的速度)

            # 速度控制参数解释：
            # - effort_limit: 最大扭矩 (N·m)，20 N·m 对实物来说足够
            # - velocity_limit: 最大转速 (rad/s)，10 rad/s ≈ 0.64 m/s（轮径0.127m）
            # - stiffness: 速度控制时刚度设为0（柔顺控制）
            # - damping: 增益/阻尼，值越大，速度跟踪越硬

            effort_limit_sim=20.0,     # ✅ 最大扭矩 20 N·m
            velocity_limit_sim=10.0,    # ✅ 最大转速 10 rad/s ≈ 0.64 m/s
            stiffness=0.0,              # ✅ 速度控制时刚度设为0
            damping=100.0,              # ✅ 从 5.0 提高到 100.0（增强速度跟踪）
        ),
    },

    soft_joint_pos_limit_factor=1.0,
)
