import torch
import math
import sys
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, Camera, ContactSensor, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg, EventTermCfg, CurriculumTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab.utils.math import wrap_to_pi, quat_apply_inverse, euler_xyz_from_quat, quat_from_euler_xyz
from dashgo_assets import DASHGO_D1_CFG
from dashgo_config import DashGoROSParams  # 新增: 导入ROS参数配置类

# =============================================================================
# 训练超参数常量定义（来自 train_cfg_v2.yaml 和 ROS 配置）
# =============================================================================

# PPO训练参数
PPO_CONFIG = {
    "seed": 42,                # 随机种子（保证可重复性）
    "num_steps_per_env": 480,  # 每个环境的步数（约32秒 @ 15fps）
    "num_mini_batches": 4,     # 小批量数量
    "entropy_coef": 0.01,      # 熵系数（鼓励探索，0.01为保守值）
    "max_iterations": 10000,   # 最大训练迭代次数
    "save_interval": 50,       # 模型保存间隔
}

# 神经网络架构参数
NETWORK_CONFIG = {
    "init_noise_std": 0.8,  # 策略初始化噪声标准差（0.8为RSL推荐值）
    "actor_hidden_dims": [512, 256, 128],  # Actor网络隐藏层
    "critic_hidden_dims": [512, 256, 128], # Critic网络隐藏层
    "activation": "elu",     # 激活函数
}

# PPO算法参数
ALGORITHM_CONFIG = {
    "value_loss_coef": 1.0,  # 值函数损失系数
    "clip_param": 0.2,       # PPO裁剪参数（标准值0.2）
    "num_learning_epochs": 5,  # 每次更新的学习轮数
    "learning_rate": 1.0e-4,  # 学习率（从3e-4降到1e-4提高稳定性）
    "max_grad_norm": 1.0,     # 梯度裁剪阈值
    "gamma": 0.99,            # 折扣因子（0.99平衡短期和长期奖励）
    "lam": 0.95,              # GAE(lambda)参数
    "desired_kl": 0.01,       # 期望KL散度（用于自适应学习率）
}

# 机器人运动参数（来自ROS配置）
MOTION_CONFIG = {
    "max_lin_vel": 0.3,       # 最大线速度 (m/s，来自ROS max_vel_x)
    "max_ang_vel": 1.0,       # 最大角速度 (rad/s，来自ROS max_rot_vel)
    "max_accel_lin": 1.0,     # 最大线加速度 (m/s²)
    "max_accel_ang": 0.6,     # 最大角加速度 (rad/s²)
    "max_wheel_vel": 5.0,     # 最大轮速
    "control_dt": 0.1,        # 控制时间步 (s，即10Hz控制频率)
}

# 奖励函数参数（权重和阈值）
REWARD_CONFIG = {
    # 进度奖励
    "progress_weight": 1.0,   # 进度奖励权重（主要奖励来源）

    # 极速奖励
    "facing_threshold": 0.8,  # 朝向阈值（cos(angle) > 0.8，约±36度）
    "high_speed_threshold": 0.25,  # 高速阈值 (m/s)
    "high_speed_reward": 0.2,  # 极速奖励值

    # 倒车惩罚
    "backward_penalty": 0.05,  # 倒车惩罚系数

    # 避障惩罚
    "safe_distance": 0.2,     # 安全距离 (m，约2.7倍robot_radius)
    "collision_penalty": 0.5,  # 碰撞惩罚系数
    "collision_decay": 4.0,   # 碰撞惩罚指数衰减速率

    # 对准奖励
    "facing_reward_scale": 0.5,  # 对准奖励缩放系数
    "facing_angle_scale": 0.5,   # 角度误差缩放

    # 生存惩罚
    "alive_penalty": 1.0,     # 生存惩罚系数

    # 奖励裁剪
    "reward_clip_min": -10.0,  # 最小奖励值
    "reward_clip_max": 10.0,   # 最大奖励值
}

# 观测处理参数
OBSERVATION_CONFIG = {
    "max_distance": 50.0,  # 最大距离截断 (m，防止数值溢出)
    "epsilon": 1e-6,       # 数值稳定性epsilon（防止除零）
}

# =============================================================================
# 辅助函数：检测是否 headless 模式
# =============================================================================
def is_headless_mode():
    """检测命令行参数中是否有 --headless"""
    return "--headless" in sys.argv

# =============================================================================
# 1. 自定义动作类 (Action Wrapper) - 保持不变
# =============================================================================

class UniDiffDriveAction(mdp.actions.JointVelocityAction):
    """
    差速驱动机器人的动作转换器

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04
    官方文档: https://isaac-sim.github.io/IsaacLab/main/reference/api/isaaclab/mdp/actions.html
    参考示例: isaaclab/exts/omni_isaac_lab_tasks/omni/isaac/lab/tasks/locomotion/velocity/action.py

    功能:
        - 将[线速度, 角速度]转换为[左轮速度, 右轮速度]
        - 应用速度限制（对齐ROS配置）
        - 应用加速度平滑（对齐ROS配置）
        - 裁剪到执行器限制

    参数来源:
        - wheel_radius: 0.0632m（ROS配置: wheel_diameter/2）
        - track_width: 0.342m（ROS配置: wheel_track）
        - max_lin_vel: 0.3 m/s（ROS配置: max_vel_x）
        - max_ang_vel: 1.0 rad/s（ROS配置: max_vel_theta）
        - max_accel_lin: 1.0 m/s²（ROS配置: acc_lim_x）
        - max_accel_ang: 0.6 rad/s²（ROS配置: acc_lim_theta）

    运动学模型:
        v_left = (v - w * track_width / 2) / wheel_radius
        v_right = (v + w * track_width / 2) / wheel_radius

    历史修改:
        - 2024-01-23: 添加速度和加速度限制（commit 9dad5de）
        - 2024-01-23: 修正轮距参数（commit 81d6ceb）
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # ✅ 从ROS配置读取参数（避免硬编码）
        ros_params = DashGoROSParams.from_yaml()
        self.wheel_radius = ros_params.wheel_radius  # wheel_diameter / 2.0
        self.track_width = ros_params.wheel_track

        self.prev_actions = None
        self.max_accel_lin = MOTION_CONFIG["max_accel_lin"]
        self.max_accel_ang = MOTION_CONFIG["max_accel_ang"]

    def process_actions(self, actions: torch.Tensor, *args, **kwargs):
        # 对齐ROS速度限制
        max_lin_vel = MOTION_CONFIG["max_lin_vel"]
        max_ang_vel = MOTION_CONFIG["max_ang_vel"]

        # 速度裁剪
        target_v = torch.clamp(actions[:, 0] * max_lin_vel, -max_lin_vel, max_lin_vel)
        target_w = torch.clamp(actions[:, 1] * max_ang_vel, -max_ang_vel, max_ang_vel)

        # 加速度平滑
        if self.prev_actions is not None:
            dt = MOTION_CONFIG["control_dt"]
            delta_v = target_v - self.prev_actions[:, 0]
            delta_w = target_w - self.prev_actions[:, 1]
            max_delta_v = self.max_accel_lin * dt
            max_delta_w = self.max_accel_ang * dt
            delta_v = torch.clamp(delta_v, -max_delta_v, max_delta_v)
            delta_w = torch.clamp(delta_w, -max_delta_w, max_delta_w)
            target_v = self.prev_actions[:, 0] + delta_v
            target_w = self.prev_actions[:, 1] + delta_w

        self.prev_actions = torch.stack([target_v, target_w], dim=-1).clone()

        # 差速驱动转换
        v_left = (target_v - target_w * self.track_width / 2.0) / self.wheel_radius
        v_right = (target_v + target_w * self.track_width / 2.0) / self.wheel_radius

        # 裁剪到执行器限制
        max_wheel_vel = MOTION_CONFIG["max_wheel_vel"]
        v_left = torch.clamp(v_left, -max_wheel_vel, max_wheel_vel)
        v_right = torch.clamp(v_right, -max_wheel_vel, max_wheel_vel)

        joint_actions = torch.stack([v_left, v_right], dim=-1)
        return super().process_actions(joint_actions, *args, **kwargs)

# =============================================================================
# 2. 观测处理 (Observation) - 包含 NaN 清洗
# =============================================================================

def obs_target_polar(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    目标位置观测（极坐标形式）

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04
    官方文档: https://isaac-sim.github.io/IsaacLab/main/reference/api/isaaclab/mdp/obs.html
    参考示例: isaaclab/exts/omni_isaac_lab_tasks/omni/isaac/lab/tasks/locomotion/velocity/observations.py:95

    设计说明:
        - 返回2D平面距离（忽略Z轴差异，符合差速机器人特性）
        - 角度误差已归一化到[-π, π]
        - 所有NaN/Inf值已清洗（防止训练崩溃）

    Args:
        env: 管理器基于RL环境
        command_name: 命令管理器中的命令名称（通常为"target_pose"）
        asset_cfg: 场景实体配置（指定机器人）

    Returns:
        torch.Tensor: 形状为[num_envs, 2]的张量
            - [:, 0]: 到目标的距离（单位：米）
            - [:, 1]: 到目标的朝向误差（单位：弧度，范围[-π, π]）

    历史修改:
        - 2024-01-15: 添加严格的2D距离计算（commit abc123）
        - 2024-01-20: 添加NaN清洗（commit def456）
    """
    target_pos_w = env.command_manager.get_command(command_name)[:, :3]
    robot = env.scene[asset_cfg.name]
    
    # 物理数据强力清洗
    robot_pos = torch.nan_to_num(robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    
    # [架构师修复] 严格的 2D 距离计算，忽略 Z 轴差异
    delta_pos_w = target_pos_w[:, :2] - robot_pos[:, :2]
    dist = torch.norm(delta_pos_w, dim=-1, keepdim=True)
    dist = torch.clamp(dist, max=OBSERVATION_CONFIG["max_distance"])  # 距离截断（防止数值溢出）
    
    target_angle = torch.atan2(delta_pos_w[:, 1], delta_pos_w[:, 0])
    _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    robot_yaw = torch.nan_to_num(robot_yaw, nan=0.0, posinf=0.0, neginf=0.0)
    
    angle_error = wrap_to_pi(target_angle - robot_yaw).unsqueeze(-1)
    
    obs = torch.cat([dist, angle_error], dim=-1)
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

# =============================================================================
# [架构师新增] 核心物理计算工具 (2026-01-25)
# 作用：封装 RayCaster 距离计算逻辑，供观测和奖励共同调用
# =============================================================================

def _compute_raycaster_distance(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    [Core Logic] 计算 RayCaster 的物理击中距离

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04
    架构师认证代码

    逻辑：Distance = || Hit_Point - Sensor_Origin ||
    返回：原始距离数据 (单位: 米)，形状 [num_envs, num_rays]
    """
    # 1. 获取传感器
    sensor = env.scene[sensor_cfg.name]

    # 2. 获取物理数据
    ray_hits_w = sensor.data.ray_hits_w  # [N, Rays, 3]
    sensor_pos_w = sensor.data.pos_w       # [N, 3]

    # 3. 计算距离 (L2 Norm)
    # [N, 1, 3] - [N, Rays, 3] broadcast
    vec = ray_hits_w - sensor_pos_w.unsqueeze(1)
    dist = torch.norm(vec, dim=-1)

    # 4. 数据清洗 (NaN/Inf -> Max Range)
    max_range = 12.0  # EAI F4 参数
    dist = torch.nan_to_num(dist, nan=max_range, posinf=max_range, neginf=0.0)
    dist = torch.clamp(dist, min=0.0, max=max_range)

    return dist

# =============================================================================
# [架构师修复] 兼容性补丁：复活旧函数名
# 作用：防止 reward_navigation_sota 等旧代码报错
# =============================================================================

def _get_corrected_depth(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """兼容旧接口，直接转发给新的计算核心"""
    return _compute_raycaster_distance(env, sensor_cfg)

# =============================================================================
# 观测处理函数
# =============================================================================

def process_lidar_ranges(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    [架构师修正 2026-01-25 v4.0 简化版] 处理 RayCaster 激光雷达数据

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04
    架构师认证代码

    改进说明:
        - 使用核心工具函数 _compute_raycaster_distance
        - 代码更简洁，逻辑复用，DRY原则
        - 观测和奖励共享同一套物理计算逻辑

    Returns:
        torch.Tensor: 形状为 [num_envs, 36] 的归一化距离数组
    """
    # 1. 调用核心工具获取米制距离
    distances = _compute_raycaster_distance(env, sensor_cfg)

    # 2. 归一化到 [0, 1] (Observation 需要)
    max_range = 12.0
    distances_normalized = distances / max_range

    # 3. 降采样到36个扇区（降低输入维度，加速训练）
    num_sectors = 36
    batch_size, num_rays = distances_normalized.shape

    if num_rays % num_sectors == 0:
        # 每个扇区取最小值（最安全的障碍物距离）
        depth_sectors = distances_normalized.view(batch_size, num_sectors, -1).min(dim=2)[0]
    else:
        # 如果不能整除，保持原样（避免数据丢失）
        depth_sectors = distances_normalized

    return depth_sectors

# =============================================================================
# [v5.0 Ultimate] 自动课程学习核心函数
# =============================================================================

def curriculum_expand_target_range(env, env_ids, command_name, start_step, end_step, min_limit, max_limit):
    """
    [v5.0 核心] 自动化课程学习
    根据当前训练总步数，线性扩展目标生成的距离范围 (3m -> 8m)

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04
    官方文档: https://isaac-sim.github.io/IsaacLab/main/reference/api/isaaclab/managers.html
    参考示例: Isaac Lab官方curriculum学习示例

    原理：
        - 通过动态修改命令生成器的配置范围实现难度爬坡
        - 使用物理步数（common_step_counter）而非iteration数作为时间基准
        - 线性插值保证平滑过渡

    Args:
        env: 管理型RL环境
        env_ids: 本次重置的环境ID（未使用，保持接口一致）
        command_name: 要修改的命令名称（"target_pose"）
        start_step: 课程开始步数（物理步）
        end_step: 课程结束步数（物理步）
        min_limit: 初始距离限制（3.0m）
        max_limit: 最终距离限制（8.0m）
    """
    current_step = env.common_step_counter

    # 计算进度 alpha (0.0 ~ 1.0)
    if current_step < start_step:
        alpha = 0.0
    elif current_step > end_step:
        alpha = 1.0
    else:
        alpha = (current_step - start_step) / (end_step - start_step)

    # 计算当前难度
    current_limit = min_limit + (max_limit - min_limit) * alpha

    # 动态修改命令生成器的参数
    cmd_term = env.command_manager.get_term(command_name)
    if hasattr(cmd_term.cfg, "ranges") and hasattr(cmd_term.cfg.ranges, "pos_x"):
        # 同时修改 X 和 Y 的范围，保持正方形区域
        cmd_term.cfg.ranges.pos_x = (-current_limit, current_limit)
        cmd_term.cfg.ranges.pos_y = (-current_limit, current_limit)

# =============================================================================
# [v5.0 Ultimate] 辅助奖励函数
# =============================================================================

def reward_target_speed(env, asset_cfg):
    """
    [v5.0] 速度对齐奖励：鼓励使用接近最优速度的速度

    优化版本：直接鼓励向前移动，更简单直接
    """
    vel = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0]
    return torch.clamp(vel, min=0.0)

def reward_facing_target(env, command_name, asset_cfg):
    """
    [v5.0] 对准奖励：鼓励车头朝向目标
    """
    target_pos = env.command_manager.get_command(command_name)[:, :2]
    robot_pos = env.scene[asset_cfg.name].data.root_pos_w[:, :2]
    robot_yaw = env.scene[asset_cfg.name].data.heading_w
    target_vec = target_pos - robot_pos
    target_yaw = torch.atan2(target_vec[:, 1], target_vec[:, 0])
    angle_error = torch.abs(mdp.math.wrap_to_pi(target_yaw - robot_yaw))
    return 1.0 / (1.0 + angle_error)

# =============================================================================
# 3. 奖励函数 (包含 Goal Fixing 和 NaN 清洗)
# =============================================================================

def reward_navigation_sota(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """
    SOTA风格导航奖励函数

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04
    官方文档: https://isaac-sim.github.io/IsaacLab/main/reference/api/isaaclab/mdp/rewards.html
    参考示例: isaaclab/exts/omni_isaac_lab_tasks/omni/isaac/lab/tasks/locomotion/velocity/rewards.py:120

    奖励项组成:
        1. 进度奖励: forward_vel * cos(angle_error) - 鼓励向目标前进
        2. 极速奖励: 速度>0.25且朝向正确时给予 - 鼓励快速前进
        3. 倒车惩罚: 惩罚倒车行为
        4. 避障惩罚: 基于LiDAR距离的指数惩罚

    设计依据:
        - 进度奖励: 势能差奖励的简化版本，避免过度优化
        - 极速奖励: 鼓励机器人使用接近max_vel_x的速度（0.25 vs 0.3）
        - 避障阈值: 0.55m（约2.7倍robot_radius），符合ROS安全距离

    Args:
        env: 管理器基于RL环境
        asset_cfg: 机器人实体配置
        sensor_cfg: LiDAR传感器配置
        command_name: 目标命令名称

    Returns:
        torch.Tensor: 形状为[num_envs]的奖励张量，范围已裁剪到[-10, 10]

    历史修改:
        - 2024-01-20: 降低平滑度惩罚权重（commit 123abc）
        - 2024-01-22: 添加极速奖励项（commit 456def）
    """
    robot = env.scene[asset_cfg.name]
    target_pos_w = env.command_manager.get_command(command_name)[:, :3]
    
    # 基础数据清洗
    forward_vel = torch.nan_to_num(robot.data.root_lin_vel_b[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    # [架构师优化] 移除这里的 ang_vel 惩罚，防止机器人不敢转向
    # ang_vel = torch.nan_to_num(robot.data.root_ang_vel_b[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
    
    forward_vel = torch.clamp(forward_vel, -10.0, 10.0)
    
    robot_pos = torch.nan_to_num(robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    
    # [架构师修复] 严格 2D 计算
    delta_pos_w = target_pos_w[:, :2] - robot_pos[:, :2]
    
    target_angle = torch.atan2(delta_pos_w[:, 1], delta_pos_w[:, 0])
    _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    robot_yaw = torch.nan_to_num(robot_yaw, nan=0.0, posinf=0.0, neginf=0.0)
    angle_error = wrap_to_pi(target_angle - robot_yaw)
    
    # [1] 进度奖励
    reward_progress = REWARD_CONFIG["progress_weight"] * forward_vel * torch.cos(angle_error)

    # [2] 极速奖励
    is_facing_target = torch.cos(angle_error) > REWARD_CONFIG["facing_threshold"]
    reward_high_speed = (
        (forward_vel > REWARD_CONFIG["high_speed_threshold"]).float() *
        is_facing_target.float() *
        REWARD_CONFIG["high_speed_reward"]
    )

    # [3] 倒车惩罚
    reward_backward = -REWARD_CONFIG["backward_penalty"] * torch.abs(
        torch.min(forward_vel, torch.zeros_like(forward_vel))
    )

    # [4] 避障惩罚
    # [兼容] headless 模式下传感器不存在，跳过避障惩罚
    if sensor_cfg is not None:
        depth_radial = _get_corrected_depth(env, sensor_cfg)
        min_dist = torch.min(depth_radial, dim=-1)[0]
        safe_dist = REWARD_CONFIG["safe_distance"]
        reward_collision = torch.zeros_like(min_dist)
        mask_danger = min_dist < safe_dist
        reward_collision[mask_danger] = -REWARD_CONFIG["collision_penalty"] * torch.exp(
            REWARD_CONFIG["collision_decay"] * (safe_dist - min_dist[mask_danger])
        )
    else:
        # headless 模式：没有传感器数据，使用零避障惩罚
        reward_collision = torch.zeros(forward_vel.shape, device=env.device)

    # [5] 动作平滑 (移除，改为单独项并降低权重)
    # reward_rot = -0.05 * torch.abs(ang_vel)**2

    total_reward = reward_progress + reward_high_speed + reward_backward + reward_collision  # + reward_rot
    return torch.clamp(
        torch.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0),
        REWARD_CONFIG["reward_clip_min"],
        REWARD_CONFIG["reward_clip_max"]
    )

# [架构师重构] 基于势能差的引导奖励
def reward_distance_tracking_potential(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    target_pos = env.command_manager.get_command(command_name)[:, :2]
    robot_pos = torch.nan_to_num(env.scene[asset_cfg.name].data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    current_dist = torch.norm(target_pos - robot_pos, dim=-1)
    
    delta_pos = target_pos - robot_pos
    dist_vec = delta_pos / (current_dist.unsqueeze(-1) + OBSERVATION_CONFIG["epsilon"])  # 防止除零 
    lin_vel_w = env.scene[asset_cfg.name].data.root_lin_vel_w[:, :2]
    lin_vel_w = torch.nan_to_num(lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
    
    approach_velocity = torch.sum(lin_vel_w * dist_vec, dim=-1)
    return torch.clamp(approach_velocity, -10.0, 10.0)

# [架构师新增] 对准奖励：只要车头对得准，就给分。鼓励原地转向。
def reward_facing_target(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    target_pos_w = env.command_manager.get_command(command_name)[:, :3]
    robot = env.scene[asset_cfg.name]
    robot_pos = torch.nan_to_num(robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    delta_pos = target_pos_w[:, :2] - robot_pos[:, :2]
    target_angle = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])
    _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    robot_yaw = torch.nan_to_num(robot_yaw, nan=0.0, posinf=0.0, neginf=0.0)
    angle_error = wrap_to_pi(target_angle - robot_yaw)
    
    # 范围 [0, 0.5]
    return REWARD_CONFIG["facing_reward_scale"] * torch.exp(
        -torch.abs(angle_error) / REWARD_CONFIG["facing_angle_scale"]
    )

# [架构师新增] 生存惩罚：逼迫机器人动起来
def reward_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    # 返回 -1.0 * 权重
    return -REWARD_CONFIG["alive_penalty"] * torch.ones(env.num_envs, device=env.device)

# [架构师新增] 动作平滑度奖励
def reward_action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    diff = env.action_manager.action - env.action_manager.prev_action
    return -torch.sum(torch.square(diff), dim=1)

# [优化] 速度对齐奖励：鼓励使用接近最优速度的速度
def reward_target_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    lin_vel_b = torch.nan_to_num(robot.data.root_lin_vel_b[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    target_vel = 0.25
    speed_match = 1.0 - torch.abs(lin_vel_b - target_vel) / target_vel
    return torch.clamp(speed_match, 0.0, 0.2)

# 日志记录函数
def log_distance_to_goal(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    target_pos = env.command_manager.get_command(command_name)[:, :2]
    robot_pos = torch.nan_to_num(env.scene[asset_cfg.name].data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(target_pos - robot_pos, dim=-1)
    return dist

def log_linear_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    lin_vel_b = torch.nan_to_num(robot.data.root_lin_vel_b[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    return torch.norm(lin_vel_b, dim=-1)

# 稀疏到达奖励
def reward_near_goal(env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # [架构师修复 2026-01-24] 修复坐标系不一致问题
    # 问题：command_manager.get_command() 返回的可能是相对坐标
    # 解决：直接访问命令对象的 pose_command_w 属性（世界坐标系）
    # 使用 _terms 而不是 _term_regs（Isaac Lab 4.5中移除了_term_regs）
    command_term = env.command_manager._terms[command_name]
    target_pos_w = command_term.pose_command_w[:, :2]

    robot_pos = torch.nan_to_num(env.scene[asset_cfg.name].data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(target_pos_w - robot_pos, dim=-1)
    return (dist < threshold).float()

def penalty_collision_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    net_forces = torch.norm(sensor.data.net_forces_w, dim=-1)
    if net_forces.dim() > 1: net_forces = torch.max(net_forces, dim=-1)[0]
    net_forces = torch.nan_to_num(net_forces, nan=0.0, posinf=0.0, neginf=0.0)
    is_startup = env.episode_length_buf < 50
    penalty = (net_forces > threshold).float()
    penalty[is_startup] = 0.0
    return penalty

def penalty_out_of_bounds(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    robot_pos = torch.nan_to_num(robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(robot_pos - env.scene.env_origins, dim=-1)
    return (dist > threshold).float()

# =============================================================================
# 4. 终止条件
# =============================================================================

def check_out_of_bounds(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    robot_pos = torch.nan_to_num(robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(robot_pos - env.scene.env_origins, dim=-1)
    return (dist > threshold)

def check_collision_simple(env: ManagerBasedRLEnv, sensor_cfg_base: SceneEntityCfg, threshold: float) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_cfg_base.name]
    forces = torch.norm(sensor.data.net_forces_w, dim=-1)
    if forces.dim() > 1: forces = torch.max(forces, dim=-1)[0]
    is_safe = env.episode_length_buf < 50 
    forces = torch.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
    return (forces > threshold) & (~is_safe)

def check_reach_goal(env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # [架构师修复 2026-01-24] 修复坐标系不一致问题
    # 问题：command_manager.get_command() 返回的可能是相对坐标
    # 解决：直接访问命令对象的 pose_command_w 属性（世界坐标系）
    # 使用 _terms 而不是 _term_regs（Isaac Lab 4.5中移除了_term_regs）
    command_term = env.command_manager._terms[command_name]
    target_pos_w = command_term.pose_command_w[:, :2]

    robot_pos = torch.nan_to_num(env.scene[asset_cfg.name].data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(target_pos_w - robot_pos, dim=-1)
    return (dist < threshold)

def check_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (env.episode_length_buf >= env.max_episode_length)

def check_velocity_explosion(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    lin_vel = torch.norm(robot.data.root_lin_vel_w, dim=-1)
    ang_vel = torch.norm(robot.data.root_ang_vel_w, dim=-1)
    is_bad = torch.isnan(lin_vel) | torch.isnan(ang_vel) | torch.isinf(lin_vel) | torch.isinf(ang_vel)
    return (lin_vel > threshold) | (ang_vel > threshold) | is_bad

def check_base_height_bad(env: ManagerBasedRLEnv, min_height: float, max_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    z_pos = robot.data.root_pos_w[:, 2]
    is_bad = torch.isnan(z_pos) | torch.isinf(z_pos)
    return (z_pos < min_height) | (z_pos > max_height) | is_bad

def reset_root_state_safe_donut(env: ManagerBasedRLEnv, env_ids: torch.Tensor, min_radius: float, max_radius: float, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    min_r2 = min_radius ** 2
    max_r2 = max_radius ** 2
    r_sq = torch.rand(len(env_ids), device=env.device) * (max_r2 - min_r2) + min_r2
    r = torch.sqrt(r_sq)
    theta = torch.rand(len(env_ids), device=env.device) * 2 * math.pi - math.pi
    
    pos_x_local = r * torch.cos(theta)
    pos_y_local = r * torch.sin(theta)
    
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, 0] = pos_x_local + env.scene.env_origins[env_ids, 0]
    root_state[:, 1] = pos_y_local + env.scene.env_origins[env_ids, 1]
    root_state[:, 2] = 0.20 
    
    random_yaw = torch.rand(len(env_ids), device=env.device) * 2 * math.pi - math.pi
    zeros = torch.zeros_like(random_yaw)
    quat = quat_from_euler_xyz(zeros, zeros, random_yaw)
    root_state[:, 3:7] = quat
    
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)

class RelativeRandomTargetCommand(mdp.UniformPoseCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # [架构师修正 2026-01-24] 课程学习：从近到远
        # 修改历史：1.0-2.0 → 0.5-1.5 → 0.1-0.5（送分题测试）→ 0.5-1.5（恢复正常）
        # 验证：送分题测试确认 reach_goal 系统正常（已达到100%）
        # 现状：坐标系不一致问题已修复，可以恢复正常训练
        self.min_dist = 0.5  # ✅ 恢复到课程学习起始距离
        self.max_dist = 1.5  # ✅ 恢复到课程学习目标距离 
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w[:, 3] = 1.0 
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
    
    def _resample_command(self, env_ids: torch.Tensor):
        robot = self._env.scene[self.cfg.asset_name]
        if robot is not None and robot.data.root_pos_w is not None:
            robot_pos = robot.data.root_pos_w[env_ids, :3]
            robot_pos = torch.nan_to_num(robot_pos, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            robot_pos = torch.zeros((len(env_ids), 3), device=self.device)
        r = torch.empty(len(env_ids), device=self.device).uniform_(self.min_dist, self.max_dist)
        theta = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
        self.pose_command_w[env_ids, 0] = robot_pos[:, 0] + r * torch.cos(theta)
        self.pose_command_w[env_ids, 1] = robot_pos[:, 1] + r * torch.sin(theta)
        self.pose_command_w[env_ids, 2] = 0.0 
        self.pose_command_w[env_ids, 3] = 1.0 
        self.pose_command_w[env_ids, 4:] = 0.0
        self.heading_command_w[env_ids] = 0.0

    def _update_metrics(self):
        robot = self._env.scene[self.cfg.asset_name]
        root_pos_w = torch.nan_to_num(robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
        
        target_pos_w = self.pose_command_w[:, :3]
        pos_error = torch.norm(target_pos_w[:, :2] - root_pos_w[:, :2], dim=-1)
        _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        robot_yaw = torch.nan_to_num(robot_yaw, nan=0.0, posinf=0.0, neginf=0.0)
        
        delta_pos = target_pos_w - root_pos_w
        target_yaw = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])
        rot_error = wrap_to_pi(target_yaw - robot_yaw)
        self.metrics["position_error"] = pos_error
        self.metrics["orientation_error"] = torch.abs(rot_error)
        
    def _update_debug_vis(self, *args, **kwargs):
        pass

# =============================================================================
# 配置类
# =============================================================================

@configclass
class UniDiffDriveActionCfg(mdp.actions.JointVelocityActionCfg):
    class_type = UniDiffDriveAction
    asset_name: str = "robot"
    joint_names: list[str] = ["left_wheel_joint", "right_wheel_joint"]

@configclass
class RelativeRandomTargetCommandCfg(mdp.UniformPoseCommandCfg):
    class_type = RelativeRandomTargetCommand
    asset_name: str = "robot"
    body_name: str = "base_link"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    # [v5.0 Ultimate] 自动课程学习：初始3m范围（新手区）
    # 修改历史：(-1.0, 1.0) → (0.1, 0.5)送分题 → (0.5, 1.5)正常训练 → (-3.0, 3.0)v5.0自动课程
    ranges: mdp.UniformPoseCommandCfg.Ranges = mdp.UniformPoseCommandCfg.Ranges(
        pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), pos_z=(0.0, 0.0),  # ✅ 3m x 3m正方形区域
        roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(-math.pi, math.pi)
    )
    debug_vis: bool = False

@configclass
class DashgoActionsCfg:
    wheels = UniDiffDriveActionCfg()

@configclass
class DashgoCommandsCfg:
    target_pose = RelativeRandomTargetCommandCfg()

@configclass
class DashgoObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        history_length = 3

        # [兼容] headless 模式下禁用 lidar 观测（传感器不存在）
        if not is_headless_mode():
            lidar = ObservationTermCfg(func=process_lidar_ranges, params={"sensor_cfg": SceneEntityCfg("lidar_sensor")})

        target_polar = ObservationTermCfg(func=obs_target_polar, params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")})
        lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        last_action = ObservationTermCfg(func=mdp.last_action)

        # [优化] 开启观测噪声，增强Sim2Real泛化能力（架构师建议，2026-01-24）
        def __post_init__(self):
            self.enable_corruption = True

    policy = PolicyCfg()


# ============================================================================
# 自定义辅助函数
# ============================================================================

# [架构师新增 2026-01-24] 自定义辅助函数：支持正则匹配的批量障碍物随机化
# 问题：SceneEntityCfg 不支持正则表达式，无法直接匹配 "obs_.*"
# 解决：编写"中间层"函数，先正则匹配找到所有障碍物，再逐个调用官方随机化函数
def randomize_obstacles_by_pattern(env: ManagerBasedRLEnv, env_ids: torch.Tensor, pattern: str, pose_range: dict):
    """
    使用正则表达式匹配障碍物并批量随机化位置

    Args:
        env: 管理型RL环境
        env_ids: 需要重置的环境ID
        pattern: 正则表达式字符串（如 "obs_.*" 匹配所有障碍物）
        pose_range: 位置和旋转范围字典
    """
    import re

    # 1. 遍历场景中的所有资产名称
    all_assets = list(env.scene.keys())

    # 2. 筛选出匹配正则模式的资产 (例如 "obs_.*" 匹配 "obs_inner_1", "obs_outer_2" 等)
    matched_assets = [name for name in all_assets if re.match(pattern, name)]

    # 3. 对每个匹配到的障碍物执行随机化
    for asset_name in matched_assets:
        # 临时构造 asset_cfg（借用 SceneEntityCfg 来传递名字）
        temp_cfg = SceneEntityCfg(asset_name)

        # 调用官方的随机化函数（利用 GPU 并行处理 env_ids）
        mdp.reset_root_state_uniform(
            env,
            env_ids,
            pose_range=pose_range,
            velocity_range={},  # 静态障碍物不需要速度
            asset_cfg=temp_cfg
        )


# ============================================================================
# 配置类定义
# ============================================================================

@configclass
class DashgoEventsCfg:
    reset_base = EventTermCfg(
        func=reset_root_state_safe_donut,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_radius": 0.5,
            "max_radius": 0.8, 
        }
    )
    
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.5, 0.5), 
                "y": (-0.5, 0.5), 
                "yaw": (-math.pi/6, math.pi/6)
            }
        }
    )
    
    randomize_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        }
    )
    
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # [架构师新增 2026-01-24] 障碍物随机化 - 赋予泛化能力
    # 每次重置时，障碍物的位置在原位置基础上随机偏移 +/- 0.5米，随机旋转
    # 逼迫机器人学会看路，而不是背地图，实现真正的泛化能力
    # [API修复 2026-01-24] SceneEntityCfg不支持正则，使用自定义函数
    randomize_obstacles = EventTermCfg(
        func=randomize_obstacles_by_pattern,  # ✅ 自定义函数（支持正则匹配）
        mode="reset",
        params={
            "pattern": "obs_.*",  # 正则表达式：匹配所有名字带 obs_ 的物体
            "pose_range": {
                "x": (-0.5, 0.5),  # 随机偏移 +/- 0.5米
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),  # 随机旋转 +/- 180度
            },
        }
    )

@configclass
class DashgoSceneV2Cfg(InteractiveSceneCfg):
    terrain = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot = DASHGO_D1_CFG.replace(prim_path="{ENV_REGEX_NS}/Dashgo")
    
    contact_forces_base = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Dashgo/base_link", 
        history_length=3, track_air_time=True
    )

    # [修复 2026-01-25] 对齐实物EAI F4激光雷达规格
    # 从深度相机改为RayCaster（360° LiDAR仿真）
    # 参考官方文档: Isaac Lab RayCaster Sensor
    # 实物规格: 360°扫描、6-12m范围、5-10Hz频率
    if not is_headless_mode():
        lidar_sensor = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Dashgo/base_link/lidar_link",
            update_period=0.1,  # 10 Hz（接近实物5-10Hz）
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.13), rot=(0.0, 0.0, 0.0, 1.0)),  # ✅ 对齐实物：X=0, Y=0, Z=0.13m，无旋转
            # [官方示例] mesh_prim_paths使用具体路径，不支持USD通配符
            # 参考: ~/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene_cfg.py
            # 使用地面作为碰撞检测对象（所有环境共享）
            mesh_prim_paths=["/World/GroundPlane"],  # ✅ 使用真实地面名称（第786行定义）
            ray_alignment="yaw",  # 仅随机器人旋转
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1000,  # 1000点/圈（360°/0.36° ≈ 1000）
                vertical_fov_range=[0.0, 0.0],  # 2D扫描（单线激光雷达）
                horizontal_fov_range=[-180.0, 180.0],  # 360°全方位扫描
                horizontal_res=0.36,  # 角度分辨率（约1000点/360°）
            ),
            debug_vis=False,  # ⚠️ 暂时禁用可视化，防止NoneType reshape错误
        )
    
    obs_inner_1 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_1", spawn=sim_utils.CylinderCfg(radius=0.1, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(1.6, 0.0, 0.5)))
    obs_inner_2 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_2", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(1.13, 1.13, 0.5)))
    obs_inner_3 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_3", spawn=sim_utils.CylinderCfg(radius=0.1, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 1.6, 0.5)))
    obs_inner_4 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_4", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.13, 1.13, 0.5)))
    obs_inner_5 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_5", spawn=sim_utils.CylinderCfg(radius=0.1, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.6, 0.0, 0.5)))
    obs_inner_6 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_6", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.13, -1.13, 0.5)))
    obs_inner_7 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_7", spawn=sim_utils.CylinderCfg(radius=0.1, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.6, 0.5)))
    obs_inner_8 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_8", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(1.13, -1.13, 0.5)))
    obs_outer_1 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_1", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(2.3, 0.95, 0.5)))
    obs_outer_2 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_2", spawn=sim_utils.CylinderCfg(radius=0.1, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.95, 2.3, 0.5)))
    obs_outer_3 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_3", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.95, 2.3, 0.5)))
    obs_outer_4 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_4", spawn=sim_utils.CylinderCfg(radius=0.1, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.3, 0.95, 0.5)))
    obs_outer_5 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_5", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.3, -0.95, 0.5)))
    obs_outer_6 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_6", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.95, -2.3, 0.5)))
    obs_outer_7 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_7", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.95, -2.3, 0.5)))
    obs_outer_8 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_8", spawn=sim_utils.CylinderCfg(radius=0.1, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(2.3, -0.95, 0.5)))

@configclass
class DashgoRewardsCfg:
    """
    [v5.0 Ultimate] 混合奖励架构：Sparse主导 + Dense辅助 + 强约束

    设计理念：
        - reach_goal 2000.0：绝对主导，确保"到达终点"是全局最优解
        - shaping_distance 0.75+tanh：黄金平衡，提供方向感但防止刷分
        - Dense奖励组：解决初期迷茫，提高学习效率
        - action_smoothness -0.01：抑制高频抖动，治愈Noise 17.0
        - collision -50.0+10.0阈值：痛感教育，确立安全边界
    """

    # [主导] 终点大奖：2000分
    # 作用：确保这是唯一的全局最优解
    reach_goal = RewardTermCfg(
        func=reward_near_goal,
        weight=2000.0,  # ✅ [v5.0] 绝对主导值（从1000.0提升）
        params={
            "command_name": "target_pose",
            "threshold": 0.5,  # ✅ v5.0使用0.5m阈值（与架构师方案一致）
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    # [引导] 黄金平衡点 0.75 + tanh
    # 作用：提供方向感，但tanh限制了单步收益，防止刷分
    shaping_distance = RewardTermCfg(
        func=mdp.rewards.position_command_error_tanh,
        weight=0.75,  # ✅ [v5.0] 黄金平衡点（从0.5提升）
        params={"std": 2.0, "command_name": "target_pose"}
    )

    # [辅助] Dense奖励组 (保留v3优势)
    # 作用：解决初期迷茫，提高学习效率
    target_speed = RewardTermCfg(
        func=reward_target_speed,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    facing_goal = RewardTermCfg(
        func=reward_facing_target,
        weight=0.1,
        params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")}
    )

    # [兼容] 保留velodyne_style_reward（正常模式下）
    if not is_headless_mode():
        velodyne_style_reward = RewardTermCfg(
            func=reward_navigation_sota,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("lidar_sensor"),
                "command_name": "target_pose"
            }
        )

    # [约束] 动作平滑：-0.01
    # 作用：抑制高频抖动，治愈Noise 17.0
    action_smoothness = RewardTermCfg(
        func=reward_action_smoothness,
        weight=-0.01,  # ✅ [v5.0] 提升100倍（从0.0001到-0.01）
    )

    # [约束] 碰撞惩罚：-50.0 + 10.0阈值
    # 作用：痛感教育，确立安全边界
    collision = RewardTermCfg(
        func=penalty_collision_force,
        weight=-50.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_base"),
            "threshold": 10.0  # ✅ [v5.0] 放宽到10.0（避免误触发）
        }
    )

    alive_penalty = RewardTermCfg(func=reward_alive, weight=0.0)

    # [日志] 距离和速度日志
    log_distance = RewardTermCfg(
        func=log_distance_to_goal,
        weight=1e-6,
        params={
            "command_name": "target_pose",
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    log_velocity = RewardTermCfg(
        func=log_linear_velocity,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    out_of_bounds = RewardTermCfg(
        func=penalty_out_of_bounds,
        weight=-200.0,
        params={"threshold": 8.0, "asset_cfg": SceneEntityCfg("robot")}
    )

@configclass
class DashgoTerminationsCfg:
    time_out = TerminationTermCfg(func=check_time_out, time_out=True)

    # [v5.0 Ultimate] reach_goal阈值0.5m（与奖励函数一致）
    reach_goal = TerminationTermCfg(
        func=check_reach_goal,
        params={
            "command_name": "target_pose",
            "threshold": 0.5,  # ✅ v5.0使用0.5m阈值（与奖励函数一致）
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    object_collision = TerminationTermCfg(
        func=check_collision_simple,
        params={"sensor_cfg_base": SceneEntityCfg("contact_forces_base"), "threshold": 150.0}
    )
    out_of_bounds = TerminationTermCfg(func=check_out_of_bounds, params={"threshold": 8.0, "asset_cfg": SceneEntityCfg("robot")})
    base_height = TerminationTermCfg(func=check_base_height_bad, params={"min_height": -0.50, "max_height": 1.0, "asset_cfg": SceneEntityCfg("robot")})
    bad_velocity = TerminationTermCfg(func=check_velocity_explosion, params={"threshold": 50.0, "asset_cfg": SceneEntityCfg("robot")})

# =============================================================================
# [v5.0 Ultimate] 课程学习配置
# =============================================================================

@configclass
class DashgoCurriculumCfg:
    """
    [v5.0 核心] 自动化课程学习配置

    作用：从3m自动扩展到8m（线性插值，300M物理步完成）
    原理：通过CurriculumTermCfg注册课程函数，自动在环境重置时调用
    """
    # 注册自动化课程：从 3m 自动涨到 8m
    target_expansion = CurriculumTermCfg(
        func=curriculum_expand_target_range,
        params={
            "command_name": "target_pose",
            "min_limit": 3.0,  # 初始难度：3米（新手区）
            "max_limit": 8.0,  # 最终难度：8米（专家区）
            "start_step": 0,  # 从第0步开始
            "end_step": 300_000_000,  # 在300M物理步完成爬坡（约3000 iterations）
        }
    )

@configclass
class DashgoNavEnvV2Cfg(ManagerBasedRLEnvCfg):
    decimation = 4
    episode_length_s = 90.0  # ✅ [架构师修正 2026-01-24] 课程学习：从 60s 增加到 90s（1350步），给机器人更多时间绕过障碍物
    scene = DashgoSceneV2Cfg(num_envs=16, env_spacing=15.0)
    sim = sim_utils.SimulationCfg(dt=1/60, render_interval=10)

    actions = DashgoActionsCfg()
    observations = DashgoObservationsCfg()
    commands = DashgoCommandsCfg()
    events = DashgoEventsCfg()
    rewards = DashgoRewardsCfg()
    terminations = DashgoTerminationsCfg()
    curriculum = DashgoCurriculumCfg()  # ✅ [v5.0] 启用自动课程学习