import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, Camera, ContactSensor, ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg, EventTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab.utils.math import wrap_to_pi, quat_apply_inverse, euler_xyz_from_quat, quat_from_euler_xyz
from dashgo_assets import DASHGO_D1_CFG

# =============================================================================
# 1. 自定义动作类 (Action Wrapper) - 保持不变
# =============================================================================

class UniDiffDriveAction(mdp.actions.JointVelocityAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.wheel_radius = 0.0632
        self.track_width = 0.342
        self.prev_actions = None
        self.max_accel_lin = 1.0
        self.max_accel_ang = 0.6

    def process_actions(self, actions: torch.Tensor, *args, **kwargs):
        # 对齐ROS速度限制
        max_lin_vel = 0.3
        max_ang_vel = 1.0

        # 速度裁剪
        target_v = torch.clamp(actions[:, 0] * max_lin_vel, -max_lin_vel, max_lin_vel)
        target_w = torch.clamp(actions[:, 1] * max_ang_vel, -max_ang_vel, max_ang_vel)

        # 加速度平滑
        if self.prev_actions is not None:
            dt = 0.1
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
        max_wheel_vel = 5.0
        v_left = torch.clamp(v_left, -max_wheel_vel, max_wheel_vel)
        v_right = torch.clamp(v_right, -max_wheel_vel, max_wheel_vel)

        joint_actions = torch.stack([v_left, v_right], dim=-1)
        return super().process_actions(joint_actions, *args, **kwargs)

# =============================================================================
# 2. 观测处理 (Observation) - 包含 NaN 清洗
# =============================================================================

def obs_target_polar(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    target_pos_w = env.command_manager.get_command(command_name)[:, :3]
    robot = env.scene[asset_cfg.name]
    
    # 物理数据强力清洗
    robot_pos = torch.nan_to_num(robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    
    # [架构师修复] 严格的 2D 距离计算，忽略 Z 轴差异
    delta_pos_w = target_pos_w[:, :2] - robot_pos[:, :2]
    dist = torch.norm(delta_pos_w, dim=-1, keepdim=True)
    dist = torch.clamp(dist, max=50.0) # 截断
    
    target_angle = torch.atan2(delta_pos_w[:, 1], delta_pos_w[:, 0])
    _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    robot_yaw = torch.nan_to_num(robot_yaw, nan=0.0, posinf=0.0, neginf=0.0)
    
    angle_error = wrap_to_pi(target_angle - robot_yaw).unsqueeze(-1)
    
    obs = torch.cat([dist, angle_error], dim=-1)
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

def _get_corrected_depth(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: Camera = env.scene[sensor_cfg.name]
    if sensor.data.output["distance_to_image_plane"] is None:
        return torch.zeros((env.num_envs, 180), device=env.device)

    depth = sensor.data.output["distance_to_image_plane"].view(sensor.data.output["distance_to_image_plane"].shape[0], -1)
    _, width = depth.shape
    
    if not hasattr(env, "_lidar_correction_factor") or env._lidar_correction_factor.shape[0] != width:
        f_px = width * 4.0 / 20.955 
        u = torch.arange(width, device=env.device).float() - (width / 2) + 0.5
        theta = torch.atan(u / f_px)
        env._lidar_correction_factor = 1.0 / torch.cos(theta)
    
    depth_radial = depth * env._lidar_correction_factor
    depth_radial = torch.nan_to_num(depth_radial, posinf=6.0, neginf=6.0, nan=6.0)
    depth_radial = torch.clamp(depth_radial, min=0.0, max=6.0)
    return depth_radial

def process_lidar_ranges(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    depth_radial = _get_corrected_depth(env, sensor_cfg)
    num_sectors = 36
    batch_size, width = depth_radial.shape
    if width % num_sectors == 0:
        depth_sectors = depth_radial.view(batch_size, num_sectors, -1).min(dim=2)[0]
    else:
        depth_sectors = depth_radial
    return depth_sectors

# =============================================================================
# 3. 奖励函数 (包含 Goal Fixing 和 NaN 清洗)
# =============================================================================

def reward_navigation_sota(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
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
    reward_progress = 1.0 * forward_vel * torch.cos(angle_error)
    
    # [2] 极速奖励
    is_facing_target = torch.cos(angle_error) > 0.8
    reward_high_speed = (forward_vel > 0.25).float() * is_facing_target.float() * 0.2

    # [3] 倒车惩罚
    reward_backward = -0.05 * torch.abs(torch.min(forward_vel, torch.zeros_like(forward_vel)))

    # [4] 避障惩罚
    depth_radial = _get_corrected_depth(env, sensor_cfg)
    min_dist = torch.min(depth_radial, dim=-1)[0]
    safe_dist = 0.55
    reward_collision = torch.zeros_like(min_dist)
    mask_danger = min_dist < safe_dist
    reward_collision[mask_danger] = -0.5 * torch.exp(4.0 * (safe_dist - min_dist[mask_danger]))

    # [5] 动作平滑 (移除，改为单独项并降低权重)
    # reward_rot = -0.05 * torch.abs(ang_vel)**2 

    total_reward = reward_progress + reward_high_speed + reward_backward + reward_collision # + reward_rot
    return torch.clamp(torch.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

# [架构师重构] 基于势能差的引导奖励
def reward_distance_tracking_potential(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    target_pos = env.command_manager.get_command(command_name)[:, :2]
    robot_pos = torch.nan_to_num(env.scene[asset_cfg.name].data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    current_dist = torch.norm(target_pos - robot_pos, dim=-1)
    
    delta_pos = target_pos - robot_pos
    dist_vec = delta_pos / (current_dist.unsqueeze(-1) + 1e-6) 
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
    return 0.5 * torch.exp(-torch.abs(angle_error) / 0.5)

# [架构师新增] 生存惩罚：逼迫机器人动起来
def reward_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    # 返回 -1.0 * 权重
    return -1.0 * torch.ones(env.num_envs, device=env.device)

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
    target_pos = env.command_manager.get_command(command_name)[:, :2]
    robot_pos = torch.nan_to_num(env.scene[asset_cfg.name].data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(target_pos - robot_pos, dim=-1)
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
    target_pos = env.command_manager.get_command(command_name)[:, :2]
    # [架构师修复] 严格 2D 距离比较
    robot_pos = torch.nan_to_num(env.scene[asset_cfg.name].data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(target_pos - robot_pos, dim=-1)
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
        self.min_dist = 1.0 
        self.max_dist = 2.0 
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
    ranges: mdp.UniformPoseCommandCfg.Ranges = mdp.UniformPoseCommandCfg.Ranges(
        pos_x=(-1.0, 1.0), pos_y=(-1.0, 1.0), pos_z=(0.0, 0.0),
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
        lidar = ObservationTermCfg(func=process_lidar_ranges, params={"sensor_cfg": SceneEntityCfg("lidar_sensor")})
        target_polar = ObservationTermCfg(func=obs_target_polar, params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")})
        lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        last_action = ObservationTermCfg(func=mdp.last_action)
    policy = PolicyCfg()

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

@configclass
class DashgoSceneV2Cfg(InteractiveSceneCfg):
    terrain = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot = DASHGO_D1_CFG.replace(prim_path="{ENV_REGEX_NS}/Dashgo")
    
    contact_forces_base = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Dashgo/base_link", 
        history_length=3, track_air_time=True
    )
    
    lidar_sensor = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Dashgo/base_link/lidar_cam", 
        update_period=0.0664, 
        height=1, width=180, 
        data_types=["distance_to_image_plane"], 
        spawn=sim_utils.PinholeCameraCfg(focal_length=4.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)), 
        offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.2), rot=(0.5, -0.5, 0.5, -0.5))
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
    # 现有的行为奖励
    velodyne_style_reward = RewardTermCfg(
        func=reward_navigation_sota, 
        weight=1.0, 
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "sensor_cfg": SceneEntityCfg("lidar_sensor"),
            "command_name": "target_pose"
        }
    )
    
    shaping_distance = RewardTermCfg(
        func=reward_distance_tracking_potential,
        weight=1.0, 
        params={
            "command_name": "target_pose",
            "asset_cfg": SceneEntityCfg("robot")
        }
    )
    
    # [架构师优化] 降低平滑度惩罚权重
    # 从 -0.05 降到 -0.01，给机器人起步的勇气
    action_smoothness = RewardTermCfg(
        func=reward_action_smoothness,
        weight=-0.01,
    )
    
    # [架构师新增] 对准奖励
    # 引导机器人原地转向，解决 "转向惩罚 > 原地不动惩罚" 的死锁
    facing_goal = RewardTermCfg(
        func=reward_facing_target,
        weight=0.5,
        params={
            "command_name": "target_pose",
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    # [优化] 速度对齐奖励
    target_speed = RewardTermCfg(
        func=reward_target_speed,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    log_distance = RewardTermCfg(
        func=log_distance_to_goal,
        weight=0.0,
        params={
            "command_name": "target_pose",
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    log_velocity = RewardTermCfg(
        func=log_linear_velocity,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot")
        }
    )
    
    # [架构师关键破局] 生存惩罚
    # 权重为正，函数返回负值，所以是扣分
    alive_penalty = RewardTermCfg(
        func=reward_alive,
        weight=0.1, 
    )

    reach_goal = RewardTermCfg(
        func=reward_near_goal, 
        weight=300.0, 
        params={
            "command_name": "target_pose", 
            "threshold": 0.8, 
            "asset_cfg": SceneEntityCfg("robot")
        }
    )
    
    collision = RewardTermCfg(func=penalty_collision_force, weight=-20.0, params={"sensor_cfg": SceneEntityCfg("contact_forces_base"), "threshold": 150.0})
    out_of_bounds = RewardTermCfg(func=penalty_out_of_bounds, weight=-200.0, params={"threshold": 8.0, "asset_cfg": SceneEntityCfg("robot")})

@configclass
class DashgoTerminationsCfg:
    time_out = TerminationTermCfg(func=check_time_out, time_out=True)
    
    reach_goal = TerminationTermCfg(
        func=check_reach_goal, 
        params={
            "command_name": "target_pose", 
            "threshold": 0.8, 
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

@configclass
class DashgoNavEnvV2Cfg(ManagerBasedRLEnvCfg):
    decimation = 4 
    episode_length_s = 50.0 
    scene = DashgoSceneV2Cfg(num_envs=16, env_spacing=15.0)
    sim = sim_utils.SimulationCfg(dt=1/60, render_interval=10)
    
    actions = DashgoActionsCfg()
    observations = DashgoObservationsCfg()
    commands = DashgoCommandsCfg()
    events = DashgoEventsCfg()
    rewards = DashgoRewardsCfg()
    terminations = DashgoTerminationsCfg()