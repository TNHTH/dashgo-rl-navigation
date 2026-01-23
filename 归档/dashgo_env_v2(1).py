import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, Camera, ContactSensor, ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg, EventTermCfg, CurriculumTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab.utils.math import wrap_to_pi, quat_apply_inverse, euler_xyz_from_quat

from dashgo_assets import DASHGO_D1_CFG

# --- 自定义观测函数 ---

def process_lidar_ranges(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: Camera = env.scene[sensor_cfg.name]
    depth_data = sensor.data.output["distance_to_image_plane"]
    ranges = depth_data.view(depth_data.shape[0], -1)
    ranges = torch.clamp(ranges, max=10.0)
    return ranges

def process_ultrasonic_data(env: ManagerBasedRLEnv, sensor_names: list) -> torch.Tensor:
    data_list = []
    for name in sensor_names:
        sensor: Camera = env.scene[name]
        d = sensor.data.output["distance_to_image_plane"].view(env.num_envs, 1)
        data_list.append(d)
    ranges = torch.cat(data_list, dim=1)
    ranges = torch.clamp(ranges, max=4.0) 
    return ranges

def process_target_relative_pos(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    target_pos_w = env.command_manager.get_command(command_name)[:, :3]
    robot = env.scene[asset_cfg.name]
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w
    delta_pos_w = target_pos_w - robot_pos_w
    delta_pos_b = quat_apply_inverse(robot_quat_w, delta_pos_w)
    return delta_pos_b[:, :2]

# --- 自定义奖励辅助函数 ---

def progress_toward_target_xy(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    target_pos = env.command_manager.get_command(command_name)[:, :2]
    robot = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_vel = robot.data.root_lin_vel_w[:, :2]
    target_vec = target_pos - robot_pos
    target_dir = target_vec / (torch.norm(target_vec, dim=-1, keepdim=True) + 1e-6)
    return torch.sum(robot_vel * target_dir, dim=-1)

def close_to_target_xy(env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    target_pos = env.command_manager.get_command(command_name)[:, :2]
    robot = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w[:, :2]
    distance = torch.norm(target_pos - robot_pos, dim=-1)
    return distance < threshold

def face_to_target_xy(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    target_pos = env.command_manager.get_command(command_name)[:, :2]
    robot = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_quat = robot.data.root_quat_w
    _, _, robot_yaw = euler_xyz_from_quat(robot_quat)
    target_vec = target_pos - robot_pos
    target_yaw = torch.atan2(target_vec[:, 1], target_vec[:, 0])
    angle_error = wrap_to_pi(target_yaw - robot_yaw)
    return 1.0 / (1.0 + torch.square(angle_error))

def undesired_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces_norm = torch.norm(sensor.data.net_forces_w, dim=-1)
    return torch.any(forces_norm > threshold, dim=-1)

def penalty_stand_still(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_velocity: float) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    lin_vel_norm = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
    return (lin_vel_norm < min_velocity).float()

# --- 1. 场景配置 ---
@configclass
class DashgoSceneV2Cfg(InteractiveSceneCfg):
    terrain = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot = DASHGO_D1_CFG.replace(prim_path="{ENV_REGEX_NS}/Dashgo")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Dashgo/base_link", history_length=3, track_air_time=True)
    
    lidar_sensor = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Dashgo/base_link/lidar_cam",
        update_period=0.1, height=1, width=80,          
        data_types=["distance_to_image_plane"], 
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.2), rot=(0.5, -0.5, 0.5, -0.5)), 
    )

    us_front = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Dashgo/base_link/us_front", update_period=0.1, height=1, width=1, data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.02, 4.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.2, 0.0, 0.1), rot=(0.5, -0.5, 0.5, -0.5)),
    )
    us_left = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Dashgo/base_link/us_left", update_period=0.1, height=1, width=1, data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.02, 4.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.15, 0.1), rot=(0.5, 0.5, 0.5, 0.5)),
    )
    us_right = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Dashgo/base_link/us_right", update_period=0.1, height=1, width=1, data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.02, 4.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, -0.15, 0.1), rot=(-0.5, -0.5, -0.5, 0.5)),
    )
    us_back = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Dashgo/base_link/us_back", update_period=0.1, height=1, width=1, data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.02, 4.0)),
        offset=CameraCfg.OffsetCfg(pos=(-0.2, 0.0, 0.1), rot=(-0.5, -0.5, 0.5, 0.5)),
    )

    goal_marker = AssetBaseCfg(
        prim_path="/World/Visuals/Goal", spawn=sim_utils.SphereCfg(radius=0.2, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)), 
    )
    obstacles_cyl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacles_Cylinder", spawn=sim_utils.CylinderCfg(radius=0.3, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), rigid_props=sim_utils.RigidBodyPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=50.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 2.0, 0.5)),
    )
    obstacles_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacles_Box", spawn=sim_utils.CuboidCfg(size=(0.5, 0.5, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)), rigid_props=sim_utils.RigidBodyPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=50.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.0, -2.0, 0.5)),
    )

# --- 2. 环境主配置 ---
@configclass
class DashgoNavEnvV2Cfg(ManagerBasedRLEnvCfg):
    decimation = 10 
    episode_length_s = 30.0 
    scene = DashgoSceneV2Cfg(num_envs=64, env_spacing=10.0, replicate_physics=True)

    @configclass
    class ActionsCfg:
        wheels = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["left_wheel_joint", "right_wheel_joint"], scale=10.0)
    actions = ActionsCfg()

    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObservationGroupCfg):
            lidar_ranges = ObservationTermCfg(func=process_lidar_ranges, params={"sensor_cfg": SceneEntityCfg("lidar_sensor")}, scale=1.0, clip=(0.0, 5.0), noise=GaussianNoiseCfg(mean=0.0, std=0.05))
            ultrasonic_ranges = ObservationTermCfg(func=process_ultrasonic_data, params={"sensor_names": ["us_front", "us_left", "us_right", "us_back"]}, scale=1.0, clip=(0.0, 4.0), noise=GaussianNoiseCfg(mean=0.0, std=0.02))
            velocity = ObservationTermCfg(func=lambda env: env.scene["robot"].data.root_lin_vel_b[:, :2], noise=GaussianNoiseCfg(mean=0.0, std=0.02))
            target_cmd = ObservationTermCfg(func=process_target_relative_pos, params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")}, scale=1.0, noise=GaussianNoiseCfg(mean=0.0, std=0.05))
        policy = PolicyCfg()
    observations = ObservationsCfg()

    @configclass
    class CommandsCfg:
        target_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot", body_name="base_link",
            # [重大修改] 将范围扩大到 ±6.0 米。
            # 这能覆盖“教室大小”的场景。虽然初期更难撞到目标，
            # 但配合我们 15.0 权重的前进奖励，机器人有足够动力去探索这么远的距离。
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(-6.0, 6.0), pos_y=(-6.0, 6.0), pos_z=(0.0, 0.0),   
                roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(-3.14, 3.14)
            ),
            resampling_time_range=(10.0, 15.0), debug_vis=False, 
        )
    commands = CommandsCfg()

    @configclass
    class EventsCfg:
        # [同步修改] 机器人的重置范围也相应扩大，防止机器人总是出生在地图中心，导致永远只需要跑短途。
        reset_robot = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {"x": (-6.0, 6.0), "y": (-6.0, 6.0), "yaw": (-3.14, 3.14)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot")})
        
        # 障碍物范围也扩大，填满整个教室
        reset_obstacles_cyl = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {"x": (-6.0, 6.0), "y": (-6.0, 6.0), "yaw": (-3.14, 3.14)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("obstacles_cyl")})
        reset_obstacles_box = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {"x": (-6.0, 6.0), "y": (-6.0, 6.0), "yaw": (-3.14, 3.14)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("obstacles_box")})
        
        randomize_mass = EventTermCfg(func=mdp.randomize_rigid_body_mass, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"), "mass_distribution_params": (0.8, 1.2), "operation": "scale"})
        physics_material = EventTermCfg(func=mdp.randomize_rigid_body_material, mode="startup", params={"asset_cfg": SceneEntityCfg("robot"), "static_friction_range": (0.4, 1.0), "dynamic_friction_range": (0.4, 0.9), "restitution_range": (0.0, 0.0), "num_buckets": 64})
    events = EventsCfg()

    @configclass
    class RewardsCfg:
        progress_to_goal = RewardTermCfg(
            func=progress_toward_target_xy, weight=3.0,
            params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")},
        )
        reach_goal = RewardTermCfg(
            func=close_to_target_xy, weight=50.0,
            params={"command_name": "target_pose", "threshold": 0.5, "asset_cfg": SceneEntityCfg("robot")},
        )
        face_goal = RewardTermCfg(
            func=face_to_target_xy, weight=0.5,
            params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")},
        )
        collision = RewardTermCfg(
            func=undesired_contacts, weight=-15.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 5.0},
        )
        action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.0005)
        stand_still = RewardTermCfg(
            func=penalty_stand_still, weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot"), "min_velocity": 0.05}
        )
    rewards = RewardsCfg()

    @configclass
    class TerminationsCfg:
        time_out = TerminationTermCfg(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True)
        base_height = TerminationTermCfg(func=lambda env: env.scene["robot"].data.root_pos_w[:, 2] < 0.05)
        reach_goal = TerminationTermCfg(
            func=close_to_target_xy, 
            params={"command_name": "target_pose", "threshold": 0.5, "asset_cfg": SceneEntityCfg("robot")},
        )
    terminations = TerminationsCfg()