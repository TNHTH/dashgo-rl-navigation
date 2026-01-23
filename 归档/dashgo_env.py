import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
# [规范] 导入所有必要的管理器配置类
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg, EventTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from dashgo_assets import DASHGO_D1_CFG

# --- 1. 场景配置 (必须独立定义) ---
@configclass
class DashgoSceneCfg(InteractiveSceneCfg):
    # 地面
    terrain = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # 机器人
    robot = DASHGO_D1_CFG.replace(prim_path="{ENV_REGEX_NS}/Dashgo")
    # 传感器
    lidar_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Dashgo/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.13)),
        ray_alignment="yaw", 
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=5.0, 
        ),
        debug_vis=False, 
        mesh_prim_paths=["/World/GroundPlane"], 
    )

# --- 2. RL环境主配置 ---
@configclass
class DashgoNavigationEnvCfg(ManagerBasedRLEnvCfg):
    # [Sim-to-Real] 决策间隔
    decimation = 10 
    episode_length_s = 20.0

    # 引用上面的场景配置
    scene = DashgoSceneCfg(num_envs=64, env_spacing=5.0, replicate_physics=True)

    # --- 动作配置 (必须是类实例) ---
    @configclass
    class ActionsCfg:
        wheels = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["left_wheel_joint", "right_wheel_joint"],
            scale=10.0, 
        )
    actions = ActionsCfg()

    # --- 观测配置 (必须包含 PolicyCfg 组) ---
    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObservationGroupCfg):
            lidar_ranges = ObservationTermCfg(
                func=lambda env: env.scene["lidar_sensor"].data.ray_hits_w[..., -1],
                scale=1.0,
                clip=(0.0, 5.0),
                noise=GaussianNoiseCfg(mean=0.0, std=0.05), 
            )
            velocity = ObservationTermCfg(
                func=lambda env: env.scene["robot"].data.root_lin_vel_b[:, :2],
                noise=GaussianNoiseCfg(mean=0.0, std=0.02), 
            )
        policy = PolicyCfg()
    observations = ObservationsCfg()

    # --- 事件(域随机化)配置 ---
    @configclass
    class EventsCfg:
        # 随机推力
        push_robot = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "asset_cfg": SceneEntityCfg("robot")},
        )
        # 随机质量
        randomize_mass = EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "mass_distribution_params": (0.8, 1.2), 
                "operation": "scale",
            },
        )
        # 随机摩擦力
        physics_material = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "static_friction_range": (0.4, 1.0),
                "dynamic_friction_range": (0.4, 0.9),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )
    events = EventsCfg()

    # --- 奖励配置 ---
    @configclass
    class RewardsCfg:
        progress = RewardTermCfg(
            func=lambda env: env.scene["robot"].data.root_lin_vel_b[:, 0],
            weight=1.5,
        )
        collision = RewardTermCfg(
            func=lambda env: torch.where(env.scene["robot"].data.root_pos_w[:, 2] < 0.06, 1.0, 0.0),
            weight=-50.0,
        )
        action_rate = RewardTermCfg(
            func=lambda env: torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1),
            weight=-0.05,
        )
    rewards = RewardsCfg()

    # --- 终止条件配置 ---
    @configclass
    class TerminationsCfg:
        time_out = TerminationTermCfg(
            func=lambda env: env.episode_length_buf >= env.max_episode_length,
            time_out=True 
        )
        base_height = TerminationTermCfg(
            func=lambda env: env.scene["robot"].data.root_pos_w[:, 2] < 0.05
        )
    terminations = TerminationsCfg()