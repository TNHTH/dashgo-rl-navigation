import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(CURRENT_DIR, "dashgo.urdf")

if not os.path.exists(URDF_PATH):
    URDF_PATH = os.path.join(CURRENT_DIR, "config", "dashgo.urdf")

DASHGO_D1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=False,
        
        # [关键修复] 关闭实例化，强制为每个机器人生成独立的物理碰撞体
        # 这能彻底解决 "穿模/幽灵机器人" 问题
        make_instanceable=False, 
        
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.1,
            angular_damping=0.1,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
        ),
        joint_drive=None,
    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2), 
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
    ),
    
    actuators={
        "dashgo_wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            stiffness=0.0,
            damping=5.0,
            effort_limit_sim=20.0,
            velocity_limit_sim=5.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)