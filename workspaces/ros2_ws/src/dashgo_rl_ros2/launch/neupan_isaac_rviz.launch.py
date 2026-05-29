import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _neupan_bridge_process(context):
    planner_yaml = LaunchConfiguration("planner_yaml").perform(context)
    neupan_root = LaunchConfiguration("neupan_root").perform(context)
    python_executable = LaunchConfiguration("neupan_python").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
    shadow_mode = LaunchConfiguration("shadow_mode").perform(context)
    scan_range_min = LaunchConfiguration("scan_range_min").perform(context)
    scan_range_max = LaunchConfiguration("scan_range_max").perform(context)
    scan_angle_min = LaunchConfiguration("scan_angle_min").perform(context)
    scan_angle_max = LaunchConfiguration("scan_angle_max").perform(context)
    scan_down_sample = LaunchConfiguration("scan_down_sample").perform(context)
    scan_offset = LaunchConfiguration("scan_offset").perform(context)
    planning_frame = LaunchConfiguration("planning_frame").perform(context)
    max_lin_vel = LaunchConfiguration("max_lin_vel").perform(context)
    py_path = os.pathsep.join([neupan_root, os.environ.get("PYTHONPATH", "")])
    return [
        ExecuteProcess(
            cmd=[
                python_executable,
                "-m",
                "dashgo_rl_ros2.neupan_bridge",
                "--ros-args",
                "-p",
                f"planner_yaml_path:={planner_yaml}",
                "-p",
                f"neupan_root:={neupan_root}",
                "-p",
                f"use_sim_time:={use_sim_time}",
                "-p",
                f"shadow_mode:={shadow_mode}",
                "-p",
                f"state_frame:={planning_frame}",
                "-p",
                f"max_lin_vel:={max_lin_vel}",
                "-p",
                "plan_stale_timeout_sec:=0.0",
                "-p",
                f"scan_range_min_override:={scan_range_min}",
                "-p",
                f"scan_range_max_override:={scan_range_max}",
                "-p",
                f"scan_angle_min:={scan_angle_min}",
                "-p",
                f"scan_angle_max:={scan_angle_max}",
                "-p",
                f"scan_down_sample:={scan_down_sample}",
                "-p",
                f"scan_offset:={scan_offset}",
            ],
            additional_env={"PYTHONPATH": py_path},
            output="screen",
        )
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory("dashgo_rl_ros2")
    default_rviz_config = os.path.join(pkg_share, "rviz", "dashgo_nav.rviz")
    default_planner_yaml = os.path.join(pkg_share, "config", "neupan_dashgo.yaml")
    default_slam_params = os.path.join(pkg_share, "config", "slam_toolbox_online_async_dashgo.yaml")
    default_robot_urdf = "/home/gwh/dashgo_rl_project/configs/robot/dashgo.urdf"

    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz = LaunchConfiguration("use_rviz")
    use_slam = LaunchConfiguration("use_slam")
    planning_frame = LaunchConfiguration("planning_frame")
    rviz_config = LaunchConfiguration("rviz_config")
    slam_params = LaunchConfiguration("slam_params")
    robot_urdf = LaunchConfiguration("robot_urdf")
    robot_description = ParameterValue(Command(["xacro", " ", robot_urdf]), value_type=str)

    return LaunchDescription(
        [
            DeclareLaunchArgument("planner_yaml", default_value=default_planner_yaml),
            DeclareLaunchArgument("neupan_root", default_value="/home/gwh/NeuPAN"),
            DeclareLaunchArgument("neupan_python", default_value="/home/gwh/NeuPAN/.venv/bin/python"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("use_slam", default_value="true"),
            DeclareLaunchArgument("planning_frame", default_value="odom"),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz_config),
            DeclareLaunchArgument("slam_params", default_value=default_slam_params),
            DeclareLaunchArgument("robot_urdf", default_value=default_robot_urdf),
            DeclareLaunchArgument("shadow_mode", default_value="false"),
            DeclareLaunchArgument("scan_range_min", default_value="0.0"),
            DeclareLaunchArgument("scan_range_max", default_value="4.0"),
            DeclareLaunchArgument("scan_angle_min", default_value="-1.5707963267948966"),
            DeclareLaunchArgument("scan_angle_max", default_value="1.5707963267948966"),
            DeclareLaunchArgument("scan_down_sample", default_value="1"),
            DeclareLaunchArgument("scan_offset", default_value="[0.0, 0.0, 0.0]"),
            DeclareLaunchArgument("max_lin_vel", default_value="0.5"),
            Node(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                condition=IfCondition(use_slam),
                parameters=[slam_params, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="map_to_odom_identity",
                output="screen",
                condition=UnlessCondition(use_slam),
                arguments=[
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--z",
                    "0",
                    "--roll",
                    "0",
                    "--pitch",
                    "0",
                    "--yaw",
                    "0",
                    "--frame-id",
                    "map",
                    "--child-frame-id",
                    "odom",
                ],
                parameters=[{"use_sim_time": use_sim_time}],
            ),
            Node(
                package="dashgo_rl_ros2",
                executable="simple_path_bridge",
                name="simple_path_bridge",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "goal_frame": planning_frame,
                        "path_points": 48,
                        "use_astar": True,
                        "astar_resolution": 0.10,
                        "astar_inflation_radius": 0.45,
                        "astar_bounds_padding": 1.2,
                    }
                ],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "robot_description": robot_description,
                        "use_sim_time": use_sim_time,
                    }
                ],
            ),
            OpaqueFunction(function=_neupan_bridge_process),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                condition=IfCondition(use_rviz),
                arguments=["-d", rviz_config],
                parameters=[{"use_sim_time": use_sim_time}],
            ),
        ]
    )
