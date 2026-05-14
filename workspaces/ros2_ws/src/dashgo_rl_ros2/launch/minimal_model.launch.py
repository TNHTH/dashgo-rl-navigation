import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("dashgo_rl_ros2")
    default_dashgo_params = os.path.join(pkg_share, "config", "dashgo_rl.yaml")
    default_model_path = os.path.join(pkg_share, "models", "policy_torchscript.pt")

    dashgo_params = LaunchConfiguration("dashgo_params")
    model_path = LaunchConfiguration("model_path")
    use_sim_time = LaunchConfiguration("use_sim_time")
    launch_bridge = LaunchConfiguration("launch_bridge")

    return LaunchDescription(
        [
            DeclareLaunchArgument("dashgo_params", default_value=default_dashgo_params),
            DeclareLaunchArgument("model_path", default_value=default_model_path),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("launch_bridge", default_value="false"),
            Node(
                package="dashgo_rl_ros2",
                executable="geo_nav_node",
                name="geo_nav_node",
                output="screen",
                parameters=[dashgo_params, {"use_sim_time": use_sim_time, "model_path": model_path}],
            ),
            Node(
                package="dashgo_rl_ros2",
                executable="goal_plan_bridge",
                name="goal_plan_bridge",
                output="screen",
                condition=IfCondition(launch_bridge),
                parameters=[dashgo_params, {"use_sim_time": use_sim_time}],
            ),
        ]
    )
