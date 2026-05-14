import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("dashgo_rl_ros2")
    dashgo_params = LaunchConfiguration("dashgo_params")
    nav2_params = LaunchConfiguration("nav2_params")
    map_yaml = LaunchConfiguration("map")
    model_path = LaunchConfiguration("model_path")
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_amcl = LaunchConfiguration("use_amcl")
    use_rviz = LaunchConfiguration("use_rviz")

    default_dashgo_params = os.path.join(pkg_share, "config", "dashgo_rl.yaml")
    default_nav2_params = os.path.join(pkg_share, "config", "nav2_planning.yaml")
    default_map = os.path.join(pkg_share, "maps", "nav.yaml")
    default_model_path = os.path.join(pkg_share, "models", "policy_torchscript.pt")
    default_rviz_config = os.path.join(pkg_share, "rviz", "dashgo_nav.rviz")

    lifecycle_nodes = ["map_server", "planner_server"]
    lifecycle_nodes_with_amcl = ["map_server", "planner_server", "amcl"]

    return LaunchDescription(
        [
            DeclareLaunchArgument("dashgo_params", default_value=default_dashgo_params),
            DeclareLaunchArgument("nav2_params", default_value=default_nav2_params),
            DeclareLaunchArgument("map", default_value=default_map),
            DeclareLaunchArgument("model_path", default_value=default_model_path),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz_config),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("use_amcl", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            Node(
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                output="screen",
                parameters=[nav2_params, {"use_sim_time": use_sim_time, "yaml_filename": map_yaml}],
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=[nav2_params, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="nav2_amcl",
                executable="amcl",
                name="amcl",
                output="screen",
                condition=IfCondition(use_amcl),
                parameters=[nav2_params, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_map_to_odom",
                condition=UnlessCondition(use_amcl),
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
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_planning",
                output="screen",
                condition=UnlessCondition(use_amcl),
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "autostart": True,
                        "node_names": lifecycle_nodes,
                    }
                ],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_planning_amcl",
                output="screen",
                condition=IfCondition(use_amcl),
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "autostart": True,
                        "node_names": lifecycle_nodes_with_amcl,
                    }
                ],
            ),
            Node(
                package="dashgo_rl_ros2",
                executable="goal_plan_bridge",
                name="goal_plan_bridge",
                output="screen",
                parameters=[dashgo_params, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="dashgo_rl_ros2",
                executable="geo_nav_node",
                name="geo_nav_node",
                output="screen",
                parameters=[dashgo_params, {"use_sim_time": use_sim_time, "model_path": model_path}],
            ),
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
