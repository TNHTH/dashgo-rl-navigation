import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _sanitized_ld_library_path():
    """移除 conda 注入的动态库路径，避免 ROS2/C++ 节点误链到错误版本。"""
    raw_entries = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    entries = [
        entry
        for entry in raw_entries
        if entry and "miniconda" not in entry and "anaconda" not in entry
    ]
    fallback_entries = [
        "/opt/ros/humble/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
    ]
    for entry in fallback_entries:
        if os.path.isdir(entry) and entry not in entries:
            entries.append(entry)
    return ":".join(entries)


def generate_launch_description():
    dashgo_rl_share = get_package_share_directory("dashgo_rl_ros2")
    dashgo_driver_share = get_package_share_directory("dashgo_driver_ros2")
    lakibeam_share = get_package_share_directory("lakibeam_driver_ros2")

    base_params = LaunchConfiguration("base_params")
    lidar_params = LaunchConfiguration("lidar_params")
    static_tf_params = LaunchConfiguration("static_tf_params")
    dashgo_params = LaunchConfiguration("dashgo_params")
    nav2_params = LaunchConfiguration("nav2_params")
    map_yaml = LaunchConfiguration("map")
    model_path = LaunchConfiguration("model_path")
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_amcl = LaunchConfiguration("use_amcl")
    use_rviz = LaunchConfiguration("use_rviz")
    record_bag = LaunchConfiguration("record_bag")
    bag_output_dir = LaunchConfiguration("bag_output_dir")
    bag_prefix = LaunchConfiguration("bag_prefix")

    default_base_params = os.path.join(dashgo_driver_share, "config", "dashgo_driver.yaml")
    default_lidar_params = os.path.join(lakibeam_share, "config", "lakibeam_driver.yaml")
    default_static_tf = os.path.join(dashgo_driver_share, "config", "laser_static_tf.yaml")
    default_dashgo_params = os.path.join(dashgo_rl_share, "config", "dashgo_rl.yaml")
    default_nav2_params = os.path.join(dashgo_rl_share, "config", "nav2_planning.yaml")
    default_map = os.path.join(dashgo_rl_share, "maps", "nav.yaml")
    default_model_path = os.path.join(dashgo_rl_share, "models", "policy_torchscript.pt")
    default_rviz_config = os.path.join(dashgo_rl_share, "rviz", "dashgo_nav.rviz")
    default_bag_output_dir = os.path.join(os.path.expanduser("~"), "dashgo_rl_project", ".artifacts", "rosbags")

    base_launch = os.path.join(dashgo_driver_share, "launch", "base_only.launch.py")
    lidar_launch = os.path.join(lakibeam_share, "launch", "lidar_only.launch.py")
    nav_launch = os.path.join(dashgo_rl_share, "launch", "real_model_nav.launch.py")

    return LaunchDescription(
        [
            DeclareLaunchArgument("base_params", default_value=default_base_params),
            DeclareLaunchArgument("lidar_params", default_value=default_lidar_params),
            DeclareLaunchArgument("static_tf_params", default_value=default_static_tf),
            DeclareLaunchArgument("dashgo_params", default_value=default_dashgo_params),
            DeclareLaunchArgument("nav2_params", default_value=default_nav2_params),
            DeclareLaunchArgument("map", default_value=default_map),
            DeclareLaunchArgument("model_path", default_value=default_model_path),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz_config),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("use_amcl", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("record_bag", default_value="false"),
            DeclareLaunchArgument("bag_output_dir", default_value=default_bag_output_dir),
            DeclareLaunchArgument("bag_prefix", default_value="dashgo_real_robot_nav"),
            SetEnvironmentVariable(
                "LD_LIBRARY_PATH", _sanitized_ld_library_path()
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch),
                launch_arguments={"params_file": base_params}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(lidar_launch),
                launch_arguments={
                    "params_file": lidar_params,
                    "static_tf_params": static_tf_params,
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(nav_launch),
                launch_arguments={
                    "dashgo_params": dashgo_params,
                    "nav2_params": nav2_params,
                    "map": map_yaml,
                    "model_path": model_path,
                    "rviz_config": rviz_config,
                    "use_sim_time": use_sim_time,
                    "use_amcl": use_amcl,
                    "use_rviz": use_rviz,
                    "record_bag": record_bag,
                    "bag_output_dir": bag_output_dir,
                    "bag_prefix": bag_prefix,
                }.items(),
            ),
        ]
    )
