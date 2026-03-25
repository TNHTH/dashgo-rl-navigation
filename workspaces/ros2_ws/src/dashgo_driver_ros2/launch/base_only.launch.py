import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
    pkg_share = get_package_share_directory("dashgo_driver_ros2")
    params_file = LaunchConfiguration("params_file")
    default_params = os.path.join(pkg_share, "config", "dashgo_driver.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            SetEnvironmentVariable(
                "LD_LIBRARY_PATH", _sanitized_ld_library_path()
            ),
            Node(
                package="dashgo_driver_ros2",
                executable="dashgo_driver_node",
                name="dashgo_driver_node",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
