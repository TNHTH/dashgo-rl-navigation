import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory("dashgo_rl_ros2")

    world = LaunchConfiguration("world")
    map_yaml = LaunchConfiguration("map")
    nav2_params = LaunchConfiguration("nav2_params")
    dashgo_params = LaunchConfiguration("dashgo_params")
    model_path = LaunchConfiguration("model_path")
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_amcl = LaunchConfiguration("use_amcl")
    use_rviz = LaunchConfiguration("use_rviz")
    gui = LaunchConfiguration("gui")

    default_world = os.path.join(pkg_share, "worlds", "navigation_env.world")
    default_map = os.path.join(pkg_share, "maps", "nav.yaml")
    default_nav2_params = os.path.join(pkg_share, "config", "nav2_planning.yaml")
    default_dashgo_params = os.path.join(pkg_share, "config", "dashgo_rl.yaml")
    default_model_path = os.path.join(pkg_share, "models", "policy_torchscript.pt")
    default_rviz_config = os.path.join(pkg_share, "rviz", "dashgo_nav.rviz")
    urdf_path = os.path.join(pkg_share, "urdf", "dashgo_d1_sim.urdf.xacro")
    spawn_entity_script = os.path.join(
        get_package_prefix("gazebo_ros"), "lib", "gazebo_ros", "spawn_entity.py"
    )

    robot_description = ParameterValue(Command(["xacro", " ", urdf_path]), value_type=str)
    real_launch = os.path.join(pkg_share, "launch", "real_model_nav.launch.py")

    gazebo_server = ExecuteProcess(
        condition=UnlessCondition(gui),
        cmd=[
            "gzserver",
            "--verbose",
            world,
            "-s",
            "libgazebo_ros_init.so",
            "-s",
            "libgazebo_ros_factory.so",
        ],
        output="screen",
    )

    gazebo_gui = ExecuteProcess(
        condition=IfCondition(gui),
        cmd=[
            "gazebo",
            "--verbose",
            world,
            "-s",
            "libgazebo_ros_init.so",
            "-s",
            "libgazebo_ros_factory.so",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value=default_world),
            DeclareLaunchArgument("map", default_value=default_map),
            DeclareLaunchArgument("nav2_params", default_value=default_nav2_params),
            DeclareLaunchArgument("dashgo_params", default_value=default_dashgo_params),
            DeclareLaunchArgument("model_path", default_value=default_model_path),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz_config),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("use_amcl", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("gui", default_value="false"),
            gazebo_server,
            gazebo_gui,
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time, "robot_description": robot_description}],
            ),
            ExecuteProcess(
                cmd=[
                    "/usr/bin/python3.10",
                    spawn_entity_script,
                    "-entity",
                    "dashgo",
                    "-topic",
                    "robot_description",
                    "-x",
                    "0",
                    "-y",
                    "0",
                    "-z",
                    "0.0632",
                ],
                output="screen",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(real_launch),
                launch_arguments={
                    "map": map_yaml,
                    "nav2_params": nav2_params,
                    "dashgo_params": dashgo_params,
                    "model_path": model_path,
                    "rviz_config": rviz_config,
                    "use_sim_time": use_sim_time,
                    "use_amcl": use_amcl,
                    "use_rviz": use_rviz,
                }.items(),
            ),
        ]
    )
