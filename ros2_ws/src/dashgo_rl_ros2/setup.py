from glob import glob
import os

from setuptools import setup


package_name = "dashgo_rl_ros2"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
        (f"share/{package_name}/maps", glob("maps/*")),
        (f"share/{package_name}/urdf", glob("urdf/*")),
        (f"share/{package_name}/worlds", glob("worlds/*")),
        (f"share/{package_name}/models", glob("models/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gwh",
    maintainer_email="gwh@example.com",
    description="DashGo 深度强化学习项目的 ROS2 控制与验证包。",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "geo_nav_node = dashgo_rl_ros2.geo_nav_node:main",
            "goal_plan_bridge = dashgo_rl_ros2.goal_plan_bridge:main",
        ],
    },
)
