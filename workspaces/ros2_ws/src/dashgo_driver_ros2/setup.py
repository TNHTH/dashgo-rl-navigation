from glob import glob
import os

from setuptools import setup


package_name = "dashgo_driver_ros2"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gwh",
    maintainer_email="gwh@example.com",
    description="DashGo 底盘 ROS2 原生串口驱动。",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dashgo_driver_node = dashgo_driver_ros2.driver_node:main",
            "static_tf_node = dashgo_driver_ros2.static_tf_node:main",
        ]
    },
)
