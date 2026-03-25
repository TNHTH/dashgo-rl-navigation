from pathlib import Path
from xml.etree import ElementTree

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def test_ros2_driver_defaults_match_ros1_authority_yaml():
    repo_root = _repo_root()
    ros1_yaml = repo_root / "drivers" / "EAI_DRIVER" / "src" / "config" / "my_dashgo_params.yaml"
    ros2_yaml = repo_root / "workspaces" / "ros2_ws" / "src" / "dashgo_driver_ros2" / "config" / "dashgo_driver.yaml"

    ros1_params = yaml.safe_load(ros1_yaml.read_text())
    ros2_params = yaml.safe_load(ros2_yaml.read_text())["dashgo_driver_node"]["ros__parameters"]

    assert ros1_params["baud"] == ros2_params["baudrate"]
    assert ros1_params["timeout"] == ros2_params["serial_timeout_sec"]
    assert ros1_params["wheel_diameter"] == ros2_params["wheel_diameter"]
    assert ros1_params["wheel_track"] == ros2_params["wheel_track"]
    assert ros1_params["encoder_resolution"] == ros2_params["encoder_resolution"]
    assert ros1_params["gear_reduction"] == ros2_params["gear_reduction"]
    assert ros1_params["motors_reversed"] == ros2_params["motors_reversed"]
    assert ros1_params["Kp"] == ros2_params["Kp"]
    assert ros1_params["Kd"] == ros2_params["Kd"]
    assert ros1_params["Ki"] == ros2_params["Ki"]
    assert ros1_params["Ko"] == ros2_params["Ko"]
    assert ros1_params["accel_limit"] == ros2_params["accel_limit"]
    assert ros1_params["sensorstate_rate"] == ros2_params["sensorstate_rate"]
    assert ros1_params["base_controller_rate"] == ros2_params["base_controller_rate"]
    assert ros1_params["base_frame"] == ros2_params["base_frame"]
    assert ros2_params["serial_port"] == "/dev/dashgo"


def test_lidar_defaults_match_selected_single_lidar_baseline():
    repo_root = _repo_root()
    legacy_launch = repo_root / "drivers" / "lakibeam_driver" / "src" / "launch" / "lakibeam1_scan.launch"
    lidar_yaml = repo_root / "workspaces" / "ros2_ws" / "src" / "lakibeam_driver_ros2" / "config" / "lakibeam_driver.yaml"
    static_tf_yaml = repo_root / "workspaces" / "ros2_ws" / "src" / "dashgo_driver_ros2" / "config" / "laser_static_tf.yaml"

    launch_root = ElementTree.fromstring(legacy_launch.read_text())
    legacy_lidar_node = launch_root.find("./node[@pkg='lakibeam1']")
    legacy_tf_node = launch_root.find("./node[@pkg='tf']")
    legacy_params = {
        element.attrib["name"]: element.attrib["value"]
        for element in legacy_lidar_node.findall("./param")
    }
    remap_target = legacy_lidar_node.find("./remap").attrib["to"]

    params = yaml.safe_load(lidar_yaml.read_text())["lakibeam_scan_node"]["ros__parameters"]
    static_tf = yaml.safe_load(static_tf_yaml.read_text())["static_tf_node"]["ros__parameters"]

    assert params["frame_id"] == legacy_params["frame_id"]
    assert params["hostip"] == legacy_params["hostip"]
    assert params["sensorip"] == legacy_params["sensorip"]
    assert params["port"] == int(legacy_params["port"])
    assert params["angle_offset"] == int(legacy_params["angle_offset"])
    assert params["scanfreq"] == int(legacy_params["scanfreq"])
    assert params["filter"] == int(legacy_params["filter"])
    assert params["laser_enable"] is True
    assert params["scan_range_start"] == int(legacy_params["scan_range_start"])
    assert params["scan_range_stop"] == int(legacy_params["scan_range_stop"])
    assert params["output_topic"] == remap_target
    assert static_tf["parent_frame"] == "base_link"
    assert static_tf["child_frame"] == "laser"
    assert static_tf["translation"] == [0.0, 0.0, 0.0]
    assert static_tf["rotation_rpy"] == [0.0, 0.0, 0.0]
    assert "/base_link /laser" in legacy_tf_node.attrib["args"]
