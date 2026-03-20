from __future__ import annotations

from pathlib import Path


def resolve_project_root(start: str | Path | None = None) -> Path:
    """从任意仓库内路径回溯到项目根目录。"""
    path = Path(start).resolve() if start is not None else Path(__file__).resolve()
    if path.is_file():
        path = path.parent

    for candidate in (path, *path.parents):
        if (candidate / ".git").exists() and (candidate / "README.md").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = resolve_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
APPS_ROOT = PROJECT_ROOT / "apps"
CONFIGS_ROOT = PROJECT_ROOT / "configs"
TOOLS_ROOT = PROJECT_ROOT / "tools"
WORKSPACES_ROOT = PROJECT_ROOT / "workspaces"
DRIVERS_ROOT = PROJECT_ROOT / "drivers"
REFERENCES_ROOT = PROJECT_ROOT / "references"
ARTIFACTS_ROOT = PROJECT_ROOT / ".artifacts"
TMP_ROOT = PROJECT_ROOT / ".tmp"

TRAINING_CONFIG_PATH = CONFIGS_ROOT / "training" / "train_cfg_v2.yaml"
DASHGO_URDF_PATH = CONFIGS_ROOT / "robot" / "dashgo.urdf"

ROS1_WS_ROOT = WORKSPACES_ROOT / "ros1_catkin_ws"
ROS2_WS_ROOT = WORKSPACES_ROOT / "ros2_ws"
ROS1_PACKAGE_ROOT = ROS1_WS_ROOT / "src" / "dashgo_rl"
ROS2_PACKAGE_ROOT = ROS2_WS_ROOT / "src" / "dashgo_rl_ros2"

EAI_DRIVER_ROOT = DRIVERS_ROOT / "EAI_DRIVER"
LAKIBEAM_DRIVER_ROOT = DRIVERS_ROOT / "lakibeam_driver"
EAI_DRIVER_CONFIG_ROOT = EAI_DRIVER_ROOT / "src" / "config"
EAI_PARAMS_YAML = EAI_DRIVER_CONFIG_ROOT / "my_dashgo_params.yaml"
EAI_PARAMS_FL_YAML = EAI_DRIVER_CONFIG_ROOT / "my_dashgo_params_fl.yaml"

DASHGO_REFERENCE_ROOT = REFERENCES_ROOT / "dashgo"

TRAIN_ARTIFACTS_ROOT = ARTIFACTS_ROOT / "train"
TRAIN_LOGS_ROOT = TRAIN_ARTIFACTS_ROOT / "logs"
TRAIN_SUCCESS_ROOT = TRAIN_ARTIFACTS_ROOT / "success"
TRAIN_SUCCESS_MODELS_ROOT = TRAIN_SUCCESS_ROOT / "models"
TRAIN_ARCHIVE_ROOT = TRAIN_ARTIFACTS_ROOT / "archive"

AUTOPILOT_ARTIFACTS_ROOT = ARTIFACTS_ROOT / "autopilot"
AUTOPILOT_RUNS_ROOT = AUTOPILOT_ARTIFACTS_ROOT / "runs"
AUTOPILOT_METRICS_ROOT = AUTOPILOT_ARTIFACTS_ROOT / "metrics"
AUTOPILOT_ANCHORS_ROOT = AUTOPILOT_ARTIFACTS_ROOT / "anchors"
AUTOPILOT_JOBS_RUNTIME_ROOT = AUTOPILOT_ARTIFACTS_ROOT / "jobs" / "runtime"
AUTOPILOT_LINEAGE_PATH = AUTOPILOT_ARTIFACTS_ROOT / "lineage.json"


def ensure_project_sys_path() -> Path:
    """供脚本入口在运行前注入项目与 src 路径。"""
    import sys

    for candidate in (PROJECT_ROOT, SRC_ROOT):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    return PROJECT_ROOT

