# DashGo 仓库重构执行清单

- 时间: 2026-03-20 13:52 CST
- 方案版本: v2
- 工作目录: `/home/gwh/dashgo_rl_project`

## 删除与清理候选

- `__pycache__/`
- `.pytest_cache/`
- `catkin_ws/build/`
- `catkin_ws/devel/`
- `ros2_ws/build/`
- `ros2_ws/install/`
- `ros2_ws/log/`

## 主要迁移映射

- `train_v2.py` -> `apps/isaac/train_v2.py`
- `play.py` -> `apps/isaac/play.py`
- `export_torchscript.py` -> `apps/isaac/export_torchscript.py`
- `verify_ultimate_v5.py` -> `apps/isaac/verify_ultimate_v5.py`
- `dashgo_assets.py` -> `src/dashgo_rl/dashgo_assets.py`
- `dashgo_config.py` -> `src/dashgo_rl/dashgo_config.py`
- `dashgo_env_v2.py` -> `src/dashgo_rl/dashgo_env_v2.py`
- `geo_nav_policy.py` -> `src/dashgo_rl/geo_nav_policy.py`
- `safety_filter.py` -> `src/dashgo_rl/safety_filter.py`
- `train_cfg_v2.yaml` -> `configs/training/train_cfg_v2.yaml`
- `config/dashgo.urdf` -> `configs/robot/dashgo.urdf`
- `catkin_ws/` -> `workspaces/ros1_catkin_ws/`
- `ros2_ws/` -> `workspaces/ros2_ws/`
- `EAI_DRIVER/` -> `drivers/EAI_DRIVER/`
- `lakibeam_driver/` -> `drivers/lakibeam_driver/`
- `dashgo/` -> `references/dashgo/`
- `logs/` -> `.artifacts/train/logs/`
- `logs_backup/` -> `.artifacts/train/archive/logs_backup/`
- `training_success/` -> `.artifacts/train/success/`
- `autopilot/runs/` -> `.artifacts/autopilot/runs/`
- `autopilot/metrics/` -> `.artifacts/autopilot/metrics/`
- `autopilot/anchors/` -> `.artifacts/autopilot/anchors/`
- `autopilot/jobs/runtime/` -> `.artifacts/autopilot/jobs/runtime/`
- `autopilot/lineage.json` -> `.artifacts/autopilot/lineage.json`

## 活跃文档更新范围

- `README.md`
- `README_GITHUB.md`
- `QUICK_REFERENCE.md`
- `docs/INDEX.md`
- 活跃 shell 脚本与部署文档
