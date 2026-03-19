# 仓库清理记录 2026-03-19

## 本轮已处理

- 已归档的历史备份文件位于 `legacy_backups/`
- 已移出主工作区的缓存/临时目录位于 `workspace_temp/`

## 本轮保留未动

- `logs/`
- `logs_backup/`
- `ros2_ws/build`
- `ros2_ws/install`
- `ros2_ws/log`
- `catkin_ws/build`
- `catkin_ws/devel`
- `.claude-temp/`

## 保留原因

- 以上目录体积大，但都可能仍承载训练、部署或历史诊断证据。
- 若后续要继续瘦身，应单独做第二轮审计，再决定压缩、迁移或删除。
