# DashGo Autoresearch 四小时守护记录

创建时间: 2026-03-25 15:17:50 +08:00

## 结论
- 已为 autoresearch 增加独立 keepalive 守护层。
- 守护窗口已设置到 `2026-03-25 19:17:29 +08:00`，至少持续 4 小时。
- keepalive 会检查 autoresearch 的真实进程与状态，若发现提前退出，会自动执行 `autoresearch-resume`。

## 本轮新增
- 新增脚本: `tools/ops/autoresearch_keepalive.py`
- 扩展运维入口: `tools/ops/dashgo-autotrain.sh`
- 新增命令:
  - `autoresearch-ensure-start [小时]`
  - `autoresearch-ensure-status`
  - `autoresearch-ensure-logs [-f]`
  - `autoresearch-ensure-stop`

## 当前状态
- keepalive 状态文件: `.artifacts/autopilot/autoresearch/keepalive_state.json`
- keepalive 事件流: `.artifacts/autopilot/autoresearch/keepalive_events.jsonl`
- keepalive 日志: `.artifacts/autopilot/autoresearch/autoresearch_keepalive.nohup.log`
- autoresearch 状态文件: `.artifacts/autopilot/autoresearch/state.json`
- autoresearch 当前目标: `构建 180° scratch 基线`

## 已验证事实
- keepalive 已启动。
- 首次检查发现 autoresearch 处于旧 `stopping` 状态，keepalive 已自动执行一次 `autoresearch-resume`。
- 当前 `autoresearch-status` 已恢复到 `train_running`，并重新进入 `构建 180° scratch 基线`。

## 使用方式
```bash
bash /home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh autoresearch-ensure-status
bash /home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh autoresearch-ensure-logs -f
bash /home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh autoresearch-status
bash /home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh autoresearch-watch 5
```
