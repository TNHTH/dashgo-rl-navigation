# DashGo Isaac 自主值守训练进度日志

## 2026-03-17 01:16 CST

- 初始化 Obsidian 任务日志:
  - `/home/gwh/文档/Obsidian Vault/03_项目记录/DashGo Isaac 自主值守训练落地_2026-03-17_01-16.md`
- 创建自主值守工作区:
  - `/home/gwh/dashgo_rl_project/autopilot/`
  - `/home/gwh/dashgo_rl_project/autopilot/runs/`
  - `/home/gwh/dashgo_rl_project/autopilot/metrics/`
  - `/home/gwh/.codex/tmp/session-state/`
- 决定:
  - 不覆盖根目录既有 ROS2 迁移状态文件。
  - 新训练任务的所有执行记忆写入 `autopilot/`。

## 2026-03-17 01:32 CST

- 新增自主值守工具骨架:
  - `autopilot/runtime.py`
  - `autopilot/io_utils.py`
  - `autopilot/tensorboard_utils.py`
  - `doctor_training_env.py`
  - `eval_checkpoint.py`
  - `monitor_training.py`
- 修复训练链遗产:
  - `GeoNavPolicy` 兼容 `OnPolicyRunner(num_obs, num_privileged_obs, num_actions)` 构造。
  - `train_v2.py` 接入世代化 run 目录、run_meta、lineage 与 checkpoint 重定向。
  - `train_v2.py` 自动启用 `--enable_cameras`，避免四向深度相机初始化崩溃。
- 完成 Isaac smoke:
  - 命令: `~/IsaacLab/isaaclab.sh -p train_v2.py --headless --num_envs 2 --gen gen1 --run_name smoke_train_camfix --max_iterations 1 --save_interval 1`
  - 结果: 成功跑完 1 iter，生成 checkpoint `model_0.pt` 与 TensorBoard 事件文件。
  - 运行目录: `/home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_013156_smoke_train_camfix`
- 补充验证:
  - `doctor_training_env.py` 在 IsaacLab Python 下确认 `tensorboard` 可用。
  - `monitor_training.py` 可读取 smoke run 的最新标量与 checkpoint 信息。

## 2026-03-17 01:45 CST

- 完成首轮短波次训练对比:
  - `wave1_short`: `/home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_013517_wave1_short`
  - `wave2_resume`: `/home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_013631_wave2_resume`
- 结果:
  - `wave1_short` 到 `iter 20` 时仍为 `reach_goal=0.0 / time_out=1.0`，但均值 reward 退化尚不严重。
  - `wave2_resume` 从旧 `model_19.pt` 续训到 `iter 98` 后明显学坏，最终仍为 `reach_goal=0.0 / time_out=1.0`，且 `mean_reward` 进一步下降。
- 结论:
  - 旧 checkpoint 续训链不再作为 Gen1 主线，仅保留为遗产对照。

## 2026-03-17 01:47 CST

- 修复课程遗产:
  - `curriculum_adaptive_distance()` 改为直接更新当前 `RelativeRandomTargetCommand.max_dist`，不再只改无效配置对象。
  - Gen1 初始目标上限从 `1.5m` 下调到 `1.0m`。
- 启动修正后的 A/B:
  - 冷启动: `/home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_014017_wave3_cold_currfix`
  - 旧链续训: `/home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_014109_wave3_resume_currfix`
- 当前观察:
  - `wave3_cold_currfix` 在 `iter 8-11` 已出现非零 `reach_goal` 样本，说明课程修正有效。
  - `wave3_resume_currfix` 在同样修正下仍几乎全程 `reach_goal=0.0 / time_out=1.0`。
- 当前决策:
  - 将 `wave3_cold_currfix` 标记为当前最佳候选，后续只从这条冷启动链继续推进。
  - 下一步转入“冷启动出现成功样本后又退回 timeout”的主因审计。

## 2026-03-17 02:20 CST

- 完成传感器与奖励链根因审计:
  - 四向相机 `offset convention` 从默认 `ros` 改为 `world`。
  - 四向相机安装高度从 `0.13m` 提升到 `0.22m`。
  - `log_distance` 奖励权重改为 `0.0`，不再把“远离目标”计成正奖励。
  - 新增 `inspect_live_env.py` 与 `inspect_curriculum.py` 活体诊断工具。
- 关键验证:
  - `/tmp/dashgo_inspect_live_result4.json` 显示相机修复后 `step1 min_obstacle_distance mean≈1.0067`，近障惩罚归零。
  - `/tmp/dashgo_curriculum_payload_after_fix.json` 证明课程函数在真实“已完成回合”条件下会从 `1.0 -> 1.5`。
- 课程学习第二根因定位:
  - 初始化 `env.reset()` 也会调用课程函数，原逻辑把这次预回合 reset 当成失败样本灌入 `success_history`。
  - 已用 `episode_length_buf > 0` 过滤掉预回合 reset，并增加可控 trace：`DASHGO_CURRICULUM_TRACE_PATH`。
- 训练波次:
  - `wave4_cold_sensor_aclfix`
    - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_015834_wave4_cold_sensor_aclfix`
    - 结果: 在 `1.0m` easy curriculum 上稳定 `reach_goal=1.0 / collision=0.0`，但 `Curriculum/target_adaptive` 仍卡在 `1.0`，触发 ACL 深挖。
  - `wave5_acltrace_resume`
    - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_021002_wave5_acltrace_resume`
    - 结果: 追踪确认旧 ACL 被初始化假失败污染；这条 run 仅作为根因取证，不作为主模型来源。
  - `wave6_cold_aclresetfix`
    - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_021239_wave6_cold_aclresetfix`
    - 结果: `Curriculum/target_adaptive` 在 `iter 8-10` 从 `1.0 -> 2.0`，随后 `object_collision` 在 `iter 18+` 升到 `1.0`，证明课程升级过快。
    - 决策: 取 `model_16.pt` 作为“升级前稳定 checkpoint”。
  - `wave7_resume_step025`
    - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_021453_wave7_resume_step025`
    - 改动: 将 ACL `step_size` 从 `0.5` 降到 `0.25`，并从 `wave6` 的 `model_16.pt` 续训。
    - 结果: 课程只升到 `1.5m`，在 `iter 28-39` 稳定保持 `reach_goal=1.0 / object_collision=0.0 / position_error≈0.152 / mean_reward≈2.14`。
- 当前决策:
  - `wave7_resume_step025/checkpoints/model_39.pt` 升级为当前主线 checkpoint。
  - 下一步不再回退到冷启动从零开始，而是沿 `step_size=0.25 + model_39.pt` 继续长训，观察是否能稳定跨过 `1.5m` 并逐步推进到 `1.75m / 2.0m`。

## 2026-03-17 02:24 CST

- 完成长训波次:
  - `wave8_resume_step025_long`
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_021749_wave8_resume_step025_long`
- 关键结果:
  - 课程沿 trace 从 `1.0 -> 1.25 -> 1.5 -> 1.75 -> 2.0 -> 2.25 -> 2.5`
  - `2.25m` 阶段仍可保持 `reach_goal=1.0`，但近障与不安全速度惩罚显著增加。
  - `2.5m` 阶段真实成功率开始明显波动，并在 run 末段演化成 `time_out` 主导。
  - 这说明 `step_size=0.25` 已经把“2.0m 直接碰撞崩”修成了“2.5m 后期粘住超时”，主问题从升级过快切换成降级不及时。
- 训练决策:
  - 不接受 `wave8` 的末尾模型 `model_118.pt` 作为主线。
  - 选取 `wave8/checkpoints/model_60.pt` 作为更干净的恢复点。
  - 将 ACL `downgrade_threshold` 从 `0.4` 提高到 `0.6`，让课程在高难度失稳时更快退出瓶颈。
- 新波次已启动:
  - `wave9_resume_step025_down060`
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_022026_wave9_resume_step025_down060`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_021749_wave8_resume_step025_long/checkpoints/model_60.pt`
  - 目标: 验证 `downgrade_threshold=0.6` 是否能避免 `2.5m` 长时间 timeout 粘住。

## 2026-03-17 02:31 CST

- 完成 `wave9_resume_step025_down060`:
  - 实际路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_022025_wave9_resume_step025_down060`
  - 结果:
    - 课程成功从 `1.0m` 推进到 `4.5m`
    - 中途在 `4.25m` 附近出现过碰撞/成功率波动，但能够恢复
    - 末段恢复到 `reach_goal=1.0 / object_collision=0.0 / time_out=0.0`
  - 决策:
    - 将 `wave9/checkpoints/model_139.pt` 升级为新的主线候选 checkpoint。
- 完成 `wave10_resume_down060_long`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_022303_wave10_resume_down060_long`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_022025_wave9_resume_step025_down060/checkpoints/model_139.pt`
  - 结果:
    - 课程稳定推进到 `5.0m`
    - 末段 `Curriculum/target_adaptive=5.0`
    - 末段 `reach_goal=1.0 / time_out=0.0 / object_collision=0.0 / position_error≈0.214`
  - 决策:
    - 将 `wave10/checkpoints/model_198.pt` 升级为当前主线 checkpoint。
    - Gen1 静态随机化阶段已从“能否学会到达”转入“如何把高难度下的避障质量和 reward 质量继续做干净”的阶段。
- 补齐监控遗产:
  - `monitor_training.py` 已增加 IsaacLab `python.sh` 回退逻辑。
  - 现在在系统 Python 下也能稳定读出 TensorBoard 最新标量。

## 2026-03-17 09:23 CST

- 完成 Gen2 第一版脚本化动态障碍接线:
  - 修改 [dashgo_env_v2.py](/home/gwh/dashgo_rl_project/dashgo_env_v2.py)
  - 新增 `USE_AUTOPILOT_GEN2_DYNAMIC`
  - 新增 `configure_dynamic_obstacles(reset)` 与 `animate_dynamic_obstacles(interval)`
  - 复用 `obs_inner_1 / obs_inner_3 / obs_inner_5` 作为 `crossing / head_on / stop_go` 三类动态模板的运动学载体
  - `gen2` 下初始目标距离与课程起点改为 `3.0m`
- 完成 Gen2 烟测:
  - 命令: `~/IsaacLab/isaaclab.sh -p train_v2.py --headless --gen gen2 --run_name smoke_gen2_dynamic --num_envs 4 --max_iterations 1 --save_interval 1 --checkpoint /home/gwh/dashgo_rl_project/autopilot/runs/gen1/20260317_022303_wave10_resume_down060_long/checkpoints/model_198.pt`
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_092127_smoke_gen2_dynamic`
  - 结果:
    - 环境成功创建
    - Event Manager 正确注册 `configure_dynamic_obstacles` 与 `drive_dynamic_obstacles`
    - 成功从静态最佳 checkpoint `model_198.pt` 续训进入 `gen2`
    - `Curriculum/target_adaptive` 起点为 `3.0`
- 完成动态障碍活体验证:
  - 临时脚本输出: `/tmp/gen2_dynamic_motion.json`
  - 结果:
    - `env_0` 的 `obs_inner_1` 在 step 0/2/4/6 出现连续位移变化
    - `obs_inner_3 / obs_inner_5` 保持静止，符合“每回合仅激活一种动态模板”的当前设计
    - 更新频率与 `0.1s interval / 0.0667s env.step` 一致，表现为约每两步更新一次
- 当前决策:
  - Gen2 动态障碍接线视为通过
  - 下一步直接启动正式 `gen2` 短波次训练，优先观察碰撞、超时和课程是否还能维持在 `3.0m+`

## 2026-03-17 09:31 CST

- `wave11_gen2_bootstrap_short` 判定为坏配置:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_092616_wave11_gen2_bootstrap_short`
  - 配置: `num_envs=16`
  - 现象:
    - Isaac 启动阶段持续报 `Unable to allocate descriptor sets / Failed to allocate ParameterBlock resources`
    - `run_meta.json` 长时间停在 `initialized`
    - `monitor_training.py` 无 checkpoint、无 TensorBoard 标量
  - 处置:
    - 杀掉残留 Isaac Python 进程
    - 保持其余配置不变，只把 `num_envs` 降到 `8`
- 完成 `wave12_gen2_bootstrap_env8_short`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_092736_wave12_gen2_bootstrap_env8_short`
  - 配置: `num_envs=8`, `max_iterations=20`, 起点 checkpoint=`model_198.pt`
  - 结果:
    - 成功进入并完成学习循环
    - 产出 checkpoint `model_217.pt`
    - 由于 `90s` episode 太长，本轮结束时尚无 episode 终止，指标仍全部为零
  - 决策:
    - 不改奖励、不改动态模板
    - 直接从 `model_217.pt` 延长同配置波次，直到出现真实终止分布
- 完成 `wave13_gen2_env8_long`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_092852_wave13_gen2_env8_long`
  - 配置: `num_envs=8`, `max_iterations=80`, 起点 checkpoint=`wave12/model_217.pt`
  - 中段观察:
    - `Episode_Termination/object_collision=1.0`
    - `Train/mean_reward≈-200.63`
    - `Metrics/target_pose/position_error≈2.14`
    - 说明动态阶段初始适应存在明显碰撞坍缩
  - 末段结果:
    - `latest_checkpoint=model_296.pt`
    - `Curriculum/target_adaptive=3.0`
    - `Episode_Termination/reach_goal=1.0`
    - `Episode_Termination/object_collision=0.0`
    - `Episode_Termination/time_out=0.0`
    - `Train/mean_episode_length≈1050.22`
    - `Metrics/target_pose/position_error≈0.252`
  - 当前判断:
    - 动态阶段不是“学不会”，而是“进入后先撞崩、随后恢复”
    - 当前更应该继续沿同配置续训，让 ACL 从 `3.0m` 往上爬，而不是立刻改奖励或改模板

## 2026-03-17 09:36 CST

- 完成 `wave14_gen2_env8_continue` 复盘:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_093210_wave14_gen2_env8_continue`
  - 配置: `num_envs=8`, `max_iterations=80`, 起点 checkpoint=`wave13/model_296.pt`
  - 运行中段最佳观测:
    - `Curriculum/target_adaptive=3.25`
    - `Episode_Termination/reach_goal=1.0`
    - `Episode_Termination/object_collision=0.0`
    - `Metrics/target_pose/position_error≈0.253`
    - 对应更优恢复点: `checkpoints/model_320.pt`
  - run 末段 latest 结果:
    - `latest_checkpoint=model_375.pt`
    - `Curriculum/target_adaptive=3.0`
    - `Episode_Termination/reach_goal≈0.8333`
    - `Episode_Termination/time_out≈0.1667`
    - `Metrics/target_pose/position_error≈0.483`
- 当前决策:
  - 不把 `model_375.pt` 当作当前最佳 Gen2 模型。
  - 将 `wave14/checkpoints/model_320.pt` 标记为当前动态主线恢复点。
  - 下一轮继续保持 `num_envs=8`、动态模板不变、奖励不变，只做单变量课程改动：
    - `downgrade_threshold: 0.6 -> 0.65`
  - 目标: 验证在动态阶段 `3.25m` 窗口里，是否可以更快退出坏样本并减少回落到 `3.0m` 的概率。

## 2026-03-17 09:40 CST

- 完成失败实验 `wave15_gen2_model320_down065`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_093919_wave15_gen2_model320_down065`
  - 配置: `num_envs=8`, 起点 checkpoint=`wave14/model_320.pt`
  - 单变量改动: `downgrade_threshold: 0.6 -> 0.65`
  - 结果:
    - 前半段在 episode 尚未结束前看起来正常
    - 从 `iter 387+` 开始出现第一批完整 episode 后，指标稳定塌成:
      - `Episode_Termination/time_out = 8.0`
      - `Episode_Termination/reach_goal = 0.0`
      - `Episode_Termination/object_collision = 0.0`
      - `Metrics/target_pose/position_error ≈ 1.196`
      - `Train/mean_reward ≈ -197.10`
- 决策:
  - 判定 `downgrade_threshold=0.65` 为负向改动，整条 `wave15` 不进入主线候选。
  - 立即回滚代码到 `downgrade_threshold=0.6`。
  - 下一轮启动对照实验:
    - 仍从 `wave14/model_320.pt` 起跑
    - 保持 `num_envs=8`、动态模板不变、奖励不变
    - 只验证“在不改阈值的情况下，这个 checkpoint 是否还能重新恢复到 3.25m 窗口”

## 2026-03-17 09:49 CST

- 完成 `wave16_gen2_model320_baseline060` 对照实验:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_094230_wave16_gen2_model320_baseline060`
  - 结果:
    - 即使回滚到 `downgrade_threshold=0.6`，从 `model_320.pt` 续训后仍在首批完整 episode 进入 `object_collision=1.0` / `reach_goal=0.0`
  - 判断:
    - 问题不只是阈值，而是“checkpoint 恢复链缺课程状态”
- 完成恢复链修复:
  - 修改 [train_v2.py](/home/gwh/dashgo_rl_project/train_v2.py)
  - 新增 checkpoint sidecar:
    - 保存时写出 `model_xxx.curriculum.json`
    - 恢复时同时恢复 `current_dist / success_history / command max_dist`
    - 恢复后自动 `env.reset()` 以重新采样匹配当前课程的目标分布
- 为 `wave14/model_350.pt` 人工补齐验证用 sidecar:
  - 文件: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_093210_wave14_gen2_env8_continue/checkpoints/model_350.curriculum.json`
  - 用途: 验证 sidecar 修复能否真正把课程上下文接回去
- 完成 `wave17_gen2_model350_sidecar`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_094829_wave17_gen2_model350_sidecar`
  - 关键结果:
    - 恢复日志明确打印 `已恢复课程状态: current_dist=3.250, success_history=100`
    - 训练启动后 `Curriculum/target_adaptive` 直接维持在高位，并迅速推进:
      - `step 362-377`: `target_adaptive≈3.75 -> 4.0`，`reach_goal=1.0`，`collision=0.0`
      - `step 380`: 开始出现 `collision≈0.2917`，说明 `4.0 -> 4.25` 是当前断崖
      - `step 422`: 课程推到 `5.0`，但末段已不再保持到达成功
  - 当前结论:
    - sidecar 修复成功，已解决“中途 checkpoint 恢复失忆”主因
    - 当前最佳高难度 checkpoint 更新为 `wave17/checkpoints/model_370.pt`
    - 下一轮只改一个变量: `Gen2 step_size 0.25 -> 0.125`
    - 目标: 让课程以更细粒度跨过 `4.0 -> 4.25` 断崖

## 2026-03-17 09:54 CST

- 完成 `wave18_gen2_model370_step0125`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_095240_wave18_gen2_model370_step0125`
  - 单变量改动: `Gen2 step_size 0.25 -> 0.125`
  - 关键结果:
    - 课程从 `4.0` 先推进到 `4.125`
    - `step 405-411` 保持:
      - `Curriculum/target_adaptive = 4.25`
      - `Episode_Termination/reach_goal = 1.0`
      - `Episode_Termination/object_collision = 0.0`
      - `Metrics/target_pose/position_error ≈ 0.244`
    - 新断崖出现在 `4.375`:
      - `step 412+` 开始 `object_collision = 1.0`
      - 末段进一步演化成 `time_out = 6.0`
- 决策:
  - 接受 `step_size=0.125` 为正向改动，但不接受 `wave18` latest `model_449.pt` 作为主线。
  - 将当前最佳 checkpoint 更新为 `wave18/checkpoints/model_410.pt`。
  - 下一轮继续单变量:
    - `Gen2 step_size 0.125 -> 0.0625`
  - 目标: 继续把断崖从 `4.375` 推迟，争取在 `4.3125` 仍保持稳定成功。

## 2026-03-17 10:00 CST

- 完成 `wave19_gen2_model410_step00625` 复盘:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_095624_wave19_gen2_model410_step00625`
  - 单变量改动: `Gen2 step_size 0.125 -> 0.0625`
  - 恢复链:
    - 启动日志确认已恢复 `current_dist=4.250 / window_size=100 / success_history=100`
  - TensorBoard 对齐结果:
    - `step 431-439`: `Curriculum/target_adaptive=4.25`，`object_collision=1.0`，`reach_goal=0.0`，`position_error≈1.903`
    - `step 441-447`: 在同一 `4.25m` 难度下短暂恢复到 `reach_goal=1.0 / collision=0.0 / position_error≈0.223`
    - `step 449-466`: 再次滑回 `collision=1.0`
    - `step 466+`: 进一步转成 `time_out≈5.0` 主导
    - 全程未升到 `4.3125m`
- 结论:
  - 判定 `step_size=0.0625` 为负向改动，不接受为新的 Gen2 默认课程步长。
  - 当前最佳 checkpoint 维持不变，仍为 `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_095240_wave18_gen2_model370_step0125/checkpoints/model_410.pt`
  - 根因判断从“课程粒度仍不够细”转为“高难度 fine-tune 更新过猛，导致在 4.25m 内部遗忘”
- 执行动作:
  - 回滚 [dashgo_env_v2.py](/home/gwh/dashgo_rl_project/dashgo_env_v2.py) 中的 Gen2 `step_size` 到 `0.125`
  - 修改 [train_cfg_v2.yaml](/home/gwh/dashgo_rl_project/train_cfg_v2.yaml) 的 `learning_rate: 3.0e-4 -> 1.5e-4`
- 新波次已启动:
  - `wave20_gen2_model410_lr015e4_step0125`
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_100521_wave20_gen2_model410_lr015e4_step0125`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_095240_wave18_gen2_model370_step0125/checkpoints/model_410.pt`
  - 当前观测:
    - 启动成功，环境仍为 `num_envs=8`
    - sidecar 恢复日志已确认 `current_dist=4.250, success_history=100`
    - `iteration 410` 已开始，`Curriculum/target_adaptive=4.25`
  - 下一步:
    - 持续监控 `wave20`，判断降低学习率是否能保住 `4.25m`，并重新逼近 `4.375m`

## 2026-03-17 10:10 CST

- 完成 `wave20_gen2_model410_lr015e4_step0125`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_100521_wave20_gen2_model410_lr015e4_step0125`
  - 单变量改动: `learning_rate 3.0e-4 -> 1.5e-4`
  - 关键结果:
    - `step 443-465`: `reach_goal=1.0 / collision=0.0 / position_error≈0.251`
    - `step 466-477`: 退化为 `time_out=7.0`
    - `step 479-484`: 再次恢复到 `reach_goal=1.0 / collision=0.0`
    - `step 485+`: 转为 collision 主导
- 结论:
  - 降低学习率是部分正向改动，显著延长了 `4.25m` 的稳定窗口，但仍未推动课程升到 `4.375m`。
  - 当前更干净的高难度 checkpoint 更新为 `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_100521_wave20_gen2_model410_lr015e4_step0125/checkpoints/model_450.pt`
- 下一轮单变量:
  - `upgrade_threshold 0.8 -> 0.75`
  - 目标: 验证更宽松的升级门槛能否把 `wave20/model_450.pt` 的稳定窗口转成真正的课程升级

## 2026-03-17 10:12 CST

- 完成 `wave21_gen2_model450_up075_lr015e4`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_101037_wave21_gen2_model450_up075_lr015e4`
  - 单变量改动: `upgrade_threshold 0.8 -> 0.75`
  - 关键结果:
    - `step 500-505`: `reach_goal=1.0 / collision=0.0 / position_error≈0.253`
    - `step 506-527`: 长时间 `time_out=5.0` 主导
    - `step 529`: 最终塌成 `collision=1.0`
    - 全程 `Curriculum/target_adaptive` 都停在 `4.25`
- 结论:
  - 放宽升级阈值不是当前瓶颈，`upgrade_threshold=0.75` 判为负向改动并回滚。
  - 当前问题从 ACL 门槛继续收缩为“高难度策略稳定性不足”。
- 下一轮单变量:
  - `entropy_coef 0.01 -> 0.005`
  - 起点 checkpoint 继续使用 `wave20/model_450.pt`

## 2026-03-17 10:13 CST

- 完成 `wave22_gen2_model450_entropy0005_lr015e4`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_101300_wave22_gen2_model450_entropy0005_lr015e4`
  - 单变量改动: `entropy_coef 0.01 -> 0.005`
  - 关键结果:
    - `Mean action noise std` 从 `0.99` 下降到 `0.98`
    - `step 509-525`: `reach_goal=1.0 / collision=0.0 / position_error≈0.252`
    - `step 526`: 降到 `reach_goal≈0.833 / collision≈0.167`
    - `step 527+`: 再次变成 `collision=1.0`
    - sidecar 的 `success_history mean` 仍只有 `0.62~0.68`，课程没有升到 `4.375`
- 结论:
  - 降低熵系数也是部分正向改动，能把成功窗口拉长并降低随机抖动，但仍不足以让 ACL 升级。
  - 进一步定位到新的代码遗产:
    - [train_cfg_v2.yaml](/home/gwh/dashgo_rl_project/train_cfg_v2.yaml) 的 `num_steps_per_env=24` 是为 `4096` 环境写的旧值
    - 在当前 `num_envs=8` 下，单次 PPO rollout 只有 `192` 步，更新噪声过大
- 新波次已启动:
  - `wave23_gen2_model520_steps96_entropy0005_lr015e4`
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_101619_wave23_gen2_model520_steps96_entropy0005_lr015e4`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_101300_wave22_gen2_model450_entropy0005_lr015e4/checkpoints/model_520.pt`
  - 单变量改动: `num_steps_per_env 24 -> 96`
  - 当前观测:
    - 启动成功，sidecar 已恢复 `current_dist=4.250 / success_history=100`
    - 训练日志已确认 `每轮步数=96`
  - 下一步:
    - 继续监控更长 rollout 是否能把 `4.25m` 的成功窗口变成可持续提升

## 2026-03-17 10:31 CST

- 完成 `wave23_gen2_model520_steps96_entropy0005_lr015e4` 收尾:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_101619_wave23_gen2_model520_steps96_entropy0005_lr015e4`
  - 单变量改动: `num_steps_per_env 24 -> 96`
  - 关键结果:
    - `iteration 520-522`: `Curriculum/target_adaptive=4.125`
    - `iteration 523`: `target_adaptive≈4.0625 / reach_goal≈0.354 / collision≈0.073`
    - `iteration 524`: `target_adaptive≈3.802 / reach_goal≈0.583 / collision≈0.417`
    - `iteration 525-528`: 短暂恢复到 `target_adaptive=3.75 / reach_goal=1.0 / collision=0.0 / position_error≈0.223`
    - `iteration 570` 左右: 课程已降到 `3.0`，并转成 timeout 主导
    - `iteration 599`: `target_adaptive=3.0 / reach_goal=0.0 / object_collision≈0.46875 / time_out≈0.53125 / position_error≈1.200 / mean_reward≈-242.63`
- 结论:
  - `num_steps_per_env=96` 是负向改动，不仅没有把 `4.25m` 成功窗口转成课程升级，反而把课程从高位直接打回 `3.0m`。
  - 当前更稳妥的主线 checkpoint 仍保持为 `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_100521_wave20_gen2_model410_lr015e4_step0125/checkpoints/model_450.pt`。
  - 根因判断进一步收缩为：在 `num_envs=8` 的小 batch 条件下，问题更像“同一批 noisy sample 被重复优化过头”，而不是 rollout 长度不足。
- 执行动作:
  - 回滚 [train_cfg_v2.yaml](/home/gwh/dashgo_rl_project/train_cfg_v2.yaml) 中的 `runner.num_steps_per_env` 到 `24`
  - 保留 `learning_rate=1.5e-4`、`entropy_coef=0.005`、`step_size=0.125`、`upgrade_threshold=0.8`、`downgrade_threshold=0.6`
  - 下一轮只做单变量改动: `num_learning_epochs 5 -> 3`
- 新波次计划:
  - `wave24_gen2_model450_epochs3_entropy0005_lr015e4`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_100521_wave20_gen2_model410_lr015e4_step0125/checkpoints/model_450.pt`
  - 目标: 验证降低 PPO 同 batch 重复更新次数，是否能减少 `4.25m` 难度内部的遗忘与坍缩

## 2026-03-17 10:33 CST

- `wave24_gen2_model450_epochs3_entropy0005_lr015e4` 已成功启动:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_103328_wave24_gen2_model450_epochs3_entropy0005_lr015e4`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_100521_wave20_gen2_model410_lr015e4_step0125/checkpoints/model_450.pt`
  - 单变量改动:
    - `runner.num_steps_per_env: 96 -> 24`（回滚到已验证基线）
    - `algorithm.num_learning_epochs: 5 -> 3`
  - 启动验证:
    - 课程 sidecar 已成功恢复 `current_dist=4.250 / success_history=100`
    - `iteration 450-474` 期间课程稳定在 `4.25m`
    - 首批有效 episode 样本在 `iteration 475` 已出现 `reach_goal=0.5 / collision=0.0 / time_out=0.0 / position_error≈1.051 / mean_reward≈-52.50`
    - `monitor_training.py` 已确认 checkpoint 正常落盘到 `model_470.pt`
- 当前判断:
  - `wave24` 目前健康启动，尚未出现 `wave23` 那种快速降级回 `3.0m` 的迹象。
  - 下一步继续监控其是否能把 `4.25m` 上的非零成功样本扩大成稳定窗口，并观察是否首次逼近 `4.375m`。

## 2026-03-17 10:36 CST

- 完成 `wave24_gen2_model450_epochs3_entropy0005_lr015e4`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_103328_wave24_gen2_model450_epochs3_entropy0005_lr015e4`
  - 单变量改动: `num_learning_epochs 5 -> 3`
  - 关键结果:
    - `step 476-505`: 在 `4.25m` 上形成连续 `30` 个 iteration 的纯成功窗口
      - `reach_goal=1.0`
      - `object_collision=0.0`
      - `time_out=0.0`
      - `position_error≈0.242`
    - `step 506+`: 突然转成 `time_out=7.0 / collision=0.0 / position_error≈1.520`
    - sidecar:
      - `model_470 ~ model_500` 的 `success_history mean ≈ 0.68`
      - `model_510+` 回落到 `≈ 0.61`
  - 横向对比:
    - `wave20` 在 `4.25m` 的最长纯成功窗口为 `23` 步
    - `wave22` 也是 `23` 步
    - `wave24` 拉长到了 `30` 步
- 结论:
  - `num_learning_epochs=3` 是部分正向改动，应保留。
  - 当前最佳 checkpoint 更新为 `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_103328_wave24_gen2_model450_epochs3_entropy0005_lr015e4/checkpoints/model_500.pt`。
  - 新瓶颈收敛为：`192` 样本 rollout 被 `4` 个 mini-batch 切成 `48` 样本块，仍可能导致高难度后段超时塌缩。
- 下一轮单变量:
  - `num_mini_batches 4 -> 2`
  - 保持:
    - `num_steps_per_env=24`
    - `num_learning_epochs=3`
    - `learning_rate=1.5e-4`
    - `entropy_coef=0.005`
    - `step_size=0.125`
    - `upgrade_threshold=0.8`
    - `downgrade_threshold=0.6`
- 新波次计划:
  - `wave25_gen2_model500_epochs3_minib2_entropy0005_lr015e4`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_103328_wave24_gen2_model450_epochs3_entropy0005_lr015e4/checkpoints/model_500.pt`
  - 目标: 验证更大的 mini-batch 是否能把 `4.25m` 成功窗口进一步稳定化，而不是后段突然 timeout

## 2026-03-17 10:39 CST

- 完成 `wave25_gen2_model500_epochs3_minib2_entropy0005_lr015e4`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_103804_wave25_gen2_model500_epochs3_minib2_entropy0005_lr015e4`
  - 单变量改动: `num_mini_batches 4 -> 2`
  - 关键结果:
    - `model_540 ~ model_550`: `current_dist=4.25 / success_history mean≈0.60`
    - `model_560+`: 课程被打回 `4.0`
    - `step 560-579`: `reach_goal=0.0 / object_collision=1.0 / time_out=0.0 / position_error≈2.772`
- 结论:
  - `num_mini_batches=2` 是负向改动，会把 `wave24` 的后段 timeout 问题提前转成更激烈的碰撞坍缩。
  - 当前最佳 checkpoint 维持不变，仍为 `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_103328_wave24_gen2_model450_epochs3_entropy0005_lr015e4/checkpoints/model_500.pt`。
- 下一轮单变量:
  - 回滚 `num_mini_batches: 2 -> 4`
  - `learning_rate 1.5e-4 -> 1.0e-4`
  - 其余保持不变
- 新波次计划:
  - `wave26_gen2_model500_epochs3_lr010e4_entropy0005`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_103328_wave24_gen2_model450_epochs3_entropy0005_lr015e4/checkpoints/model_500.pt`
  - 目标: 验证更保守的更新幅度，是否能保住 `wave24` 的 `30` 步成功窗口并减轻后段 timeout

## 2026-03-17 10:43 CST

- 完成 `wave26_gen2_model500_epochs3_lr010e4_entropy0005`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_104256_wave26_gen2_model500_epochs3_lr010e4_entropy0005`
  - 单变量改动: `learning_rate 1.5e-4 -> 1.0e-4`
  - 关键结果:
    - `model_560+`: 课程被打回 `4.0`
    - `step 568-579`: `reach_goal=0.0 / object_collision=1.0 / time_out=0.0 / position_error≈2.576`
- 结论:
  - 固定把学习率降到 `1.0e-4` 也是负向改动，不能替代 `wave24` 基线。
  - 当前最佳 checkpoint 仍保持为 `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_103328_wave24_gen2_model450_epochs3_entropy0005_lr015e4/checkpoints/model_500.pt`。
- 下一轮单变量:
  - 回滚 `learning_rate: 1.0e-4 -> 1.5e-4`
  - `desired_kl 0.01 -> 0.005`
  - 其余保持 `wave24` 基线
- 新波次计划:
  - `wave27_gen2_model500_epochs3_kl0005_entropy0005_lr015e4`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_103328_wave24_gen2_model450_epochs3_entropy0005_lr015e4/checkpoints/model_500.pt`
  - 目标: 验证更早触发 PPO 自适应降速，是否能保住 `wave24` 后段稳定性

## 2026-03-17 13:20 CST

- 用户基于 `play.py` 三组避障观察命令反馈:
  - 已有倒车能力
  - 避障与导航能力明显不足
  - 存在“一点一点往前蹭”的行为
- 根因复查:
  - `reward_target_speed()` 之前会在近障时把“期望进度”压到接近 `0`，低速前蹭也可能拿到相对不差的局部奖励。
  - `shaping_distance` 仍然直接绑定最终目标距离，会和局部避障/局部 waypoint 信号互相打架。
  - `RelativeRandomTargetCommand` 的 `reference_path_w` 仍是直线插值，路径语义本身仍偏弱。
- 已做单变量修正 1:
  - 修改 [dashgo_env_v2.py](/home/gwh/dashgo_rl_project/dashgo_env_v2.py)
  - `progress_velocity / target_speed / facing_goal` 已统一对齐到 `waypoint`
  - 新增 `progress_stall` 奖励项，专门惩罚持续低进展的前蹭/发呆
  - `monitor_training.py` 增加 `Episode_Reward/progress_stall` 与 `Episode_Reward/target_speed` 监控
- `wave28_gen2_model500_waypointreward_stallfix`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_125448_wave28_gen2_model500_waypointreward_stallfix`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_103328_wave24_gen2_model450_epochs3_entropy0005_lr015e4/checkpoints/model_500.pt`
  - 结果:
    - 中段 `step 604-611` 曾出现 `reach_goal=1.0 / collision=0.0 / time_out=0.0 / position_error≈0.252`
    - 末段塌成 `time_out=6.0 / collision=0.0 / target_adaptive=3.75 / position_error≈2.008`
    - `progress_stall` 在末段开始生效，但不足以阻止整轮回落
  - 决策:
    - 不接受 latest `model_619.pt` 作为新主线
    - 选取 `model_600.pt` 作为后续试验的恢复点
- 已做单变量修正 2（已回滚）:
  - 尝试在 `get_waypoint_pose_w()` 中加入基于当前障碍物位置的轻量 detour
  - 对应 run: `wave29_gen2_model600_waypoint_detour`
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_130856_wave29_gen2_model600_waypoint_detour`
  - 结果:
    - 第一批完整 episode 直接表现为 `time_out=8.0 / collision=0.0 / target_adaptive=3.75`
    - 后段继续退化到 `collision=1.0`，课程回落到 `3.25~3.5`
  - 决策:
    - 判定为负向改动
    - 已回滚 detour 代码，不保留这条实现
- 已做单变量修正 3:
  - 将 `shaping_distance` 从最终目标距离改为 `waypoint` 距离
  - 对应 run: `wave30_gen2_model600_waypoint_shaping`
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_131351_wave30_gen2_model600_waypoint_shaping`
  - 关键结果:
    - `step 652-655`: 出现 `reach_goal=1.0 / collision=0.0 / time_out=0.0 / position_error≈0.251 / target_adaptive=3.75`
    - 后段主要退化成 `time_out` 主导，而非快速转成碰撞主导
    - latest 仍回落到 `target_adaptive=3.375 / time_out=7.0`
  - 决策:
    - 这是相对 `wave29` 的部分正向改动，应保留
    - 新的续训锚点更新为 `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_131351_wave30_gen2_model600_waypoint_shaping/checkpoints/model_640.pt`
- 已做单变量修正 4:
  - 强化 `progress_stall` 的触发灵敏度与权重
  - 对应 run: `wave31_gen2_model640_waypoint_shaping_stalltuned`
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_131758_wave31_gen2_model640_waypoint_shaping_stalltuned`
  - 结果:
    - 第一批完整 episode 直接表现为 `time_out=8.0 / collision=0.0 / progress_stall≈-0.029 / target_adaptive=3.625`
    - 未观察到优于 `wave30` 的成功窗口
  - 决策:
    - 判定为负向改动
    - 已中止 `wave31`，不把这条 stall-tune 当成新基线
- 当前结论:
  - `wave30` 的 `waypoint shaping` 是当前这轮里唯一明确的正向改动
  - 现阶段主失败模式已从“容易碰撞”收敛到“高难度下 timeout / 局部导航信号不够强”
  - 下一轮更合理的单变量不再是继续加重 stall penalty，而是把训练期 `reference_path` 从线性插值升级为更可靠的 obstacle-aware 路径语义

## 2026-03-17 14:26 CST

- 完成 `wave32_gen2_model640_obstaclepath`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_141833_wave32_gen2_model640_obstaclepath`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_131351_wave30_gen2_model600_waypoint_shaping/checkpoints/model_640.pt`
  - 单变量改动:
    - 回退 `wave31` 的 `progress_stall` 强化参数到 `wave30` 基线
    - 将训练侧 `reference_path` 从线性插值升级为首阻挡障碍绕行的 obstacle-aware 双折线
  - 关键结果:
    - 前 `60+` iterations 无初始化错误，说明改动与现有观测/动作/网络合同兼容
    - `step 704`: 首批完整 episode 表现为 `time_out=8.0 / collision=0.0 / target_adaptive=3.625`
    - `step 711-751`: 一度恢复到 `target_adaptive=3.5 / reach_goal=1.0 / collision=0.0 / position_error≈0.250`
    - `step 753+`: 再次退化为 `collision≈0.917~1.0 / reach_goal=0.0 / target_adaptive≈3.25`
  - 决策:
    - 判定为负向改动，不进入主线
    - 回滚 obstacle-aware reference_path 大改
- 启动 `wave33_gen2_model640_shortlookahead`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_142502_wave33_gen2_model640_shortlookahead`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_131351_wave30_gen2_model600_waypoint_shaping/checkpoints/model_640.pt`
  - 单变量改动:
    - 保留 `wave30` 的 waypoint shaping
    - 保留线性 `reference_path`
    - 仅收紧前进 lookahead:
      - `lookahead_max_forward: 1.2 -> 0.9`
      - `lookahead_gain_forward: 3.0 -> 2.2`
- 当前观察:
  - run 已成功起跑，无新的环境/合同级错误
  - 主目标是减少切角和贴边，验证是否比 `wave30` 更稳

## 2026-03-17 14:42 CST

- 完成 `wave33_gen2_model640_shortlookahead`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_142502_wave33_gen2_model640_shortlookahead`
  - 关键结果:
    - latest 指标退化为 `target_adaptive=3.25 / reach_goal=0.0 / object_collision=1.0 / time_out=0.0 / position_error≈2.032`
  - 决策:
    - 判定为负向改动
    - 回滚短前瞻参数，恢复 `wave30` 的 lookahead 基线

- 完成吞吐边界 smoke:
  - `12 envs`: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_143253_wave34_smoke_env12`
  - `10 envs`: 通过命令行 smoke 复现
  - `9 envs`: 通过命令行 smoke 复现
  - 三者共同现象:
    - 仿真启动阶段复现 `Unable to allocate descriptor sets`
    - 同步复现 `Failed to allocate ParameterBlock resources`
    - 出现 `HydraEngine::render failed`
  - 决策:
    - 当前 `4 向深度相机 + headless rendering + RTX` 组合下，`8 envs` 确认为硬上限
    - 不再继续试更高 `num_envs`

- 新的根因判断:
  - `wave30` latest 表现为 `time_out=7.0 / collision=0.0 / position_error≈1.86 / orientation_error≈1.44rad`
  - 说明主要矛盾是“局部路径朝向约束不足”，不是“吞吐不够”或“速度过快”

- 已做单变量修正 5:
  - 修改 [dashgo_env_v2.py](/home/gwh/dashgo_rl_project/dashgo_env_v2.py)
  - `facing_goal` 权重 `0.05 -> 0.15`
  - 保持观测、动作、网络、课程、lookahead 合同不变
  - 目标: 让策略在 `wave30/model_640.pt` 基础上更愿意先把车头对准 waypoint，再推进局部路径

- 启动 `wave35_gen2_model640_waypoint_facingboost`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_144047_wave35_gen2_model640_waypoint_facingboost`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_131351_wave30_gen2_model600_waypoint_shaping/checkpoints/model_640.pt`
  - 当前观察:
    - 已通过环境初始化并成功进入训练循环
    - 当前课程显示 `target_adaptive=3.75`
    - 尚未出现第一批完整 episode，待继续观察是否重演 `wave31` 那样的早期坏模式

## 2026-03-17 14:49 CST

- 完成 `wave35_gen2_model640_waypoint_facingboost`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_144047_wave35_gen2_model640_waypoint_facingboost`
  - 单变量改动: `facing_goal` 权重 `0.05 -> 0.15`
  - 关键结果:
    - latest 仍为 `target_adaptive=3.375 / time_out=7.0 / collision=0.0 / reach_goal=0.0`
    - `position_error≈1.196` 优于 `wave30 latest≈1.859`
    - 但没有形成新的纯成功窗口
  - 决策:
    - 不保留为新主线
    - 回滚 `facing_goal` 权重到 `0.05`

- 完成 `wave36_gen2_model640_baseline_capture`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_144538_wave36_gen2_model640_baseline_capture`
  - 单变量改动: 不改训练语义，只把 `save_interval` 加密到 `5`
  - 关键结果:
    - run 全程停留在 `target_adaptive=3.75 / reach_goal=0.0 / collision=0.0 / time_out=0.0`
    - 没有重现 `wave30` 在 `647-655` 出现的纯成功窗口
    - 说明“同起点 + 同设置”并不能稳定复现原窗口，仅加密保存不足以捕获更优 resume 点
  - 决策:
    - 不把 `wave36/model_669.pt` 当成更优锚点

- 启动 `wave37_gen2_model640_seed43_capture`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_144810_wave37_gen2_model640_seed43_capture`
  - 单变量改动: `seed 42 -> 43`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_131351_wave30_gen2_model600_waypoint_shaping/checkpoints/model_640.pt`
  - 当前观察:
    - 已通过环境初始化并进入训练循环
    - 初始状态与 `wave36` 已出现差异：`position_error≈1.766 / orientation_error≈1.451`
    - 当前继续监控是否会生成新的成功窗口并产出更干净的 resume checkpoint

## 2026-03-17 14:53 CST

- 完成 `wave37_gen2_model640_seed43_capture`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_144810_wave37_gen2_model640_seed43_capture`
  - 单变量改动: `seed 42 -> 43`
  - 关键结果:
    - latest 保持在 `target_adaptive=3.75 / reach_goal=0.0 / collision=0.0 / time_out=0.0`
    - `position_error≈1.766 / orientation_error≈1.451`
    - 仍没有生成纯成功窗口
  - 决策:
    - 判定为“较干净但不足够”，不作为新锚点
    - 说明在 `wave30/model_640.pt` 这个前沿 checkpoint 上，仅换 seed 仍不足以稳定产生成功段

- 新判断:
  - 不触发重训
  - 不再继续在 `wave30/model_640.pt` 上反复试 seed
  - 改为回退到更早的稳定上游锚点 `wave28/model_600.pt`，在保留 `wave30` waypoint shaping 的前提下重建正向分支

- 启动 `wave38_gen2_model600_seed43_waypointshaping_rebuild`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_145303_wave38_gen2_model600_seed43_waypointshaping_rebuild`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_125448_wave28_gen2_model500_waypointreward_stallfix/checkpoints/model_600.pt`
  - 单变量改动: 更换续训锚点（`wave30/model_640.pt` -> `wave28/model_600.pt`），保持 seed=43 和当前代码合同不变
  - 当前观察:
    - 已通过环境初始化并进入训练循环
    - 恢复课程状态 `current_dist=4.0` 后，首轮已稳定回到 `target_adaptive=3.875`
    - 待继续观察第一批完整 episode 是否会比 `640` 锚点更容易形成成功窗口

## 2026-03-17 14:55 CST

- `wave38_gen2_model600_seed43_waypointshaping_rebuild` 中期监控:
  - latest checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_145303_wave38_gen2_model600_seed43_waypointshaping_rebuild/checkpoints/model_670.pt`
  - 当前指标:
    - `Curriculum/target_adaptive=3.5`
    - `reach_goal=1.0`
    - `object_collision=0.0`
    - `time_out=0.0`
    - `position_error≈0.251`
    - `orientation_error≈2.97`
  - 中间判断:
    - 回退到更早上游锚点这条线首次重新产出了纯成功窗口
    - 当前证据已明显优于在 `wave30/model_640.pt` 上继续做 seed sweep
  - 当前动作:
    - 继续让 `wave38` 跑完
    - 待 run 完成后在 `model_665/670/675/680` 中选最新且仍处于纯成功窗口的 checkpoint 作为新 anchor 候选

## 2026-03-17 14:57 CST

- 完成 `wave38_gen2_model600_seed43_waypointshaping_rebuild`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_145303_wave38_gen2_model600_seed43_waypointshaping_rebuild`
  - 关键结果:
    - 在 `step 632-655` 形成 `24` 步纯成功窗口
    - 窗口指标: `target_adaptive=3.75 / reach_goal=1.0 / collision=0.0 / time_out=0.0 / position_error≈0.251`
    - 随后 `step 659-668` 回落为 `3.625 / time_out=7.0`
    - `step 670-679` 又恢复为 `3.5 / reach_goal=1.0 / collision=0.0 / time_out=0.0`
  - 决策:
    - 新的训练续训锚点更新为 `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_145303_wave38_gen2_model600_seed43_waypointshaping_rebuild/checkpoints/model_655.pt`
    - 选择理由: 它是当前已实际保存、且位于最高课程难度 `3.75m` 纯成功窗口末端的 checkpoint

## 2026-03-17 15:00 CST

- `wave39_gen2_model655_extend375` 早期监控结论:
  - 从 `wave38/model_655.pt` 直接续训会在恢复后立刻把课程从 `3.75` 降到 `3.625`
  - 根因已定位:
    - 不是策略参数突然坏掉
    - 而是 `model_655.curriculum.json` 里保留了更早阶段的大量失败历史，`success_history mean=0.44`，一恢复就触发 ACL 降级
  - 决策:
    - 不把 `wave39` 当成有效策略验证 run
    - 转而构造“清洁恢复历史”的派生锚点，而不改模型权重

- 已创建派生锚点:
  - 目录: `/home/gwh/dashgo_rl_project/autopilot/anchors/wave38_model655_stablehistory`
  - 模型: `/home/gwh/dashgo_rl_project/autopilot/anchors/wave38_model655_stablehistory/model_655_stablehistory.pt`
  - sidecar: `/home/gwh/dashgo_rl_project/autopilot/anchors/wave38_model655_stablehistory/model_655_stablehistory.curriculum.json`
  - 修正内容:
    - 保持 `current_dist=3.75`
    - 将 `success_history` 调整为均值 `0.7` 的中性稳定区间
    - 目的: 避免恢复即被旧失败样本拖降级

- 启动 `wave40_gen2_model655_stablehistory_extend375`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_145909_wave40_gen2_model655_stablehistory_extend375`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/anchors/wave38_model655_stablehistory/model_655_stablehistory.pt`
  - 当前观察:
    - 已通过环境初始化并进入训练循环
    - 恢复后课程仍保持 `target_adaptive=3.75`
    - 相比 `wave39`，已确认“课程历史清洁化”修复了恢复即降级的问题

## 2026-03-17 15:02 CST

- 完成 `wave40_gen2_model655_stablehistory_extend375`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_145909_wave40_gen2_model655_stablehistory_extend375`
  - 关键结果:
    - 全程保持 `target_adaptive=3.75`
    - 没有再出现 `wave39` 那种恢复即降级
    - 但这 40 个 iteration 内未产生新的 episode 终止，因此还不能证明成功窗口被继续推长
  - 决策:
    - 保留“清洁 curriculum history”这条恢复策略
    - 继续从 `wave40/model_694.pt` 往后拉长训练，观察更长时间尺度下是否生成新的成功窗口

- 启动 `wave41_gen2_model694_stablehistory_extend375_long`:
  - 路径: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_150207_wave41_gen2_model694_stablehistory_extend375_long`
  - 起点 checkpoint: `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_145909_wave40_gen2_model655_stablehistory_extend375/checkpoints/model_694.pt`
  - 单变量改动: 不改合同与权重，只延长同一恢复链的训练时长
  - 当前目标:
    - 在保持 `3.75m` 不降级的前提下，等待新的完整 episode 终止分布出现

## 2026-03-17 15:24 CST

- 核验“为什么没有继续自动化训练、优化”:
  - 现场检查结果:
    - 当前没有活动 `train_v2.py` / `isaaclab.sh` 训练进程
    - `wave41` 实际 run 目录为 `/home/gwh/dashgo_rl_project/autopilot/runs/gen2/20260317_150112_wave41_gen2_model694_stablehistory_extend375_long`
    - 之前 `STATE.md` 里误写成 `...150207...`
    - `wave41` 与 `wave42` 的 `run_meta.json` 都显示 `status=completed`
  - 结论:
    - 训练不是挂死，而是有限波次正常跑完后停下
    - 同时暴露出一个流程问题：之前的口径没有把“当前是否有活动训练进程”作为必须核实的事实

- 已完成防复发修正:
  - 在 `autonomous-training-overseer` skill 中加入三条硬 guard:
    - `final != daemon`
    - `finite wave != crash`
    - `state hint != ground truth`
  - 在 `discover_resume_files.py` 中加入:
    - `active_training_processes`
    - `lifecycle_status`
    - `state_latest_training_run_exists`
    - `state_best_checkpoint_exists`
    - `stale_references`
  - 在 `ERROR_TRACE.md` 中登记 `INC-2026-03-17-014`

- 当前状态切换:
  - 按用户要求，不再继续训练
  - `STATE.md` 已改为 `paused_retro_after_wave42`
  - 后续若恢复训练，必须先跑新的 resume discovery，再决定是否续训
