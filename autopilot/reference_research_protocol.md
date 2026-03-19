# DashGo 改动前参考研究协议

## 目标

- 每次训练合同、奖励、课程、路径语义、脱困逻辑或部署策略要改动前，必须先做一次外部参考研究。
- 训练目标默认同时覆盖两件事：
  - 避障能力
  - 脱困能力
- 两者权重相同，不能只追求“绕得开”，也不能只追求“退得出”。

## 强制流程

1. 先查官方或一手来源：
   - 官方文档
   - 官方仓库 README / docs
   - 原始论文或 arXiv
2. 每次至少提炼 2 到 3 条可迁移策略。
3. 把外部策略映射到 DashGo 当前合同：
   - 当前观测里已经有什么
   - 当前奖励里已经有什么
   - 当前课程与 reset 里缺什么
4. 只选择 1 个 focused change 进入下一波训练。
5. 在 `autopilot/findings.md` 和 `autopilot/progress.md` 里记录：
   - 参考来源
   - 借鉴点
   - 为什么本波只做这个变量
6. 如果没有先完成这一步，不启动新的训练合同改动波次。

## 本轮已确认的可迁移方向

- Nav2 的恢复行为不是把“倒车概率”全局拉高，而是把 `backup / spin / wait / drive_on_heading` 当成明确恢复语义。
- Arena-Rosnav 的重点不是单一 reward，而是用 `Task Generator` 和 benchmark scenario 持续喂高动态、可参数化场景。
- TurtleBot3 DRL Navigation 把 `backward motion` 和 `frame stacking` 视为可控开关，而不是默认万金油。

## 对 DashGo 的直接约束

- 新波次改动前，先回答：
  - 这次改动主要补避障，还是补脱困，还是同时补两者？
  - 外部成熟方案是怎么做的？
  - DashGo 当前代码里已经具备哪些前提，缺的到底是什么？
- 如果答案仍然模糊，先继续研究，不直接改合同。
