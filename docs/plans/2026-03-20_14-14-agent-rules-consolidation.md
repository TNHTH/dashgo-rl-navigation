# Agent 长期规则归并记录

> **创建时间**: 2026-03-20 14:14:00 CST

## 目标

将本仓库分散在 `CLAUDE.md`、`.claude/rules/*.md` 与新增 `agent.md` 中的长期代理规则收敛到单一来源，减少冲突、失效路径和历史残留。

## 工作目录与输入

- 工作目录：`/home/gwh/dashgo_rl_project`
- 扫描文件：
  - `AGENTS.md`
  - `agent.md`
  - `CLAUDE.md`
  - `.claude/rules/project-specific-rules.md`
  - `.claude/rules/isaac-lab-development-iron-rules.md`
  - `.claude/rules/file-organization.md`

## 执行步骤

### 步骤 1
- 目标：确认现有规则入口与冲突来源。
- 方法：
  - 检查仓库中是否已有 `AGENTS.md`、`agent.md`、`CLAUDE.md`
  - 搜索 `CLAUDE.md` 在仓库中的引用
- 反馈：
  - 仓库原先没有本地 `AGENTS.md` 与 `agent.md`
  - `CLAUDE.md` 体量较大，混合了动态规则、旧 trigger、历史路径和项目规则
  - 多个旧文档仍引用 `CLAUDE.md`
- 判断：
  - 需要保留 `CLAUDE.md` 作为兼容入口，但不适合继续作为长期规则正文
- 结果：
  - 确定 `AGENTS.md` 作为唯一长期规则源
- 下一步：
  - 筛选可保留的长期规则

### 步骤 2
- 目标：筛掉失效或不适合作为长期合同的旧内容。
- 方法：
  - 核对历史路径是否仍存在
  - 核对当前仓库结构与关键训练入口
  - 对比 `CLAUDE.md` 与 `.claude/rules/*.md` 的重叠和冲突
- 反馈：
  - 历史路径 `dashgo/`、`multi-agent-system/` 已不存在
  - 当前结构以 `apps/isaac/`、`src/dashgo_rl/`、`configs/`、`drivers/`、`references/dashgo/`、`workspaces/` 为准
  - `apps/isaac/train_v2.py` 中仍存在 Isaac Lab 启动顺序与配置适配等关键不变量
- 判断：
  - 应保留项目定位、官方文档优先、启动链路不变量、RSL-RL 配置适配、Sim2Real 参数追溯、资源敏感配置审慎处理等长期规则
  - 应淘汰动态规则编号、频率统计、手写 trigger、失效路径和过度僵硬的“任何修改都先确认”规则
- 结果：
  - 形成长期规则保留清单与淘汰清单
- 下一步：
  - 重写 `AGENTS.md` 并收缩兼容入口

### 步骤 3
- 目标：完成规则收敛并保留兼容入口。
- 方法：
  - 重写 `AGENTS.md`
  - 收缩 `agent.md`
  - 重写 `CLAUDE.md`
- 反馈：
  - `AGENTS.md` 现在承载唯一长期规则正文
  - `agent.md` 与 `CLAUDE.md` 仅保留兼容说明
- 判断：
  - 规则入口已经统一，后续新增长期规则只需维护一处
- 结果：
  - 完成规则归并
- 下一步：
  - 验证文件内容与 git 状态

## 保留的长期规则

- 中文回复、中文代码注释
- 结论先行
- 非琐碎请求默认包含：直接答案、步骤化依据、备选方案、立即行动建议
- 广义问题先拆解再回答
- 输出关键依据与步骤，但不承诺展示隐藏思维链
- 并行读文件、先扫环境后批量改动、不可逆操作前先列候选
- 调试先做复现链路与影响面判断
- 长时间训练、重资源评测、部署、破坏性操作前先给计划并等确认
- DashGo RL 项目是局部路径规划器，不是端到端导航器
- Isaac Lab 启动顺序和 RSL-RL 配置适配属于长期不变量
- Sim2Real 参数应从当前驱动与参考配置追溯来源
- 朝向类奖励或角度 shaping 属于高风险修改点

## 淘汰或降级的内容

- `DR-001` 这类动态规则编号、频率和优先级统计
- 手写 Auto-Load Trigger 表
- `dashgo/`、`multi-agent-system/` 等失效路径假设
- 过度僵化的“任何方案、优化、改进都必须先确认”
- 把 `CLAUDE.md` 继续当作主规则入口的工作方式

## 交付文件

- `/home/gwh/dashgo_rl_project/AGENTS.md`
- `/home/gwh/dashgo_rl_project/agent.md`
- `/home/gwh/dashgo_rl_project/CLAUDE.md`
- `/home/gwh/dashgo_rl_project/docs/plans/2026-03-20_14-14-agent-rules-consolidation.md`

## 验证方法

- 重新读取 `AGENTS.md`、`agent.md`、`CLAUDE.md`，确认规则层级与入口关系清晰。
- 检查当前仓库关键路径是否与规则中的路径一致。
- 用 `git status --short -- <files>` 确认仅记录到本次变更文件。

## 风险与后续

- 仓库内旧笔记、旧技能说明和历史 issue 仍可能继续提到 `CLAUDE.md`；这些引用暂不批量改写，以避免扩大本次改动范围。
- 若后续需要进一步去冗余，可再做一轮“将 `.claude/rules/*.md` 中仍有效的技术协议整理到 `docs/05-协议规范/`，并在文头统一标记为参考资料”的清理。
