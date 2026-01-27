# Dynamic Rules

> Auto-generated rules from dialogue patterns
> Format: YAML (see dialogue_optimizer.md Layer 3)
> Maintenance: Auto-merge when ≥ 20 rules

## Active Rules

```yaml
- id: DR-001
  created: 2026-01-17
  frequency: 1
  category: user_preference
  title: "强制中文回复"
  content: "所有回复、注释、文档必须使用中文，包括代码注释和markdown文档"
  rationale: "用户明确要求'不管写在哪，以后默认回复和注释都一定要用中文'，并强调'一定要记得'"
  impact:
    token_saving: "0%"
    user_preference: critical
    priority: highest
  status: active
  examples:
    good: "明白，会用中文回复所有内容"
    bad: "Understood, will reply in Chinese"

- id: DR-002
  created: 2026-01-17
  frequency: 1
  category: tool_usage
  title: "工具失败自动切换"
  content: "哪个好用用哪个，一个失败了就换另一个方法。不要只报告失败，要自动切换"
  rationale: "用户指出git push失败后，我没有切换到mcp__github__push_files。用户明确'哪个好用用哪个，一个失败了就换另一个方法'"
  impact:
    token_saving: "0%"
    reliability: high
    priority: highest
  status: active
  examples:
    good: "git push失败 → 自动切换mcp__github__push_files → 完成"
    bad: "git push失败 → 报告'推送失败'，等待用户指示"

- id: DR-003
  created: 2026-01-18
  frequency: 3
  category: user_preference
  title: "用户要求'具体化'时的响应策略"
  content: "当用户明确要求'具体点'、'讲详细点'、'不要泛化'时，优先给出具体案例、可执行建议、数据支撑。大幅减少或完全删除通用解释、理论框架、背景原理。但如果用户的后续问题表明需要理论支撑，则补充简要说明。"
  rationale: "用户通常要求具体化是为了快速获得可执行建议，但有时候也需要理解背后的原理。一刀切地删除所有理论会导致用户无法理解建议背后的逻辑。"
  impact:
    token_saving: "20-30%"
    user_satisfaction: high
    priority: high
  status: active
  examples:
    good: "用户说'讲具体点的配偶建议' → 直接给出ENFP/INFJ具体画像+行为模式+如何识别。如果用户后续问'为什么是ENFP'，再补充认知功能理论。"
    bad: "用户说'讲具体点的配偶建议' → 先讲MBTI理论（10分钟），再讲认知功能（10分钟），最后才给具体建议。用户已经在对话中明确表示'讲的不要太泛化'，但仍提供大量泛化内容。"

- id: DR-004
  created: 2026-01-18
  frequency: 1
  category: accuracy
  title: "引用用户私人信息前必须验证来源"
  content: "引用用户的私人经历、日记内容、人名关系前，必须先用Grep/Glob工具验证信息确实存在于用户文件中。如果无法验证，则明确说明'我从之前对话中了解到...'或'我无法确认这个信息的来源'。"
  rationale: "本次对话中人名引用来源错误，用户指出后才意识到未经验证。这是一个严重的准确性问题，会损害用户信任。"
  impact:
    token_saving: "5%"
    error_prevention: critical
    user_trust: critical
    priority: highest
  status: active
  examples:
    good: "引用前先用Grep搜索关键词，确认存在后再引用。例如：Grep('某人名') → 找到日记 → 引用原文。"
    bad: "基于记忆或之前对话历史直接引用，未验证当前文件库。例如：直接说'你在日记中说某段位太高'，但实际文件中找不到这条内容。"

- id: DR-005
  created: 2026-01-18
  frequency: 2
  category: efficiency
  title: "多文档生成时的内容重复处理策略"
  content: "当生成多个可能包含重叠内容的文档时，评估用户需求：（1）如果用户需要独立文档（每个文档可单独使用），则允许必要的重复。（2）如果用户需要节省token/避免冗余，则使用引用替代重复。（3）默认情况下，优先考虑用户的独立使用需求，其次考虑token节省。"
  rationale: "原规则'避免重复'太死板，有时候用户需要每个文档都是完整的。应该根据用户需求灵活处理。例如：个人分析总览和MBTI深度分析，如果总览是快速入口，就应该包含完整内容；如果总览是索引，就应该引用详细文档。"
  impact:
    token_saving: "10-20%"
    user_experience: high
    priority: medium
  status: active
  examples:
    good: "生成个人分析总览时，先评估用户是需要'快速了解的入口'还是'完整的独立文档'。根据需求决定是否重复MBTI分析内容。"
    bad: "无论用户需求如何，都强制避免重复，导致'个人分析总览'无法独立使用，必须跳转才能看到完整信息。"

- id: DR-006
  created: 2026-01-19
  frequency: 1
  category: user_preference
  title: "极端高效对话风格（方案1）"
  content: "使用零前奏、直接输出答案、要点列表为主、删除所有泛化内容、信息密度最大化的对话风格。专业术语必须配通俗解释+用户案例，不能只说理论。"
  rationale: "用户测试了方案1（极端高效型）后反馈'这种对话很高效'。用户明确指出：不要泛化内容（'总的来说'、'值得注意的是'等套话），专业术语要解释清楚（'Se劣势'太专业，需要配'脑子里在打架'这种用户案例），要结合用户实际情况而非纯理论。"
  impact:
    token_saving: "30-40%"
    user_satisfaction: critical
    clarity: critical
    priority: highest
  status: active
  examples:
    good: "执行力问题=Se劣势（外向感觉功能弱）。解释：Se负责'当下行动'，Se弱=想得多做得少。你的案例：10月11日'一直磨蹭磨蹭到一点'。行动：1.5分钟启动法 2.删除抖音 3.只追踪1个习惯。"
    bad: "执行力问题主要是因为多方面因素造成的。值得注意的是，INTJ类型的人通常会面临这种情况。实际上，你需要从多个角度来解决这个问题。（42字，信息量=0）"

- id: DR-007
  created: 2026-01-19
  frequency: 1
  category: accuracy
  title: "结合用户个人特征而非泛化理论"
  content: "分析用户问题时，必须结合用户的具体情况、日记内容、行为模式，不能只讲理论框架（如'INTJ都是这样'）。引用用户的原话和案例，而非泛化的理论描述。避免'攻略游戏'式的建议，聚焦于'适合什么类型'和'什么情况不要错过'。"
  rationale: "用户指出'你确定是针对我来分析的不是根据intj来分析的，我的一些个性你有考虑到吗'。用户只需要知道适合什么类型+遇到什么类型不要错过，不需要'怎么攻略'的行动建议。之前的分析太理论化，没有结合用户的实际情况（被动社交、工具性为主、焦虑螺旋、AI依赖等）。"
  impact:
    relevance: critical
    user_trust: high
    accuracy: critical
    priority: highest
  status: active
  examples:
    good: "你真正需要的人：1.温和主动的技术同行（理解你的AI追求，不会强势控制）2.成熟理性的思考者（能聊自由意志，不会同辈比较）。遇到这些信号不要错过：对方主动找你>2次、能聊深度话题>30分钟、温和且靠谱。"
    bad: "INTJ最适合INFJ或ENFJ。INFJ的主导功能是Ni，能理解你的深度思考。ENFJ的Fe功能能主动发起，弥补你的被动。建议你参加哲学读书会、技术会议来遇到这些人。（太理论化，没有结合用户的实际情况）"

- id: DR-008
  created: 2026-01-20
  frequency: 1
  category: agent_design
  title: "Agent vs Skill的区别和生成规范"
  content: "创建Agent时必须区分Agent和Skill：（1）Agent=可以主动调用的自主实体，在multi-agent-system/agents/目录下创建.prompt.md文件，用Task工具调用（2）Skill=用户手动触发的工具，在.claude/skills/目录下创建.json文件，用户输入/skillname调用。（3）如果需要AI主动判断何时使用，必须创建Agent而非Skill。（4）Agent必须具备主动研究能力（自动扫描数据、识别模式、生成分析）"
  rationale: "用户创建了self-reflection-agent（实际是Skill），期望它能自动调用，但Skill不会自动触发。用户明确要求'不仅作为对话者，更作为具备主动研究能力的心理分析师'。必须记住：Skill只能手动调用，Agent可以主动调用。"
  impact:
    automation: critical
    user_expectation: high
    design_quality: critical
    priority: highest
  status: active
  examples:
    good: "用户需要心理分析 → 创建psychological-counselor.prompt.md（Agent）→ AI识别到心理问题时用Task工具主动调用 → 自动扫描用户数据 → 生成分析报告"
    bad: "用户需要心理分析 → 创建analyze.json（Skill）→ 等待用户手动输入/analyze → 不会自动触发 → 用户困惑'为什么agent不会自动调用'"

- id: DR-009
  created: 2026-01-20
  frequency: 1
  category: context_management
  title: "每个对话结束时记录到临时文档"
  content: "每个对话结束时，必须将对话记录追加到对应的临时文档中。文档位置：.claude-temp/{话题名}_{日期}/对话记录_{序号}.md。这样每个对话都不害怕超过上下文限制，可以随时恢复上下文。"
  rationale: "用户要求'记得以后每个对话都这样，这样每个对话都不害怕超过上下文限制了'。当对话因上下文限制被压缩后，可以通过读取临时文档快速恢复之前的对话内容。"
  impact:
    context_continuity: critical
    user_experience: high
    data_persistence: critical
    priority: highest
  status: active
  examples:
    good: "某话题对话 → .claude-temp/某话题对话记录_20260119/对话记录_01.md → 每次对话后追加记录 → 压缩后可快速恢复"
    bad: "长时间对话 → 上下文满了 → 被压缩 → 用户问'我们之前聊到哪了' → AI说'我也不知道' → 对话断裂"

- id: DR-010
  created: 2026-01-20
  frequency: 1
  category: accuracy
  title: "禁止武断下结论，必须基于充分证据"
  content: "在信息不足的情况下，禁止给出具体百分比、明确判断或绝对结论。当用户表示'不知道'、'不清楚'时，不要替用户做决定或给出标准答案，而是通过苏格拉底式提问帮助用户自己发现答案。对于需要判断的内容，必须说明'基于有限信息'、'可能'、'需要更多信息'。"
  rationale: "用户指出在分析某人符合度时直接给出'80%符合'的结论太武断。用户明确表示'我也不知道我喜欢什么性格的女生'，但AI却直接判断理想型是ESFJ/ENFJ。用户三次强调'不要武断下结论'。"
  impact:
    accuracy: critical
    user_trust: critical
    analysis_quality: high
    priority: highest
  status: active
  examples:
    good: "用户：我理想型是什么样的？\nAI：你提到喜欢'好看+不脾气大+关系好'的女生。我们可以慢慢探索：①你之前遇到过的女生中，谁让你觉得最舒服？为什么？②你和朋友相处时，什么样的相处模式让你觉得开心？③你绝对不能接受什么样的行为？"
    bad: "用户：我理想型是什么样的？\nAI：你的理想型是ESFJ。某人80%符合你的理想型。你应该去找ESFJ类型的女生。（用户只提供了少量信息，AI就给出了绝对判断）"

- id: DR-011
  created: 2026-01-22
  frequency: 1
  category: efficiency
  title: "方案设计先给列表，确认后再详细"
  content: "当用户要求'分析哪些可以优化/添加'时，先给简洁列表（不超过1KB），询问用户'哪些你觉得有用？'，等用户确认后再详细设计。禁止直接创建详细方案。"
  rationale: "本次对话创建17KB详细方案，但用户只需要'看看有什么'，浪费15KB token。应该先给简洁列表（<1KB），确认用户需求后再详细设计。"
  impact:
    token_saving: "85-95%"
    user_satisfaction: high
    priority: high
  status: active
  examples:
    good: "用户：分析哪些可以优化？\nAI：我发现可添加X/Y/Z（简洁列表<1KB），你觉得哪些有用？"
    bad: "用户：分析哪些可以优化？\nAI：创建17KB详细方案，包含完整代码示例和文件内容"

- id: DR-012
  created: 2026-01-22
  frequency: 1
  category: documentation
  title: "文档时间戳包含时分秒"
  content: "所有创建的文档必须包含完整时间戳（YYYY-MM-DD HH:MM:SS），格式：'创建时间: 2026-01-22 14:30:00'。只用日期（2026-01-22）是不够的。"
  rationale: "用户明确指出文档只有日期无法分辨修改先后，需要精确到秒的时间戳。"
  impact:
    clarity: critical
    version_control: high
    priority: high
  status: active
  examples:
    good: "> **创建时间**: 2026-01-22 14:30:00"
    bad: "> **日期**: 2026-01-22"

- id: DR-014
  created: 2026-01-22
  frequency: 1
  category: user_preference
  title: "评估关键词默认使用dialogue-optimizer"
  content: "当用户说'评估'或'evaluate'时，默认调用dialogue-optimizer skill进行Full Assessment，不需要额外确认。"
  rationale: "用户明确表示'以后我说评估都是dialogue-optimizer'，这是一个明确的偏好设置。"
  impact:
    efficiency: "20%"
    user_satisfaction: high
  status: active
  examples:
    good: "用户：评估本次对话 → AI：调用dialogue-optimizer Full Assessment"
    bad: "用户：评估本次对话 → AI：你要我评估什么？代码？文档？还是整体？"

- id: DR-017
  created: 2026-01-22
  frequency: 3
  category: efficiency
  title: "多文件读取必须并行"
  content: "当需要读取多个文件时，必须使用Glob+并行Read，而非串行读取。7个文件串行读取=7t，并行读取=2.3t，节省66%时间。"
  rationale: "本次对话读取7个agent文件时串行执行，浪费了66%时间。Glob工具可快速定位文件，Read工具支持并行调用。"
  impact:
    token_saving: "0%"
    time_saving: "66%"
    performance: critical
    priority: high
  status: active
  examples:
    good: "Glob('agents/*.prompt.md') → Read(product, architect, frontend) 同时读取3个文件"
    bad: "Read('product.prompt.md') → Read('architect.prompt.md') → Read('frontend.prompt.md') 串行3次"

- id: DR-018
  created: 2026-01-23
  frequency: 1
  category: agent_system
  title: "Trigger 1扩展为7-Agent独立触发系统"
  content: "Trigger 1从单一'复杂项目开发模式'扩展为'智能开发Agent系统'，支持整体流水线触发和7个Agent单独触发。每个Agent有独立的关键词列表：Product（需求分析）、Architect（架构设计）、Backend（后端开发）、Frontend（前端开发）、Code-Reviewer（代码审查与优化）、Docs（文档编写）、DevOps（部署运维）。"
  rationale: "用户说'分析优化'时没有触发Code-Reviewer Agent，原因是CLAUDE.md中缺少代码优化相关的Trigger。虽然存在code-reviewer.prompt.md，但没有对应的触发条件，违反了DR-015规则。"
  impact:
    automation: critical
    user_experience: high
    agent_coverage: complete
    priority: highest
  status: active
  examples:
    good: "用户说'分析优化' → AI识别触发Code-Reviewer Agent → Read code-reviewer.prompt.md → 执行代码分析"
    good: "用户说'写需求' → AI识别触发Product Agent → Read product.prompt.md → 撰写PRD"
    good: "用户说'新建项目' → AI触发整体流水线 → 按Phase 1-5顺序执行所有Agent"
    bad: "用户说'分析优化' → AI使用general-purpose agent（不专业） → 没有调用code-reviewer"

- id: DR-019
  created: 2026-01-23
  frequency: 1
  category: agent_creation
  title: "创建Robot-Nav-Architect Agent - 机器人导航项目架构师"
  content: "创建了专门的机器人导航项目Agent（robot-nav-architect.prompt.md），专精于NVIDIA Isaac Lab、RSL-RL、DRL局部导航、Ubuntu 20.04。核心原则：官方文档优先（最重要）。思维模式：历史导向（回顾项目历史）、博采众长（多源验证）、稳定性优先（对报错敏感）、代码高手（简洁高效+连通性）。工具集成：Context7查询官方文档、GitHub获取官方示例、Tavily搜索论文、本地工具扫描项目历史。禁忌：严禁恢复朝向奖励（原地转圈）、严禁大幅提高学习率、严禁使用未经验证的API。"
  rationale: "用户明确要求创建机器人导航项目的Agent，并强调'一定要严格遵守官方的文档和对应软件的规则'。通用Agent无法满足领域特定的需求，需要具备项目历史记忆和官方规范验证能力的专业Agent。"
  impact:
    domain_expertise: critical
    official_compliance: critical
    project_success: high
    priority: highest
  status: active
  examples:
    good: "用户说'优化奖励函数' → AI触发Robot-Nav-Architect → 并行查询官方文档+扫描项目历史+搜索官方示例 → 基于官方规范输出代码"
    good: "用户说'训练不稳定' → AI触发Robot-Nav-Architect → 检查历史错误日志 → 验证超参数是否符合官方推荐 → 给出稳定方案"

- id: DR-020
  created: 2026-01-27
  frequency: 1
  category: project_control
  title: "训练和脚本执行前必须获得用户明确许可"
  content: "任何训练命令（train_v2.py、python train.py等）或测试脚本（play.py、验证脚本等）的执行，必须先询问用户并获得明确同意（'可以执行'、'yes'、'go ahead'、'执行'等）。严禁在用户未明确同意的情况下启动训练或测试。"
  rationale: "用户明确要求'没有经过我允许，不准私自启动训练或者脚本测试，进行前必须得问我并且我同意，将这个精简后写进CLAUDE.md'。这是项目控制的核心要求，避免未经授权的长时间运行操作浪费资源或引入风险。"
  impact:
    resource_control: critical
    risk_prevention: critical
    user_trust: critical
    priority: highest
  status: active
  examples:
    good: "AI：需要启动训练吗？（用户：yes）→ AI执行训练"
    good: "AI：准备执行测试脚本play.py，是否继续？（用户：go ahead）→ AI执行"
    bad: "用户说'修改参数' → AI直接启动训练（违反规则）"
    bad: "用户说'检查配置' → AI自动运行play.py测试（违反规则）"
    bad: "用户说'优化奖励' → AI直接给代码（未验证官方规范） → 可能违反Isaac Lab API规范 → 训练失败"

```

## Rule Statistics
- Total rules: 20
- Active: 20
- Deprecated: 0
- Last updated: 2026-01-27
- Next merge check: At 25 rules
