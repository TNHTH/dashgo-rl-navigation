#!/bin/bash
# 项目初始化脚本
# 用法: ./scripts/init-project.sh "项目名称"

PROJECT_NAME=$1

if [ -z "$PROJECT_NAME" ]; then
  echo "❌ 错误: 请提供项目名称"
  echo ""
  echo "用法: ./scripts/init-project.sh <项目名称>"
  echo "示例: ./scripts/init-project.sh \"任务管理App\""
  exit 1
fi

echo "🚀 初始化项目: $PROJECT_NAME"
echo ""

# ========================================
# 1. 创建目录结构
# ========================================
echo "📁 创建目录结构..."

mkdir -p .artifacts/phase{1..5}
mkdir -p .artifacts/security
mkdir -p .backups
mkdir -p docs
mkdir -p scripts

echo "✅ 目录结构已创建"

# ========================================
# 2. 创建INDEX.md
# ========================================
echo "📝 创建INDEX.md..."

cat > docs/INDEX.md <<EOF
# 项目Artifact索引

> **项目名称**: $PROJECT_NAME
> **创建时间**: $(date +%Y-%m-%d)
> **当前阶段**: 项目启动

---

## 📊 进度概览

\`\`\`yaml
project:
  name: "$PROJECT_NAME"
  status: "初始化"
  started: "$(date +%Y-%m-%d)"

progress:
  phase1: ⏳ 待开始 - 需求分析
  phase2: ⏳ 待开始 - 架构设计
  phase3: ⏳ 待开始 - 迭代开发
  phase4: ⏳ 待开始 - 测试验证
  phase5: ⏳ 待开始 - 部署上线
\`\`\`

---

## 📁 Artifacts

### 阶段1：需求分析 ⏳ 待开始
- 状态: 未开始

### 阶段2：架构设计 ⏳ 待开始
- 状态: 未开始

### 阶段3：迭代开发 ⏳ 待开始
#### 3a. 后端开发
- 状态: 未开始

#### 3b. 前端开发
- 状态: 未开始

#### 3c. 集成调试
- 状态: 未开始

### 阶段4：测试验证 ⏳ 待开始
- 状态: 未开始

### 阶段5：部署上线 ⏳ 待开始
- 状态: 未开始

---

## 🛡️ 安全审计

暂无安全审计记录。

---

## 📦 备份历史

暂无备份记录。

---

## 🔧 维护说明

### INDEX.md更新规则
- 每个Agent任务完成后自动更新
- 记录所有生成的artifacts路径
- 更新进度状态

### 备份规则
- 本地快照: \`.backups/phase-X-[timestamp]/\`
- Git commit: 如果Git可用
- 远程备份: 可选

---

**最后更新**: $(date)
**维护者**: TNHTH
EOF

echo "✅ INDEX.md已创建"

# ========================================
# 3. 初始化Git（如果需要）
# ========================================
echo ""
echo "🔧 检查Git状态..."

if git rev-parse --git-dir > /dev/null 2>&1; then
  echo "✅ Git仓库已存在"
  echo "   当前分支: $(git branch --show-current 2>/dev/null || echo 'N/A')"
else
  echo "⚠️  未检测到Git仓库"
  echo ""
  read -p "是否初始化Git仓库？[y/N]: " init_git

  if [ "$init_git" == "y" ] || [ "$init_git" == "Y" ]; then
    git init
    echo ".artifacts/" >> .gitignore
    echo ".backups/" >> .gitignore
    git add .
    git commit -m "chore: initialize project $PROJECT_NAME"
    echo "✅ Git仓库已初始化"
  else
    echo "⏭️  跳过Git初始化"
  fi
fi

# ========================================
# 4. 创建README
# ========================================
echo ""
echo "📖 创建README.md..."

cat > README.md <<EOF
# $PROJECT_NAME

## 项目概述

本项目使用 **TNHTH 智能Agent工作流系统** 开发。

## 开发阶段

- [ ] 阶段1: 需求分析
- [ ] 阶段2: 架构设计
- [ ] 阶段3: 迭代开发
  - [ ] 3a. 后端开发
  - [ ] 3b. 前端开发
  - [ ] 3c. 集成调试
- [ ] 阶段4: 测试验证
- [ ] 阶段5: 部署上线

## Artifacts索引

所有项目产物（需求文档、架构设计、代码等）的索引请查看：
- **[docs/INDEX.md](docs/INDEX.md)** - 完整的Artifact地图

## 备份与回滚

### 查看可用备份
\`\`\`bash
./scripts/list-backups.sh
\`\`\`

### 回滚到指定备份
\`\`\`bash
# 本地快照回滚
./scripts/rollback.sh .backups/phase-X-TIMESTAMP

# Git回滚（如果Git可用）
./scripts/rollback.sh git 1
\`\`\`

### 创建备份点
\`\`\`bash
./scripts/backup-phase.sh phase1 agent-name
\`\`\`

## 快速开始

1. **启动TNHTH**
   \`\`\`bash
   claude
   \`\`\`

2. **描述项目需求**
   \`\`\`
   我要开发一个[项目描述]
   \`\`\`

3. **系统自动识别Agent并开始工作**

## 技术栈

待阶段2架构设计完成后更新。

## 项目结构

\`\`\`
$PROJECT_NAME/
├── .artifacts/          # 所有Agent生成的产物
│   ├── phase1-product/  # 需求分析产物
│   ├── phase2-architecture/ # 架构设计产物
│   ├── phase3-backend/  # 后端开发产物
│   ├── phase3-frontend/ # 前端开发产物
│   ├── phase3-integration/ # 集成调试产物
│   ├── phase4-testing/  # 测试验证产物
│   ├── phase5-deployment/ # 部署上线产物
│   └── security/        # 安全审计报告
├── .backups/            # 备份快照
├── docs/                # 项目文档
│   └── INDEX.md         # Artifact索引
├── scripts/             # 工具脚本
│   ├── init-project.sh  # 项目初始化
│   ├── backup-phase.sh  # 创建备份
│   ├── rollback.sh      # 回滚
│   └── list-backups.sh  # 列出备份
└── README.md            # 本文件
\`\`\`

---

**创建时间**: $(date)
**工具系统**: TNHTH 智能Agent工作流 v3.0
EOF

echo "✅ README.md已创建"

# ========================================
# 5. 完成
# ========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 项目初始化完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📂 项目位置: $(pwd)"
echo "📝 项目名称: $PROJECT_NAME"
echo ""
echo "🎯 下一步操作："
echo "1. 启动TNHTH:"
echo "   claude"
echo ""
echo "2. 在TNHTH中描述项目需求，例如："
echo ""
echo "   我要开发一个$PROJECT_NAME，核心功能是..."
echo ""
echo "3. 系统将自动识别Agent并开始工作"
echo ""
echo "📚 参考文档:"
echo "   - docs/INDEX.md - Artifact索引"
echo "   - README.md - 项目说明"
echo ""
echo "🛠️  可用命令:"
echo "   ./scripts/list-backups.sh    # 查看备份"
echo "   ./scripts/backup-phase.sh phase1 product-agent  # 创建备份"
echo "   ./scripts/rollback.sh .backups/phase-X-TIMESTAMP  # 回滚"
echo ""
