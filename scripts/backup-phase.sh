#!/bin/bash
# 阶段备份脚本 - 三层备份机制
# 用法: ./scripts/backup-phase.sh <阶段> <Agent名称>
# 示例: ./scripts/backup-phase.sh phase1 product-agent

PHASE=$1
AGENT=$2

if [ -z "$PHASE" ] || [ -z "$AGENT" ]; then
  echo "❌ 错误: 请提供阶段和Agent名称"
  echo ""
  echo "用法: ./scripts/backup-phase.sh <阶段> <Agent名称>"
  echo ""
  echo "示例:"
  echo "  ./scripts/backup-phase.sh phase1 product-agent"
  echo "  ./scripts/backup-phase.sh phase2 architect-agent"
  echo "  ./scripts/backup-phase.sh phase3a backend-agent"
  exit 1
fi

echo "📦 创建三层备份..."
echo "阶段: $PHASE"
echo "Agent: $AGENT"
echo ""

# ========================================
# 1. 本地快照（始终执行，最可靠）
# ========================================
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR=".backups/phase-${PHASE}-${TIMESTAMP}"
mkdir -p "$BACKUP_DIR"

echo "📂 [1/3] 创建本地快照..."

# 备份artifacts
if [ -d ".artifacts" ]; then
  cp -r .artifacts/ "$BACKUP_DIR/"
  ARTIFACT_COUNT=$(find "$BACKUP_DIR/.artifacts" -type f 2>/dev/null | wc -l)
  echo "   ✅ Artifacts: $ARTIFACT_COUNT 个文件"
else
  echo "   ⚠️  .artifacts/ 目录不存在（可能首次备份）"
fi

# 备份INDEX.md
if [ -f "docs/INDEX.md" ]; then
  mkdir -p "$BACKUP_DIR/docs"
  cp docs/INDEX.md "$BACKUP_DIR/docs/"
  echo "   ✅ INDEX.md: 已备份"
fi

# 记录元数据
cat > "$BACKUP_DIR/backup-metadata.txt" <<EOF
Phase: $PHASE
Agent: $AGENT
Timestamp: $(date)
Backup Type: Local Snapshot
Artifact Count: ${ARTIFACT_COUNT:-0}
EOF

echo "   ✅ 本地快照: $BACKUP_DIR"

# ========================================
# 2. Git提交（如果Git可用）
# ========================================
GIT_COMMIT_HASH=""

echo ""
echo "📦 [2/3] 检查Git可用性..."

if git rev-parse --git-dir > /dev/null 2>&1; then
  echo "   ✅ Git可用，创建commit..."

  # 添加文件
  git add .artifacts/ docs/INDEX.md 2>/dev/null || git add .artifacts/

  # 创建commit message
  cat > .git/commit-msg.txt <<EOF
feat(${PHASE}): complete ${PHASE} by ${AGENT}

- ${AGENT}: 阶段任务完成
- Local snapshot: ${BACKUP_DIR}
- Artifacts: ${ARTIFACT_COUNT:-0} files
- Updated INDEX.md

Timestamp: $(date)
EOF

  # 提交
  if git commit -F .git/commit-msg.txt > /dev/null 2>&1; then
    GIT_COMMIT_HASH=$(git rev-parse --short HEAD)
    rm -f .git/commit-msg.txt
    echo "   ✅ Git commit: $GIT_COMMIT_HASH"

    # 更新元数据
    echo "Git Commit: $GIT_COMMIT_HASH" >> "$BACKUP_DIR/backup-metadata.txt"
  else
    echo "   ⚠️  Git commit失败（可能没有变更）"
    rm -f .git/commit-msg.txt
  fi

else
  echo "   ⚠️  Git不可用，跳过Git commit"
fi

# ========================================
# 3. 远程推送（可选，询问用户）
# ========================================
echo ""
if [ -n "$GIT_COMMIT_HASH" ]; then
  echo "📦 [3/3] 远程推送..."
  echo ""
  read -p "   是否推送到GitHub？[y/N]: " push_to_github

  if [ "$push_to_github" == "y" ] || [ "$push_to_github" == "Y" ]; then
    echo "   正在推送..."
    if git push origin $(git branch --show-current 2>/dev/null || echo "main") 2>&1; then
      echo "   ✅ 已推送到GitHub"
      echo "Git Push: $(git log -1 --oneline)" >> "$BACKUP_DIR/backup-metadata.txt"
    else
      echo "   ⚠️  Git push失败"
      echo "   💡 提示: 可以稍后手动执行 'git push'"
    fi
  else
    echo "   ⏭️  跳过远程推送"
  fi
else
  echo "📦 [3/3] 远程推送..."
  echo "   ⏭️  跳过（无Git commit）"
fi

# ========================================
# 4. 备份摘要
# ========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 备份完成"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 备份摘要:"
echo "   阶段:     $PHASE"
echo "   Agent:    $AGENT"
echo "   时间戳:   $(date +%Y-%m-%d\ %H:%M:%S)"
echo ""
echo "📦 备份位置:"
if [ -n "$GIT_COMMIT_HASH" ]; then
  echo "   本地快照: ✅ $BACKUP_DIR"
  echo "   Git commit: ✅ $GIT_COMMIT_HASH"
else
  echo "   本地快照: ✅ $BACKUP_DIR"
  echo "   Git commit: ⚠️  不可用"
fi
echo ""
echo "🔄 回滚命令:"
echo "   ./scripts/rollback.sh $BACKUP_DIR"
if [ -n "$GIT_COMMIT_HASH" ]; then
  echo "   ./scripts/rollback.sh git 1"
fi
echo ""
