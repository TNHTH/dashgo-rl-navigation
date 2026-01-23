#!/bin/bash
# 智能回滚脚本 - 支持本地快照和Git回滚
# 用法1: ./scripts/rollback.sh .backups/phase-X-TIMESTAMP  # 本地快照回滚
# 用法2: ./scripts/rollback.sh git [N]                 # Git回滚N步
# 用法3: ./scripts/rollback.sh git <commit-hash>        # Git回滚到指定commit

TARGET=$1

if [ -z "$TARGET" ]; then
  echo "❌ 错误: 请指定回滚目标"
  echo ""
  echo "用法:"
  echo "  本地快照回滚:  ./scripts/rollback.sh .backups/phase-X-TIMESTAMP"
  echo "  Git回滚1步:    ./scripts/rollback.sh git 1"
  echo "  Git回滚N步:    ./scripts/rollback.sh git <N>"
  echo "  Git回滚到hash: ./scripts/rollback.sh git <commit-hash>"
  echo ""
  echo "可用备份："
  ./scripts/list-backups.sh | head -20
  exit 1
fi

# ========================================
# Git回滚
# ========================================
if [ "$TARGET" == "git" ]; then
  STEPS=${2:-1}

  echo "🔄 Git回滚"
  echo ""

  # 检查Git是否可用
  if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Git不可用，无法执行Git回滚"
    echo ""
    echo "建议："
    echo "  1. 初始化Git: git init"
    echo "  2. 使用本地快照回滚:"
    echo "     ./scripts/list-backups.sh"
    exit 1
  fi

  # 显示当前状态
  echo "当前状态:"
  git log --oneline -3
  echo ""

  # 确认回滚
  echo "⚠️  警告: 这将丢弃最近的 $STEPS 个提交"
  read -p "确认回滚？[yes/N]: " confirm

  if [ "$confirm" != "yes" ]; then
    echo "❌ 已取消回滚"
    exit 0
  fi

  # 回滚前备份当前状态
  TIMESTAMP=$(date +%Y%m%d-%H%M%S)
  SAFE_BACKUP=".backups/before-git-reset-${TIMESTAMP}"
  mkdir -p "$SAFE_BACKUP"

  if [ -d ".artifacts" ]; then
    cp -r .artifacts/ "$SAFE_BACKUP/" 2>/dev/null
  fi

  if [ -f "docs/INDEX.md" ]; then
    mkdir -p "$SAFE_BACKUP/docs"
    cp docs/INDEX.md "$SAFE_BACKUP/docs/"
  fi

  echo "💾 当前状态已备份: $SAFE_BACKUP"
  echo ""

  # 执行回滚
  git reset --hard HEAD~$STEPS

  echo ""
  echo "✅ Git回滚完成"
  echo ""
  echo "回滚后状态:"
  git log --oneline -3
  echo ""
  echo "💾 如需撤销回滚，备份在: $SAFE_BACKUP"

  exit 0
fi

# ========================================
# Git回滚到指定commit hash
# ========================================
if git rev-parse --verify "$TARGET" > /dev/null 2>&1; then
  echo "🔄 Git回滚到commit: $TARGET"
  echo ""

  # 检查Git是否可用
  if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Git不可用"
    exit 1
  fi

  # 显示commit信息
  echo "目标commit:"
  git log -1 --oneline $TARGET
  echo ""

  # 确认回滚
  echo "⚠️  警告: 这将重置到指定commit"
  read -p "确认回滚？[yes/N]: " confirm

  if [ "$confirm" != "yes" ]; then
    echo "❌ 已取消回滚"
    exit 0
  fi

  # 回滚前备份
  TIMESTAMP=$(date +%Y%m%d-%H%M%S)
  SAFE_BACKUP=".backups/before-git-reset-${TIMESTAMP}"
  mkdir -p "$SAFE_BACKUP"

  if [ -d ".artifacts" ]; then
    cp -r .artifacts/ "$SAFE_BACKUP/" 2>/dev/null
  fi

  if [ -f "docs/INDEX.md" ]; then
    mkdir -p "$SAFE_BACKUP/docs"
    cp docs/INDEX.md "$SAFE_BACKUP/docs/"
  fi

  echo "💾 当前状态已备份: $SAFE_BACKUP"
  echo ""

  # 执行回滚
  git reset --hard "$TARGET"

  echo ""
  echo "✅ Git回滚完成"
  echo ""
  echo "回滚后状态:"
  git log --oneline -3
  echo ""
  echo "💾 如需撤销回滚，备份在: $SAFE_BACKUP"

  exit 0
fi

# ========================================
# 本地快照回滚
# ========================================
if [ ! -d "$TARGET" ]; then
  echo "❌ 错误: 备份目录不存在"
  echo "   目标: $TARGET"
  echo ""
  echo "可用备份："
  ./scripts/list-backups.sh | head -20
  exit 1
fi

echo "🔄 回滚到本地快照"
echo ""
echo "📂 目标: $TARGET"
echo ""

# 显示回滚信息
if [ -f "$TARGET/backup-metadata.txt" ]; then
  echo "📝 备份信息:"
  cat "$TARGET/backup-metadata.txt"
  echo ""
fi

# 显示将要被覆盖的文件
echo "⚠️  以下内容将被覆盖:"
if [ -d ".artifacts" ]; then
  echo "   .artifacts/ ($(find .artifacts -type f 2>/dev/null | wc -l) 个文件)"
fi
if [ -f "docs/INDEX.md" ]; then
  echo "   docs/INDEX.md"
fi
echo ""

# 确认回滚
read -p "确认回滚？[yes/N]: " confirm

if [ "$confirm" != "yes" ]; then
  echo "❌ 已取消回滚"
  exit 0
fi

# 回滚前备份当前状态
CURRENT_BACKUP=".backups/before-rollback-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$CURRENT_BACKUP"

if [ -d ".artifacts" ]; then
  cp -r .artifacts/ "$CURRENT_BACKUP/" 2>/dev/null
  echo "💾 当前.artifacts/已备份: $CURRENT_BACKUP"
fi

if [ -f "docs/INDEX.md" ]; then
  mkdir -p "$CURRENT_BACKUP/docs"
  cp docs/INDEX.md "$CURRENT_BACKUP/docs/"
  echo "💾 当前INDEX.md已备份: $CURRENT_BACKUP/docs/"
fi

echo ""

# 执行回滚
echo "🔄 正在回滚..."

# 删除现有artifacts
if [ -d ".artifacts" ]; then
  rm -rf .artifacts/ 2>/dev/null
fi

# 从备份恢复artifacts
if [ -d "$TARGET/.artifacts" ]; then
  cp -r "$TARGET/.artifacts" .artifacts/ 2>/dev/null
  echo "✅ .artifacts/ 已恢复"
fi

# 恢复INDEX.md
if [ -f "$TARGET/docs/INDEX.md" ]; then
  cp "$TARGET/docs/INDEX.md" docs/INDEX.md 2>/dev/null
  echo "✅ docs/INDEX.md 已恢复"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 回滚完成"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 回滚摘要:"
echo "   从: $TARGET"
echo "   到: $(pwd)"
echo ""
echo "💾 如需撤销回滚，备份在:"
echo "   $CURRENT_BACKUP"
echo ""

# 显示恢复后的文件
if [ -f "docs/INDEX.md" ]; then
  echo "📄 已恢复的INDEX.md内容预览:"
  echo "   ─────────────────────────────"
  head -20 docs/INDEX.md | sed 's/^/   /'
  echo "   ─────────────────────────────"
  echo "   (完整内容: docs/INDEX.md)"
fi
