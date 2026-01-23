#!/bin/bash
# 列出可用备份脚本
# 用法: ./scripts/list-backups.sh

echo "📦 可用备份列表"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ========================================
# 1. Git提交历史
# ========================================
if git rev-parse --git-dir > /dev/null 2>&1; then
  echo "🔖 Git提交历史:"
  echo "   ────────────────────────────────────"
  git log --oneline -10 | sed 's/^/   /'
  echo ""
else
  echo "⚠️  Git不可用"
  echo ""
fi

# ========================================
# 2. 本地快照备份
# ========================================
echo "📁 本地快照备份:"
echo "   ────────────────────────────────────"

if [ -d ".backups" ]; then
  BACKUP_COUNT=0

  # 按时间倒序排列备份
  for backup in $(ls -dt .backups/phase-* 2>/dev/null); do
    if [ -d "$backup" ]; then
      BACKUP_NAME=$(basename "$backup")
      BACKUP_COUNT=$((BACKUP_COUNT + 1))

      # 提取时间戳
      TIMESTAMP=$(echo "$BACKUP_NAME" | grep -oP '\d{8}-\d{6}' || echo "Unknown")

      # 显示备份信息
      echo ""
      echo "   📂 $BACKUP_NAME"

      # 显示元数据
      if [ -f "$backup/backup-metadata.txt" ]; then
        while IFS= read -r line; do
          # 跳过空行
          [ -z "$line" ] && continue

          # 高亮重要信息
          case "$line" in
            Phase:*|Agent:*)
              echo "      🔹 $line"
              ;;
            Timestamp:*)
              echo "      🕐 $line"
              ;;
            *)
              echo "      📝 $line"
              ;;
          esac
        done < "$backup/backup-metadata.txt"
      else
        echo "      (无元数据)"
      fi

      # 显示文件数量
      if [ -d "$backup/.artifacts" ]; then
        FILE_COUNT=$(find "$backup/.artifacts" -type f 2>/dev/null | wc -l)
        DIR_COUNT=$(find "$backup/.artifacts" -type d 2>/dev/null | wc -l)
        echo "      📊 $FILE_COUNT 个文件, $DIR_COUNT 个目录"
      fi

      # 只显示前10个
      if [ $BACKUP_COUNT -ge 10 ]; then
        echo ""
        echo "   ... (还有更多备份，只显示前10个)"
        break
      fi
    fi
  done

  if [ $BACKUP_COUNT -eq 0 ]; then
    echo "   (暂无本地快照备份)"
  fi

else
  echo "   (.backups/ 目录不存在)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 回滚命令:"
echo ""
echo "   # 本地快照回滚"
echo "   ./scripts/rollback.sh .backups/phase-X-TIMESTAMP"
echo ""
echo "   # Git回滚1步"
echo "   ./scripts/rollback.sh git 1"
echo ""
echo "   # 查看完整备份列表"
echo "   ls -la .backups/"
echo ""
