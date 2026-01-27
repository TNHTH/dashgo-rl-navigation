#!/bin/bash
# Dialogue Optimizer Recovery Script
# Usage: ./.claude/scripts/restore_optimizer.sh

echo "🔧 Dialogue Optimizer Emergency Recovery"
echo "========================================"
echo ""

# Backup current rules
if [ -f ".claude/rules/dynamic_rules.md" ]; then
    cp .claude/rules/dynamic_rules.md .claude/rules/dynamic_rules.md.backup.$(date +%Y%m%d_%H%M%S)
    echo "✅ Backed up dynamic_rules.md"
fi

# Restore core protected files
echo "🔄 Restoring protected files..."
git checkout HEAD -- CLAUDE.md 2>/dev/null && echo "✅ CLAUDE.md restored" || echo "⚠️  CLAUDE.md: Not in git or no changes"
git checkout HEAD -- .claude/skills/dialogue_optimizer.md 2>/dev/null && echo "✅ dialogue_optimizer.md restored" || echo "⚠️  dialogue_optimizer.md: Not in git or no changes"

echo ""
echo "📋 Recovery complete"
echo "💡 Tips:"
echo "  - Review backups in .claude/rules/"
echo "  - If not in git, manually restore from backups"
echo "  - Check file permissions if issues persist"
