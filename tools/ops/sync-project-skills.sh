#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$PROJECT_ROOT/.codex/skills.manifest.json"
CODEX_SKILL_BIN="$HOME/codex-skills-config/tools/codex-skill"
GLOBAL_SKILLS_DIR="$HOME/.codex/skills"
SOURCE_SKILLS_DIR="$HOME/codex-skills-config/skills"

if [ ! -f "$MANIFEST" ]; then
  echo "[ERROR] manifest 不存在: $MANIFEST" >&2
  exit 1
fi
if [ ! -x "$CODEX_SKILL_BIN" ]; then
  echo "[ERROR] codex-skill 不存在或不可执行: $CODEX_SKILL_BIN" >&2
  exit 1
fi

echo "[1/5] codex-skill doctor"
"$CODEX_SKILL_BIN" doctor

echo "[2/5] codex-skill validate"
"$CODEX_SKILL_BIN" validate

backup_manifest="$(mktemp)"
cp "$MANIFEST" "$backup_manifest"

required_skills=$(python3.10 - <<'PY' "$MANIFEST"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
for name in data.get('skills', {}).keys():
    print(name)
PY
)

echo "[3/5] 校验源码与全局安装"
while IFS= read -r skill; do
  [ -n "$skill" ] || continue
  if [ ! -d "$SOURCE_SKILLS_DIR/$skill" ]; then
    echo "[ERROR] 技能源码缺失: $SOURCE_SKILLS_DIR/$skill" >&2
    rm -f "$backup_manifest"
    exit 1
  fi
  if [ ! -d "$GLOBAL_SKILLS_DIR/$skill" ]; then
    echo "[INFO] 全局缺失，使用 codex-skill 安装: $skill"
    "$CODEX_SKILL_BIN" use --project "$PROJECT_ROOT" "$skill"
  fi
done <<< "$required_skills"

# 恢复项目富 manifest，避免 use 命令覆盖 groups / sources / optional_extensions
cp "$backup_manifest" "$MANIFEST"
rm -f "$backup_manifest"

echo "[4/5] manifest 完整性校验"
python3.10 - <<'PY' "$MANIFEST"
import json, sys
required_top = {'project', 'sources', 'groups', 'skills'}
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
missing = sorted(required_top - set(data.keys()))
if missing:
    raise SystemExit(f"manifest 缺字段: {', '.join(missing)}")
for skill in data['skills'].keys():
    if not data['skills'][skill].get('required', False):
        raise SystemExit(f"技能未标记 required: {skill}")
print('manifest ok')
PY

echo "[5/5] 输出项目技能矩阵"
bash "$PROJECT_ROOT/tools/ops/report-project-skills.sh"
