#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$PROJECT_ROOT/.codex/skills.manifest.json"
GLOBAL_SKILLS_DIR="$HOME/.codex/skills"
SOURCE_SKILLS_DIR="$HOME/codex-skills-config/skills"

if [ ! -f "$MANIFEST" ]; then
  echo "[ERROR] manifest 不存在: $MANIFEST" >&2
  exit 1
fi

python3.10 - <<'PY' "$MANIFEST" "$GLOBAL_SKILLS_DIR" "$SOURCE_SKILLS_DIR"
import json
import os
import subprocess
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
global_dir = Path(sys.argv[2])
source_dir = Path(sys.argv[3])
data = json.loads(manifest.read_text(encoding="utf-8"))

print("[项目]")
project = data.get("project", {})
print(f"- name: {project.get('name', 'unknown')}")
print(f"- governance_backend: {project.get('governance_backend', 'unknown')}")
print(f"- manifest_version: {project.get('manifest_version', 'unknown')}")
print()

print("[分组]")
groups = data.get("groups", {})
for group_name, skills in groups.items():
    print(f"- {group_name}: {', '.join(skills)}")
print()

print("[技能矩阵]")
skills = data.get("skills", {})
for skill_name in skills:
    global_ok = (global_dir / skill_name).is_dir()
    source_ok = (source_dir / skill_name).is_dir()
    status = []
    status.append("global:ok" if global_ok else "global:missing")
    status.append("source:ok" if source_ok else "source:missing")
    print(f"- {skill_name}: {' | '.join(status)}")
print()

print("[来源摘要]")
for source_name, source_meta in data.get("sources", {}).items():
    repo = source_meta.get("repo", "unknown")
    skills_list = ", ".join(source_meta.get("skills", []))
    print(f"- {source_name}: {repo}")
    print(f"  skills: {skills_list}")
print()

print("[GitHub CLI]")
gh_path = subprocess.run(["bash", "-lc", "command -v gh || true"], capture_output=True, text=True).stdout.strip()
if not gh_path:
    print("- gh: missing")
    print("- auth: 未检测（未安装 gh）")
else:
    print(f"- gh: {gh_path}")
    auth = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if auth.returncode == 0:
        print("- auth: ok")
    else:
        print("- auth: not-authenticated")
        print("- prerequisite: 使用 `gh auth login` 完成 GitHub 认证后再执行 Issue / PR 生命周期操作")
print()

print("[可选扩展]")
for item in data.get("optional_extensions", []):
    print(f"- {item.get('skill')}: {item.get('reason')}")
PY
