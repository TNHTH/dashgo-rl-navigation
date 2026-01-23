#!/bin/bash
# Claude Code 启动脚本
# 确保环境变量正确设置

# 加载 nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# 设置 Claude Code API 配置
export ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic"
export ANTHROPIC_AUTH_TOKEN="cb873b854b084e9b81d9407da9b38d29.GBp8KsgR9bg8I620"

# 显示配置信息
echo "=========================================="
echo "Claude Code 启动配置"
echo "=========================================="
echo "ANTHROPIC_BASE_URL: $ANTHROPIC_BASE_URL"
echo "ANTHROPIC_AUTH_TOKEN: 已设置"
echo "=========================================="
echo ""

# 启动 Claude Code
claude "$@"
