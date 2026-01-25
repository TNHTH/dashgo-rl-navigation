#!/usr/bin/env python3
"""
DashGo 自动分析脚本

功能：
1. 分析训练日志（TensorBoard事件文件）
2. 生成训练报告
3. 提出优化建议
4. 自动修改训练参数

用法：
    python3 auto_analyze.py [mode]
    mode: "auto" | "interactive"
"""

import os
import sys
import glob
import re
from collections import defaultdict
from datetime import datetime

# 尝试导入TensorBoard（如果可用）
try:
    from tensorboard.backend.event_processing import event_accumulator
    import tensorflow as tf
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard未安装，使用简化分析模式")

PROJECT_DIR = "/home/gwh/dashgo_rl_project"
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
ISSUE_DIR = os.path.join(PROJECT_DIR, "issues")


class TrainingAnalyzer:
    """训练分析器"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.final_metrics = {}

    def parse_tensorboard_logs(self):
        """解析TensorBoard日志"""
        if not TENSORBOARD_AVAILABLE:
            print("⚠️  使用简化模式解析日志")
            return self._parse_logs_simple()

        print("📊 正在解析TensorBoard日志...")

        # 查找事件文件
        event_files = glob.glob(os.path.join(LOG_DIR, "events.out.tfevents.*"))

        if not event_files:
            print("❌ 未找到TensorBoard事件文件")
            return False

        ea = event_accumulator.EventAccumulator()

        for event_file in event_files:
            ea.Reload()
            try:
                for event in ea.LoadFromPath(event_file):
                    if event.HasField('value'):
                        for value in event.value:
                            tag = value.tag
                            step = value.step
                            simple_value = value.simple_value

                            # 记录指标
                            self.metrics[tag].append((step, simple_value))
            except Exception as e:
                print(f"⚠️  解析 {event_file} 时出错: {e}")

        return True

    def _parse_logs_simple(self):
        """简化模式：直接解析训练输出日志"""
        print("📊 使用简化模式解析训练日志...")

        log_file = os.path.join(PROJECT_DIR, "training_output.log")

        if not os.path.exists(log_file):
            print("❌ 未找到训练输出日志")
            return False

        # 解析关键指标
        with open(log_file, 'r') as f:
            content = f.read()

        # 提取迭代信息
        iterations = re.findall(r'Iteration (\d+)', content)
        # 提取reach_goal率
        reach_goals = re.findall(r'reach_goal.*?(\d+\.?\d*%?).*?(\d+\.?\d*%?)', content)
        # 提取Policy Noise
        noises = re.findall(r'action noise std: ([\d.]+)', content)
        # 提取奖励
        rewards = re.findall(r'Mean reward: ([-\d.]+)', content)

        if iterations:
            self.final_metrics['max_iteration'] = int(iterations[-1])

        if reach_goals:
            self.final_metrics['final_reach_goal'] = reach_goals[-1]

        if noises:
            self.final_metrics['final_policy_noise'] = float(noises[-1])

        if rewards:
            self.final_metrics['final_reward'] = float(rewards[-1])

        return True

    def analyze_performance(self):
        """分析训练性能"""
        print("\n" + "="*60)
        print("📊 训练性能分析")
        print("="*60)

        # 分析reach_goal趋势
        if 'final_reach_goal' in self.final_metrics:
            reach = self.final_metrics['final_reach_goal']
            print(f"\n🎯 最终reach_goal率: {reach}")

            # 判断性能
            reach_value = float(reach.rstrip('%'))
            if reach_value >= 60:
                status = "✅ 优秀"
                suggestion = "策略已收敛，可以考虑部署测试"
            elif reach_value >= 40:
                status = "⚠️  良好"
                suggestion = "策略基本达到目标，建议继续训练或微调"
            elif reach_value >= 20:
                status = "🔶 一般"
                suggestion = "策略有进步，建议调整奖励权重或学习率"
            else:
                status = "❌ 较差"
                suggestion = "策略未收敛，建议检查奖励函数或训练参数"

            print(f"   评价: {status}")
            print(f"   建议: {suggestion}")

        # 分析Policy Noise
        if 'final_policy_noise' in self.final_metrics:
            noise = self.final_metrics['final_policy_noise']
            print(f"\n📈 最终Policy Noise: {noise}")

            if noise < 1.0:
                print(f"   评价: ✅ 稳定")
            elif noise < 5.0:
                print(f"   评价: ⚠️ 略高")
                print(f"   建议: 考虑降低学习率或增加action_smoothness权重")
            else:
                print(f"   评价: ❌ 不稳定")
                print(f"   建议: 策略可能崩溃，建议检查奖励函数")

        return True

    def generate_suggestions(self):
        """生成优化建议"""
        print("\n" + "="*60)
        print("💡 优化建议")
        print("="*60)

        suggestions = []

        # 基于final_reach_goal生成建议
        if 'final_reach_goal' in self.final_metrics:
            reach = float(self.final_metrics['final_reach_goal'].rstrip('%'))

            if reach < 20:
                suggestions.append({
                    'type': 'reward',
                    'issue': 'reach_goal率过低',
                    'action': '增加reach_goal奖励权重',
                    'details': '从2000.0提升到3000.0或更高'
                })
                suggestions.append({
                    'type': 'curriculum',
                    'issue': '可能需要更渐进的课程学习',
                    'action': '降低初始目标范围',
                    'details': '从3m降低到2m，让机器人更容易成功'
                })
            elif reach > 60:
                suggestions.append({
                    'type': 'success',
                    'issue': '策略已收敛',
                    'action': '部署测试',
                    'details': '可以导出ONNX模型进行实物测试'
                })

        # 基于final_policy_noise生成建议
        if 'final_policy_noise' in self.final_metrics:
            noise = self.final_metrics['final_policy_noise']

            if noise > 5.0:
                suggestions.append({
                    'type': 'stability',
                    'issue': 'Policy Noise过高',
                    'action': '增强平滑约束',
                    'details': 'action_smoothness从-0.01提升到-0.02'
                })
                suggestions.append({
                    'type': 'learning_rate',
                    'issue': '学习率可能过高',
                    'action': '降低学习率',
                    'details': 'learning_rate从1.5e-4降低到1e-4'
                })

        # 显示建议
        if suggestions:
            for i, sugg in enumerate(suggestions, 1):
                icon = "🔧" if sugg['type'] != 'success' else "✅"
                print(f"\n{icon} 建议 {i}: {sugg['issue']}")
                print(f"   类型: {sugg['type']}")
                print(f"   行动: {sugg['action']}")
                print(f"   详情: {sugg['details']}")
        else:
            print("\n✅ 当前策略表现良好，无需调整")

        return suggestions

    def generate_report(self):
        """生成训练报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(ISSUE_DIR, f"training_report_{timestamp}.md")

        with open(report_file, 'w') as f:
            f.write(f"# DashGo 训练报告\n\n")
            f.write(f"> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"> **训练日志**: {LOG_DIR}\n\n")

            f.write("## 📊 最终指标\n\n")
            f.write("| 指标 | 数值 |\n")
            f.write("|------|------|\n")

            for key, value in self.final_metrics.items():
                f.write(f"| {key} | {value} |\n")

            f.write("\n## 💡 优化建议\n\n")

            suggestions = self.generate_suggestions()

            for i, sugg in enumerate(suggestions, 1):
                f.write(f"\n### 建议 {i}: {sugg['issue']}\n")
                f.write(f"- **类型**: {sugg['type']}\n")
                f.write(f"- **行动**: {sugg['action']}\n")
                f.write(f"- **详情**: {sugg['details']}\n")

        print(f"\n📄 训练报告已保存: {report_file}")
        return report_file


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="DashGo自动训练分析")
    parser.add_argument("mode", nargs="?", default="interactive",
                       choices=["auto", "interactive"],
                       help="运行模式")
    args = parser.parse_args()

    print("🔍 DashGo 训练分析器")
    print("="*60)

    # 创建分析器
    analyzer = TrainingAnalyzer()

    # 解析日志
    if not analyzer.parse_tensorboard_logs():
        print("❌ 日志解析失败")
        return 1

    # 分析性能
    analyzer.analyze_performance()

    # 生成建议
    suggestions = analyzer.generate_suggestions()

    # 生成报告
    report_file = analyzer.generate_report()

    # 如果是auto模式，尝试自动应用建议
    if args.mode == "auto" and suggestions:
        print("\n" + "="*60)
        print("🤖 自动优化模式")
        print("="*60)

        # TODO: 实现自动修改参数的逻辑
        # 这需要修改train_cfg_v2.yaml或dashgo_env_v2.py
        print("⚠️  自动修改功能待实现（需要手动修改配置文件）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
