#!/usr/bin/env python3
"""
Reach Goal 奖励诊断工具

用途：诊断为什么 Episode_Reward/reach_goal 始终为0
创建时间：2026-01-30 01:45:00
"""
import yaml
import sys
import os

def load_config(filepath="train_cfg_v2.yaml"):
    """加载YAML配置文件"""
    if not os.path.exists(filepath):
        print(f"❌ 配置文件不存在: {filepath}")
        return None

    with open(filepath) as f:
        try:
            cfg = yaml.safe_load(f)
            print(f"✅ 成功加载配置: {filepath}")
            return cfg
        except yaml.YAMLError as e:
            print(f"❌ YAML解析失败: {e}")
            return None

def extract_config(cfg, path):
    """从配置中提取指定路径的值"""
    keys = path.split('.')
    value = cfg

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None

    return value

def print_section(title, data):
    """打印配置区块"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    if isinstance(data, dict):
        for key, value in data.items():
            print(f"  {key}: {value}")
    else:
        print(f"  {data}")

def diagnose_thresholds(cfg):
    """诊断阈值配置"""
    print("\n[🔍 阈值诊断]")

    try:
        # 尝试从环境配置中获取
        terminations = extract_config(cfg, 'env terminations reach_goal')
        rewards = extract_config(cfg, 'env rewards reach_goal')

        if terminations and rewards:
            term_thresh = terminations.get('params', {}).get('threshold', 'N/A')
            reward_thresh = rewards.get('params', {}).get('threshold', 'N/A')

            print(f"  终止阈值: {term_thresh}")
            print(f"  奖励阈值: {reward_thresh}")

            # 比较
            if term_thresh == 'N/A' or reward_thresh == 'N/A':
                print(f"\n  ⚠️  无法比较（阈值未设置）")
                return False

            if term_thresh == reward_thresh:
                print(f"\n  ✅ 阈值一致: {term_thresh}")
                return True
            elif term_thresh > reward_thresh:
                print(f"\n  ❌ 终止阈值({term_thresh}) > 奖励阈值({reward_thresh})")
                print(f"     问题: 机器人触发终止了，但还没拿到奖励！")
                print(f"     修复: 将奖励阈值改为 {term_thresh}")
                return False
            else:
                print(f"\n  ⚠️  终止阈值({term_thresh}) < 奖励阈值({reward_thresh})")
                print(f"     理论上奖励应该先触发，但实际为0")
                print(f"     可能原因: 函数实现问题或计算顺序问题")
                return False
        else:
            print(f"  ❌ 无法找到 reach_goal 配置")
            return False

    except Exception as e:
        print(f"  ❌ 诊断失败: {e}")
        return False

def diagnose_functions(cfg):
    """诊断函数配置"""
    print("\n[🔍 函数诊断]")

    try:
        term_func = extract_config(cfg, 'env terminations reach_goal func')
        reward_func = extract_config(cfg, 'env rewards reach_goal func')

        print(f"  终止函数: {term_func}")
        print(f"  奖励函数: {reward_func}")

        # 检查是否使用相同的函数
        if term_func and reward_func:
            if 'terminal_reward' in reward_func:
                print(f"\n  ⚠️  使用 terminal_reward（可能绑定到reset）")
                print(f"     问题: 奖励可能在终止后计算")
                print(f"     修复: 改用自定义函数或 is_close_to_target")
                return False
            elif 'is_close_to_target' in reward_func:
                print(f"\n  ✅ 使用 is_close_to_target（独立函数）")
                return True
            else:
                print(f"\n  ℹ️  使用自定义函数: {reward_func}")
                return True

    except Exception as e:
        print(f"  ❌ 诊断失败: {e}")
        return False

def diagnose_weights(cfg):
    """诊断权重配置"""
    print("\n[🔍 权重诊断]")

    try:
        term_weight = extract_config(cfg, 'env terminations reach_goal weight')
        reward_weight = extract_config(cfg, 'env rewards reach_goal weight')

        print(f"  终止权重: {term_weight}")
        print(f"  奖励权重: {reward_weight}")

        if reward_weight == 0 or reward_weight is None:
            print(f"\n  ❌ 奖励权重为0或未设置！")
            return False

        if reward_weight > 0:
            print(f"\n  ✅ 奖励权重为正: {reward_weight}")
            return True
        else:
            print(f"\n  ⚠️  奖励权重为负: {reward_weight}")
            return False

    except Exception as e:
        print(f"  ❌ 诊断失败: {e}")
        return False

def main():
    print("="*60)
    print("Reach Goal 奖励诊断工具 v1.0")
    print("="*60)

    # 1. 加载配置
    cfg = load_config("train_cfg_v2.yaml")
    if not cfg:
        print("\n❌ 无法加载配置，退出")
        return 1

    # 2. 打印完整配置
    print("\n[📋 完整配置]")
    try:
        env_cfg = extract_config(cfg, 'env')
        if env_cfg:
            print_section("环境配置", env_cfg.get('terminations', {}))
            print_section("奖励配置", env_cfg.get('rewards', {}))
    except Exception as e:
        print(f"  ⚠️  无法打印完整配置: {e}")

    # 3. 诊断阈值
    result1 = diagnose_thresholds(cfg)

    # 4. 诊断函数
    result2 = diagnose_functions(cfg)

    # 5. 诊断权重
    result3 = diagnose_weights(cfg)

    # 6. 总结
    print("\n" + "="*60)
    print("[📊 诊断总结]")
    print("="*60)

    results = {
        "阈值配置": result1,
        "函数配置": result2,
        "权重配置": result3
    }

    all_pass = all(results.values())

    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")

    print("\n" + "="*60)

    if all_pass:
        print("✅ 所有诊断通过，配置看起来正常")
        print("   如果 reach_goal 仍为0，可能需要检查:")
        print("   1. 函数实现是否正确")
        print("   2. 计算顺序是否正确")
        print("   3. 是否有其他覆盖的配置")
    else:
        print("❌ 发现问题，请根据上述建议修复")
        print("\n推荐的修复方案:")
        if not result1:
            print("  1. 统一终止和奖励阈值为 0.5m")
        if not result2:
            print("  2. 改用自定义奖励函数（参考 issue 文档）")
        if not result3:
            print("  3. 设置奖励权重为 1.0 或更高")

    print("="*60)

    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
