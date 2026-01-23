#!/usr/bin/env python3
"""
配置参数验证脚本

开发基准: Isaac Sim 4.5 + Ubuntu 20.04
用途: 验证train_cfg_v2.yaml中的所有参数是否被实际支持
"""

import sys
import os

# 添加Isaac Lab路径
isaac_lab_path = os.path.expanduser("~/IsaacLab")
sys.path.insert(0, isaac_lab_path)
sys.path.insert(0, os.path.join(isaac_lab_path, "source"))

def verify_algorithm_config():
    """验证algorithm配置参数"""
    print("=" * 60)
    print("验证algorithm配置参数")
    print("=" * 60)

    try:
        from rsl_rl.algorithms import PPO
        import inspect

        # 获取PPO支持的参数
        sig = inspect.signature(PPO.__init__)
        supported_params = set(sig.parameters.keys())
        supported_params.discard('self')  # 移除self参数

        print(f"\n✓ PPO支持的参数 ({len(supported_params)}个):")
        for param in sorted(supported_params):
            print(f"  - {param}")

        # 读取配置文件
        import yaml
        cfg_path = "train_cfg_v2.yaml"
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)

        algorithm_cfg = config.get('algorithm', {})
        configured_params = set(algorithm_cfg.keys())

        # 移除特殊字段
        configured_params.discard('class_name')

        print(f"\n✓ 配置文件中的参数 ({len(configured_params)}个):")
        for param in sorted(configured_params):
            status = "✓" if param in supported_params else "✗ 不支持"
            print(f"  {status} {param}")

        # 检查不支持的参数
        unsupported = configured_params - supported_params
        if unsupported:
            print(f"\n✗ 发现不支持的参数 ({len(unsupported)}个):")
            for param in unsupported:
                print(f"  - {param}")
            return False
        else:
            print(f"\n✓ 所有algorithm参数都有效！")
            return True

    except Exception as e:
        print(f"\n✗ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_policy_config():
    """验证policy配置参数"""
    print("\n" + "=" * 60)
    print("验证policy配置参数")
    print("=" * 60)

    try:
        from rsl_rl.policies import ActorCritic
        import inspect

        # 获取ActorCritic支持的参数
        sig = inspect.signature(ActorCritic.__init__)
        supported_params = set(sig.parameters.keys())
        supported_params.discard('self')

        print(f"\n✓ ActorCritic支持的参数 ({len(supported_params)}个):")
        for param in sorted(supported_params):
            print(f"  - {param}")

        # 读取配置文件
        import yaml
        cfg_path = "train_cfg_v2.yaml"
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)

        policy_cfg = config.get('policy', {})
        configured_params = set(policy_cfg.keys())

        # 移除特殊字段
        configured_params.discard('class_name')

        print(f"\n✓ 配置文件中的参数 ({len(configured_params)}个):")
        for param in sorted(configured_params):
            status = "✓" if param in supported_params else "✗ 不支持"
            print(f"  {status} {param}")

        # 检查不支持的参数
        unsupported = configured_params - supported_params
        if unsupported:
            print(f"\n✗ 发现不支持的参数 ({len(unsupported)}个):")
            for param in unsupported:
                print(f"  - {param}")
            return False
        else:
            print(f"\n✓ 所有policy参数都有效！")
            return True

    except Exception as e:
        print(f"\n✗ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_environment_config():
    """验证环境配置"""
    print("\n" + "=" * 60)
    print("验证环境配置")
    print("=" * 60)

    try:
        from omegaconf import OmegaConf
        from dashgo_env_v2 import DashgoNavEnvV2Cfg

        # 创建配置
        env_cfg = DashgoNavEnvV2Cfg()
        print(f"\n✓ 环境配置创建成功")
        print(f"  - num_envs: {env_cfg.scene.num_envs}")
        print(f"  - episode_length_s: {env_cfg.episode_length_s}")

        return True

    except Exception as e:
        print(f"\n✗ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("DashGo配置参数验证")
    print("=" * 60)

    results = []

    # 验证algorithm配置
    results.append(("algorithm", verify_algorithm_config()))

    # 验证policy配置
    results.append(("policy", verify_policy_config()))

    # 验证environment配置
    results.append(("environment", verify_environment_config()))

    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)

    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status} - {name}")

    all_passed = all(success for _, success in results)

    if all_passed:
        print("\n✓ 所有配置验证通过！可以开始训练。")
        return 0
    else:
        print("\n✗ 部分配置验证失败，请修复后再训练。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
