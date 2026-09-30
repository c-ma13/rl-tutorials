"""
Lesson 22: RL 工程实践
练习代码 - 练习版本

本练习演示 RL 工程实践，包括：
1. 使用 Stable-Baselines3 快速训练
2. Reward Shaping 技术
3. 超参数调优
4. 使用 Weights & Biases 记录实验
5. 向量化环境加速

注意：本练习需要安装额外的库：
pip install stable-baselines3 wandb gymnasium[classic_control]
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import time

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: 基础训练（Stable-Baselines3）
# ============================================================

def basic_training_sb3():
    """
    使用 Stable-Baselines3 进行基础训练

    演示：
    1. 创建环境
    2. 创建模型
    3. 训练
    4. 评估
    5. 保存/加载
    """
    print("=" * 60)
    print("Part 1: 基础训练（Stable-Baselines3）")
    print("=" * 60)

    # TODO 1: 使用 SB3 训练 PPO
    # 提示：
    # 1. 创建环境: gym.make("CartPole-v1")
    # 2. 创建模型: PPO("MlpPolicy", env, verbose=1)
    #    可选超参数: learning_rate=3e-4, n_steps=2048,
    #    batch_size=64, n_epochs=10, gamma=0.99
    # 3. 训练: model.learn(total_timesteps=20000)
    # 4. 评估: 运行 10 个 episode 计算平均奖励
    # 5. 保存: model.save("e:/rl_lessons/lessons_22/ppo_cartpole")
    # 6. 返回 model, mean_reward
    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----


# ============================================================
# Part 2: Reward Shaping
# ============================================================

class ShapedRewardWrapper(gym.Wrapper):
    """
    Reward Shaping 包装器

    为 CartPole 添加 shaped reward：
    - 原始奖励：每步 +1
    - Shaped 奖励：基于杆子角度和位置
    """

    def __init__(self, env, shaping_coef=0.1):
        super().__init__(env)
        self.shaping_coef = shaping_coef

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # TODO 2: 实现 Reward Shaping
        # 提示：
        # 1. 提取状态: position = obs[0], angle = obs[2]
        # 2. 计算 shaped reward:
        #    - 惩罚大角度: angle_penalty = -abs(angle)
        #    - 惩罚远离中心: position_penalty = -abs(position)
        # 3. 组合: shaped_reward = reward + shaping_coef * (angle_penalty + position_penalty)
        # 4. 返回 obs, shaped_reward, terminated, truncated, info
        # ---- 在这里实现 ----
        pass
        return obs, reward, terminated, truncated, info
        # ---- 代码结束 ----


def compare_reward_shaping():
    """比较有无 Reward Shaping 的训练效果"""
    print("\n" + "=" * 60)
    print("Part 2: Reward Shaping 对比")
    print("=" * 60)

    # 无 Shaping
    print("\n训练无 Reward Shaping 的模型...")
    env_no_shaping = gym.make("CartPole-v1")
    model_no_shaping = PPO("MlpPolicy", env_no_shaping, verbose=0)
    model_no_shaping.learn(total_timesteps=10000)

    # 有 Shaping
    print("训练有 Reward Shaping 的模型...")
    env_with_shaping = ShapedRewardWrapper(gym.make("CartPole-v1"), shaping_coef=0.1)
    model_with_shaping = PPO("MlpPolicy", env_with_shaping, verbose=0)
    model_with_shaping.learn(total_timesteps=10000)

    # 评估（在原始环境上）
    def evaluate(model, n_episodes=10):
        eval_env = gym.make("CartPole-v1")
        rewards = []
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            rewards.append(episode_reward)
        return np.mean(rewards)

    reward_no_shaping = evaluate(model_no_shaping)
    reward_with_shaping = evaluate(model_with_shaping)

    print(f"\n无 Shaping 平均奖励: {reward_no_shaping:.2f}")
    print(f"有 Shaping 平均奖励: {reward_with_shaping:.2f}")
    print(f"提升: {reward_with_shaping - reward_no_shaping:.2f}")

    return reward_no_shaping, reward_with_shaping


# ============================================================
# Part 3: 超参数调优
# ============================================================

def hyperparameter_tuning():
    """
    超参数调优示例

    比较不同学习率的效果
    """
    print("\n" + "=" * 60)
    print("Part 3: 超参数调优")
    print("=" * 60)

    # TODO 3: 实现超参数调优
    # 提示：
    # 1. 定义学习率列表: [1e-4, 3e-4, 1e-3]
    # 2. 对每个学习率:
    #    a) 创建环境和模型: PPO("MlpPolicy", env, learning_rate=lr, verbose=0)
    #    b) 训练: model.learn(total_timesteps=10000)
    #    c) 评估: 运行 10 个 episode，记录平均奖励
    # 3. 记录结果到 dict: results[lr] = mean_reward
    # 4. 打印最佳学习率
    # 5. 返回 results
    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----


# ============================================================
# Part 4: 向量化环境加速
# ============================================================

def vectorized_env_speedup():
    """
    演示向量化环境的加速效果

    比较单环境 vs 多环境并行训练
    """
    print("\n" + "=" * 60)
    print("Part 4: 向量化环境加速")
    print("=" * 60)

    # TODO 4: 实现向量化环境训练
    # 提示：
    # 1. 单环境训练:
    #    env = gym.make("CartPole-v1")
    #    model = PPO("MlpPolicy", env, verbose=0)
    #    用 time.time() 记录训练时间
    # 2. 向量化环境训练（4 个并行）:
    #    env = DummyVecEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])
    #    model = PPO("MlpPolicy", env, verbose=0)
    #    记录训练时间
    # 3. 计算加速比: speedup = time_single / time_vec
    # 4. 返回 time_single, time_vec, speedup
    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----


# ============================================================
# Part 5: W&B 集成（可选）
# ============================================================

class WandbCallback(BaseCallback):
    """
    Weights & Biases 回调

    记录训练指标到 W&B
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        # 检查是否有新的 episode 完成
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer) > len(self.episode_rewards):
            # 获取最新的 episode 信息
            info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(info['r'])
            self.episode_lengths.append(info['l'])

            # 如果安装了 wandb，记录到 W&B
            try:
                import wandb
                wandb.log({
                    "episode_reward": info['r'],
                    "episode_length": info['l'],
                    "timesteps": self.num_timesteps,
                })
            except ImportError:
                pass  # wandb 未安装

        return True


def train_with_wandb():
    """
    使用 W&B 记录训练过程

    注意：需要先安装 wandb 并登录
    """
    print("\n" + "=" * 60)
    print("Part 5: W&B 集成（可选）")
    print("=" * 60)

    try:
        import wandb

        # 初始化 W&B
        wandb.init(
            project="rl-engineering-practice",
            config={
                "algorithm": "PPO",
                "env": "CartPole-v1",
                "learning_rate": 3e-4,
                "total_timesteps": 20000,
            }
        )

        # 创建环境和模型
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=1)

        # 训练（使用 W&B 回调）
        model.learn(total_timesteps=20000, callback=WandbCallback())

        # 结束 W&B 运行
        wandb.finish()

        print("训练完成！请访问 W&B 查看结果。")

    except ImportError:
        print("未安装 wandb，跳过此部分。")
        print("安装方法: pip install wandb")


# ============================================================
# Part 6: 可视化
# ============================================================

def plot_results(results_dict):
    """绘制实验结果对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reward Shaping 对比
    ax = axes[0]
    methods = ['无 Shaping', '有 Shaping']
    rewards = [results_dict['no_shaping'], results_dict['with_shaping']]
    colors = ['#e74c3c', '#2ecc71']

    bars = ax.bar(methods, rewards, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('平均奖励')
    ax.set_title('Reward Shaping 效果对比')
    ax.set_ylim([0, max(rewards) * 1.2])

    # 在柱子上标注数值
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.1f}',
                ha='center', va='bottom', fontsize=12)

    ax.grid(True, alpha=0.3, axis='y')

    # 超参数调优结果
    ax = axes[1]
    if results_dict.get('hp_tuning'):
        lrs = list(results_dict['hp_tuning'].keys())
        lr_labels = [f'{lr:.0e}' for lr in lrs]
        rewards = list(results_dict['hp_tuning'].values())

        bars = ax.bar(lr_labels, rewards, color='skyblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('学习率')
        ax.set_ylabel('平均奖励')
        ax.set_title('学习率对比')
        ax.set_ylim([0, max(rewards) * 1.2])

        # 标注最佳
        best_idx = np.argmax(rewards)
        bars[best_idx].set_color('#2ecc71')

        for bar, reward in zip(bars, rewards):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{reward:.1f}',
                    ha='center', va='bottom', fontsize=12)

        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_22/rl_engineering_results.png',
                dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("RL 工程实践演示")
    print("=" * 60)

    results = {}

    # Part 1: 基础训练
    model, base_reward = basic_training_sb3()
    results['base'] = base_reward

    # Part 2: Reward Shaping
    no_shaping, with_shaping = compare_reward_shaping()
    results['no_shaping'] = no_shaping
    results['with_shaping'] = with_shaping

    # Part 3: 超参数调优
    hp_results = hyperparameter_tuning()
    results['hp_tuning'] = hp_results

    # Part 4: 向量化环境
    time_single, time_vec, speedup = vectorized_env_speedup()

    # Part 5: W&B（可选）
    # train_with_wandb()  # 取消注释以启用

    # 可视化结果
    print("\n" + "=" * 60)
    print("生成结果可视化...")
    print("=" * 60)
    plot_results(results)

    # 总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print(f"1. 基础 PPO 训练: {results['base']:.2f}")
    print(f"2. Reward Shaping 提升: {results['with_shaping'] - results['no_shaping']:.2f}")
    if results.get('hp_tuning'):
        print(f"3. 最佳学习率: {max(results['hp_tuning'], key=results['hp_tuning'].get):.0e}")
    print(f"4. 向量化加速比: {speedup:.2f}x")

    print("\nRL 工程实践演示完成！")
    print("\n关键要点：")
    print("1. 使用成熟的库（SB3）可以快速开始")
    print("2. Reward Shaping 可以加速学习")
    print("3. 超参数调优很重要")
    print("4. 向量化环境可以显著加速训练")
    print("5. 实验管理（W&B）帮助追踪和比较实验")
