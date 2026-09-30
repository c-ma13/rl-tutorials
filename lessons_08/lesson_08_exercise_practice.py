"""
Lesson 08 练习：Q-Learning — Off-Policy TD 控制

目标：
1. 实现 Q-Learning 算法
2. 实现 Double Q-Learning 算法
3. SARSA vs Q-Learning 对比实验
4. 策略可视化和路径分析
5. 最大化偏差验证实验
6. Expected SARSA（可选）

环境：CliffWalking-v0 (4×12 网格)
用于展示 On-Policy vs Off-Policy 的经典差异
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, List, Dict
import sys
import os

# 复用 Lesson 07 的辅助函数
sys.path.append(os.path.join(os.path.dirname(__file__), '../../lessons_07/exercises'))

# ============================================================
# Part 1: Q-Learning 算法
# ============================================================

def q_learning(
    env,
    num_episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon_start: float = 0.1,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995
) -> Tuple[np.ndarray, List[float]]:
    """
    Q-Learning 算法。

    TODO 1: 实现 Q-Learning
    算法:
        对每个 episode：
            S ← 起始
            重复:
                A ← ε-greedy(Q, S)        # 行为策略
                执行 A, 观察 R, S'
                Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
                                            ‾‾‾‾‾‾‾‾‾‾‾
                                            目标策略：贪心
                S ← S'

    关键区别：用 max_a Q(S',a) 而不是 Q(S',A')

    参数:
        env, num_episodes, alpha, gamma, epsilon参数

    返回:
        Q: shape (n_states, n_actions)
        reward_history: 每个 episode 的总奖励
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    reward_history = []
    epsilon = epsilon_start

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return Q, reward_history


# ============================================================
# Part 2: Double Q-Learning 算法
# ============================================================

def double_q_learning(
    env,
    num_episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon_start: float = 0.1,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Double Q-Learning 算法。

    TODO 2: 实现 Double Q-Learning
    算法:
        维护 Q_1 和 Q_2
        对每个 episode：
            S ← 起始
            重复:
                A ← ε-greedy(Q_1 + Q_2, S)
                执行 A, 观察 R, S'

                以 0.5 概率：
                    A* = argmax_a Q_1(S', a)
                    Q_1(S,A) ← Q_1(S,A) + α[R + γ Q_2(S', A*) - Q_1(S,A)]
                否则：
                    A* = argmax_a Q_2(S', a)
                    Q_2(S,A) ← Q_2(S,A) + α[R + γ Q_1(S', A*) - Q_2(S,A)]

    提示:
    - 用 Q_1 + Q_2 选动作
    - 随机选择更新哪个 Q
    - 用另一个 Q 评估

    返回:
        Q_1, Q_2, reward_history
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q_1 = np.zeros((n_states, n_actions))
    Q_2 = np.zeros((n_states, n_actions))
    reward_history = []
    epsilon = epsilon_start

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return Q_1, Q_2, reward_history


# ============================================================
# Part 3: Expected SARSA（可选）
# ============================================================

def expected_sarsa(
    env,
    num_episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon_start: float = 0.1,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995
) -> Tuple[np.ndarray, List[float]]:
    """
    Expected SARSA 算法。

    TODO 3: 实现 Expected SARSA
    更新:
        Q(S,A) ← Q(S,A) + α[R + γ Σ_a π(a|S') Q(S',a) - Q(S,A)]

    其中 π(a|S') 是 ε-greedy 策略的概率分布:
        π(a|S') = ε/|A| + (1-ε) · I(a = argmax Q(S',a))

    提示:
    - 计算所有动作的期望 Q 值
    - ε-greedy 下：最优动作概率 (1-ε) + ε/n_actions
    - 其他动作概率 ε/n_actions

    返回:
        Q, reward_history
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    reward_history = []
    epsilon = epsilon_start

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return Q, reward_history


# ============================================================
# Part 4: SARSA vs Q-Learning 对比
# ============================================================

def compare_sarsa_qlearning(
    env,
    num_episodes: int = 500,
    num_runs: int = 20,
    alpha: float = 0.5,
    gamma: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    对比 SARSA 和 Q-Learning 的训练曲线。

    TODO 4: 实现对比实验
    提示:
    - 需要导入 Lesson 07 的 sarsa 函数
    - 多次运行取平均
    - 记录训练期间的奖励曲线

    返回:
        results: dict, key=算法名, value=平均奖励曲线
    """
    # 尝试导入 SARSA
    try:
        from lesson_07_exercise import sarsa
    except ImportError:
        print("警告: 无法导入 SARSA，将跳过对比")
        return {}

    results = {
        'SARSA': np.zeros(num_episodes),
        'Q-Learning': np.zeros(num_episodes),
    }

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return results


# ============================================================
# Part 5: 路径分析
# ============================================================

def analyze_paths(Q: np.ndarray, env, n_eval: int = 1000) -> Dict[str, float]:
    """
    分析学到的策略的路径特征。

    TODO 5: 实现路径分析
    统计:
    - 平均步数
    - 掉悬崖次数
    - 经过安全区域（行2）的比例
    - 平均奖励

    返回:
        stats: dict, 包含各项统计
    """
    stats = {
        'avg_steps': 0.0,
        'cliff_falls': 0,
        'safe_path_ratio': 0.0,
        'avg_reward': 0.0,
    }

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return stats


# ============================================================
# Part 6: 最大化偏差实验
# ============================================================

def maximization_bias_experiment(
    num_episodes: int = 300,
    num_runs: int = 50
) -> Dict[str, List[float]]:
    """
    验证最大化偏差和 Double Q-Learning 的效果。

    TODO 6: 实现最大化偏差实验
    使用一个简单的环境来展示偏差:
    - 状态 A: 可以选择 left 或 right
    - left → 终止, reward = 0
    - right → 状态 B
    - 状态 B: 10 个动作，真实 Q 值都是 0，但有噪声
    - Q-Learning 会高估 B 的价值
    - Double Q-Learning 不会

    返回:
        results: dict, key=算法名, value=Q(A, right) 的学习曲线
    """
    # 简化实现：使用 CliffWalking 的某个状态
    # 这里我们用一个自定义的简单环境来演示

    class SimpleMDP:
        """简单的 MDP 用于展示最大化偏差"""
        def __init__(self):
            self.state = 0

        def reset(self):
            self.state = 0
            return self.state, {}

        def step(self, action):
            if self.state == 0:
                if action == 0:  # left
                    return 1, 0.0, True, False, {}
                else:  # right
                    self.state = 1
                    return 1, 0.0, False, False, {}
            else:  # state 1
                # 10 个动作，真实期望都是 -0.1，但有噪声
                reward = -0.1 + np.random.randn() * 0.5
                return 2, reward, True, False, {}

    results = {
        'Q-Learning': [],
        'Double Q-Learning': [],
    }

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return results


# ============================================================
# 辅助函数
# ============================================================

def print_policy(policy: np.ndarray, shape: Tuple[int, int] = (4, 12)):
    """打印策略"""
    action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    rows, cols = shape
    cliff_states = set(range(37, 47))

    print("\n策略:")
    for row in range(rows):
        line = ""
        for col in range(cols):
            s = row * cols + col
            if s == 36:
                line += " S "
            elif s == 47:
                line += " G "
            elif s in cliff_states:
                line += " C "
            else:
                line += f" {action_symbols[policy[s]]} "
        print(line)


def extract_policy(Q: np.ndarray) -> np.ndarray:
    """从 Q 提取贪心策略"""
    return np.argmax(Q, axis=1)


def moving_average(data: List[float], window: int = 50) -> List[float]:
    """移动平均"""
    result = []
    cumsum = np.cumsum(data)
    for i in range(len(data)):
        if i < window:
            result.append(cumsum[i] / (i + 1))
        else:
            result.append((cumsum[i] - cumsum[i - window]) / window)
    return result


# ============================================================
# 主测试
# ============================================================

if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')

    print("CliffWalking 环境:")
    print(f"  状态数: {env.observation_space.n}")
    print(f"  动作数: {env.action_space.n}")

    # --- 测试 1: Q-Learning ---
    print("\n" + "=" * 60)
    print("测试 1: Q-Learning 训练")
    print("=" * 60)

    Q_q, rewards_q = q_learning(
        env, num_episodes=500, alpha=0.5, gamma=1.0,
        epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=0.995
    )

    policy_q = extract_policy(Q_q)
    print_policy(policy_q)

    avg_q = moving_average(rewards_q, window=50)
    print(f"\n训练曲线（移动平均）:")
    for ep in [0, 100, 200, 300, 400, 499]:
        print(f"  Episode {ep:>4d}: {avg_q[ep]:.1f}")

    # --- 测试 2: Double Q-Learning ---
    print("\n" + "=" * 60)
    print("测试 2: Double Q-Learning 训练")
    print("=" * 60)

    Q1_dq, Q2_dq, rewards_dq = double_q_learning(
        env, num_episodes=500, alpha=0.5, gamma=1.0,
        epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=0.995
    )

    Q_dq = (Q1_dq + Q2_dq) / 2
    policy_dq = extract_policy(Q_dq)
    print_policy(policy_dq)

    print(f"\nQ-Learning vs Double Q-Learning 最后 50 episode:")
    print(f"  Q-Learning:        {np.mean(rewards_q[-50:]):.1f}")
    print(f"  Double Q-Learning: {np.mean(rewards_dq[-50:]):.1f}")

    # --- 测试 3: SARSA vs Q-Learning 对比 ---
    print("\n" + "=" * 60)
    print("测试 3: SARSA vs Q-Learning 对比")
    print("=" * 60)

    print("正在运行对比实验（20 次取平均）...")
    comparison = compare_sarsa_qlearning(env, num_episodes=500, num_runs=20)

    if comparison:
        print(f"\n{'算法':<20} {'训练期平均':<15} {'最后 50 ep':<15}")
        print("-" * 50)
        for name, rewards in comparison.items():
            train_avg = np.mean(rewards)
            final_avg = np.mean(rewards[-50:])
            print(f"{name:<20} {train_avg:<15.1f} {final_avg:<15.1f}")

    # --- 测试 4: 路径分析 ---
    print("\n" + "=" * 60)
    print("测试 4: 路径分析（贪心策略评估）")
    print("=" * 60)

    print("\nQ-Learning 路径特征:")
    stats_q = analyze_paths(Q_q, env, n_eval=1000)
    print(f"  平均步数: {stats_q['avg_steps']:.1f}")
    print(f"  平均奖励: {stats_q['avg_reward']:.1f}")
    print(f"  掉悬崖次数: {stats_q['cliff_falls']}")
    print(f"  经过安全区域: {stats_q['safe_path_ratio']*100:.1f}%")

    if 'SARSA' in comparison:
        try:
            from lesson_07_exercise import sarsa
            Q_sarsa, _ = sarsa(env, num_episodes=500, alpha=0.5, gamma=1.0)
            print("\nSARSA 路径特征:")
            stats_sarsa = analyze_paths(Q_sarsa, env, n_eval=1000)
            print(f"  平均步数: {stats_sarsa['avg_steps']:.1f}")
            print(f"  平均奖励: {stats_sarsa['avg_reward']:.1f}")
            print(f"  掉悬崖次数: {stats_sarsa['cliff_falls']}")
            print(f"  经过安全区域: {stats_sarsa['safe_path_ratio']*100:.1f}%")
        except:
            pass

    # --- 测试 5: Expected SARSA ---
    print("\n" + "=" * 60)
    print("测试 5: Expected SARSA")
    print("=" * 60)

    Q_exp, rewards_exp = expected_sarsa(
        env, num_episodes=500, alpha=0.5, gamma=1.0,
        epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=0.995
    )

    print(f"Expected SARSA 最后 50 episode 平均: {np.mean(rewards_exp[-50:]):.1f}")

    # --- 测试 6: 最大化偏差 ---
    print("\n" + "=" * 60)
    print("测试 6: 最大化偏差实验")
    print("=" * 60)

    print("正在运行最大化偏差实验...")
    bias_results = maximization_bias_experiment(num_episodes=300, num_runs=50)

    print(f"\n{'算法':<25} {'初期 Q(A,right)':<20} {'最终 Q(A,right)':<20}")
    print("-" * 65)
    for name, values in bias_results.items():
        early = np.mean(values[:50])
        final = np.mean(values[-50:])
        print(f"{name:<25} {early:<20.3f} {final:<20.3f}")

    print("\n说明:")
    print("- 真实最优值应该接近 0（left 和 right 期望相同）")
    print("- Q-Learning 因最大化偏差会高估 right 的价值")
    print("- Double Q-Learning 能减少这种偏差")

    print("\n" + "=" * 60)
    print("练习完成！")
    print("=" * 60)
    print("\n关键收获:")
    print("1. Q-Learning 是 Off-Policy：用 ε-greedy 收集数据，学贪心策略")
    print("2. 更新用 max_a Q(S',a) 而非 Q(S',A')，学到的是 Q*")
    print("3. CliffWalking 上：Q-Learning 学最短路径，SARSA 学安全路径")
    print("4. 最大化偏差：max 操作导致高估，Double Q-Learning 解决")
    print("5. Expected SARSA 介于 SARSA 和 Q-Learning 之间，方差更低")
    print("6. Off-Policy 的优势：数据效率高，可重用旧数据")

    env.close()
