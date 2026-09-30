"""
Lesson 12: 探索策略 (Exploration Strategies)
练习代码 - 参考实现

本练习实现多种探索策略，包括：
1. K-Armed Bandit 环境
2. ε-Greedy 策略（固定和衰减）
3. UCB (Upper Confidence Bound) 策略
4. Thompson Sampling 策略
5. Softmax/Boltzmann 策略
6. 乐观初始化 (Optimistic Initialization)
7. 对比实验：累积遗憾曲线、最优动作选择率
8. 非平稳环境测试
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# ============================================================
# Part 1: K-Armed Bandit 环境
# ============================================================

class KArmedBandit:
    """K-Armed Bandit 环境"""

    def __init__(self, k: int = 10, stationary: bool = True,
                 random_walk_std: float = 0.01):
        """
        初始化 K-Armed Bandit 环境

        参数:
            k: 臂的数量
            stationary: 是否为平稳环境
            random_walk_std: 非平稳环境下的随机游走标准差
        """
        self.k = k
        self.stationary = stationary
        self.random_walk_std = random_walk_std
        self.reset()

    def reset(self):
        """重置环境"""
        # TODO 1: 实现环境重置
        # 提示：
        # 1. 从标准正态分布初始化真实奖励均值 q_true
        # 2. 记录最优动作（q_true 最大的动作）
        # 3. 初始化时间步计数器
        # ---- 在这里实现 ----
        self.q_true = np.random.randn(self.k)
        self.best_action = np.argmax(self.q_true)
        self.t = 0
        # ---- 代码结束 ----

    def step(self, action: int) -> float:
        """
        执行动作并返回奖励

        参数:
            action: 选择的动作（0 到 k-1）

        返回:
            reward: 获得的奖励
        """
        # TODO 2: 实现动作执行
        # 提示：
        # 1. 从 N(q_true[action], 1) 采样奖励
        # 2. 如果是非平稳环境，更新 q_true（随机游走）
        # 3. 更新时间步
        # ---- 在这里实现 ----
        reward = np.random.randn() + self.q_true[action]

        if not self.stationary:
            # 非平稳环境：q_true 进行随机游走
            self.q_true += np.random.randn(self.k) * self.random_walk_std

        self.t += 1
        return reward
        # ---- 代码结束 ----


# ============================================================
# Part 2: 探索策略基类
# ============================================================

class ExplorationStrategy:
    """探索策略基类"""

    def __init__(self, k: int):
        """
        初始化策略

        参数:
            k: 动作数量
        """
        self.k = k
        self.reset()

    def reset(self):
        """重置策略状态"""
        self.Q = np.zeros(self.k)  # 动作价值估计
        self.N = np.zeros(self.k)  # 动作选择次数
        self.t = 0  # 时间步

    def select_action(self) -> int:
        """选择动作"""
        raise NotImplementedError

    def update(self, action: int, reward: float):
        """更新动作价值估计"""
        self.t += 1
        self.N[action] += 1
        # 增量式更新
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


# ============================================================
# Part 3: ε-Greedy 策略
# ============================================================

class EpsilonGreedy(ExplorationStrategy):
    """ε-Greedy 探索策略"""

    def __init__(self, k: int, epsilon: float = 0.1, decay: bool = False):
        """
        初始化 ε-Greedy 策略

        参数:
            k: 动作数量
            epsilon: 探索概率
            decay: 是否使用衰减的 epsilon
        """
        super().__init__(k)
        self.epsilon_init = epsilon
        self.epsilon = epsilon
        self.decay = decay

    def select_action(self) -> int:
        """选择动作"""
        # TODO 3: 实现 ε-Greedy 动作选择
        # 提示：
        # 1. 如果使用衰减，更新 epsilon = epsilon_init / (1 + t * 0.001)
        # 2. 以概率 epsilon 随机选择动作
        # 3. 以概率 1-epsilon 选择当前最优动作（打破平局）
        # ---- 在这里实现 ----
        if self.decay:
            self.epsilon = self.epsilon_init / (1 + self.t * 0.001)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            # 打破平局：在最大值中随机选择
            max_q = np.max(self.Q)
            best_actions = np.where(self.Q == max_q)[0]
            return np.random.choice(best_actions)
        # ---- 代码结束 ----


# ============================================================
# Part 4: UCB 策略
# ============================================================

class UCB(ExplorationStrategy):
    """UCB (Upper Confidence Bound) 探索策略"""

    def __init__(self, k: int, c: float = 2.0):
        """
        初始化 UCB 策略

        参数:
            k: 动作数量
            c: 探索参数
        """
        super().__init__(k)
        self.c = c

    def select_action(self) -> int:
        """选择动作"""
        # TODO 4: 实现 UCB 动作选择
        # 提示：
        # 1. 如果存在未选择过的动作（N[a] == 0），优先选择
        # 2. 否则计算 UCB 值：Q[a] + c * sqrt(ln(t) / N[a])
        # 3. 选择 UCB 值最大的动作
        # ---- 在这里实现 ----
        # 优先选择未尝试过的动作
        unvisited = np.where(self.N == 0)[0]
        if len(unvisited) > 0:
            return np.random.choice(unvisited)

        # 计算 UCB 值
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        return np.argmax(ucb_values)
        # ---- 代码结束 ----


# ============================================================
# Part 5: Thompson Sampling 策略
# ============================================================

class ThompsonSampling(ExplorationStrategy):
    """Thompson Sampling 探索策略（假设高斯奖励）"""

    def __init__(self, k: int, prior_mean: float = 0.0, prior_std: float = 1.0):
        """
        初始化 Thompson Sampling 策略

        参数:
            k: 动作数量
            prior_mean: 先验均值
            prior_std: 先验标准差
        """
        super().__init__(k)
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.sum_rewards = np.zeros(k)  # 累积奖励

    def select_action(self) -> int:
        """选择动作"""
        # TODO 5: 实现 Thompson Sampling 动作选择
        # 提示：
        # 1. 对每个动作，从后验分布采样一个值
        # 2. 后验均值 = sum_rewards[a] / (N[a] + 1)
        # 3. 后验标准差 = 1 / sqrt(N[a] + 1)
        # 4. 选择采样值最大的动作
        # ---- 在这里实现 ----
        samples = np.zeros(self.k)
        for a in range(self.k):
            if self.N[a] == 0:
                # 使用先验分布
                samples[a] = np.random.randn() * self.prior_std + self.prior_mean
            else:
                # 使用后验分布
                posterior_mean = self.sum_rewards[a] / self.N[a]
                posterior_std = 1.0 / np.sqrt(self.N[a])
                samples[a] = np.random.randn() * posterior_std + posterior_mean

        return np.argmax(samples)
        # ---- 代码结束 ----

    def update(self, action: int, reward: float):
        """更新动作价值估计"""
        self.t += 1
        self.N[action] += 1
        self.sum_rewards[action] += reward
        self.Q[action] = self.sum_rewards[action] / self.N[action]


# ============================================================
# Part 6: Softmax/Boltzmann 策略
# ============================================================

class Softmax(ExplorationStrategy):
    """Softmax/Boltzmann 探索策略"""

    def __init__(self, k: int, temperature: float = 1.0):
        """
        初始化 Softmax 策略

        参数:
            k: 动作数量
            temperature: 温度参数（控制探索程度）
        """
        super().__init__(k)
        self.temperature = temperature

    def select_action(self) -> int:
        """选择动作"""
        # TODO 6: 实现 Softmax 动作选择
        # 提示：
        # 1. 计算 exp(Q[a] / temperature)
        # 2. 归一化得到概率分布
        # 3. 根据概率分布采样动作
        # 4. 注意数值稳定性：减去最大值
        # ---- 在这里实现 ----
        # 数值稳定性：减去最大值
        q_shifted = self.Q - np.max(self.Q)
        exp_q = np.exp(q_shifted / self.temperature)
        probs = exp_q / np.sum(exp_q)

        return np.random.choice(self.k, p=probs)
        # ---- 代码结束 ----


# ============================================================
# Part 7: 乐观初始化策略
# ============================================================

class OptimisticInitialization(ExplorationStrategy):
    """乐观初始化 + Greedy 策略"""

    def __init__(self, k: int, initial_value: float = 5.0):
        """
        初始化乐观初始化策略

        参数:
            k: 动作数量
            initial_value: 初始 Q 值（乐观值）
        """
        super().__init__(k)
        self.Q = np.ones(k) * initial_value  # 乐观初始化

    def select_action(self) -> int:
        """选择动作（纯 Greedy）"""
        # TODO 7: 实现乐观初始化的动作选择
        # 提示：
        # 1. 直接选择当前 Q 值最大的动作
        # 2. 打破平局
        # ---- 在这里实现 ----
        max_q = np.max(self.Q)
        best_actions = np.where(self.Q == max_q)[0]
        return np.random.choice(best_actions)
        # ---- 代码结束 ----


# ============================================================
# Part 8: 实验运行函数
# ============================================================

def run_experiment(env: KArmedBandit, strategy: ExplorationStrategy,
                   num_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    运行单次实验

    参数:
        env: Bandit 环境
        strategy: 探索策略
        num_steps: 运行步数

    返回:
        rewards: 每步获得的奖励
        optimal_actions: 每步是否选择了最优动作
        regrets: 每步的遗憾值
    """
    # TODO 8: 实现实验运行
    # 提示：
    # 1. 重置环境和策略
    # 2. 循环 num_steps 步
    # 3. 每步：选择动作、执行、更新策略
    # 4. 记录奖励、是否最优动作、遗憾值
    # 5. 遗憾 = q_true[best_action] - q_true[action]
    # ---- 在这里实现 ----
    env.reset()
    strategy.reset()

    rewards = np.zeros(num_steps)
    optimal_actions = np.zeros(num_steps, dtype=bool)
    regrets = np.zeros(num_steps)

    for t in range(num_steps):
        action = strategy.select_action()
        reward = env.step(action)
        strategy.update(action, reward)

        rewards[t] = reward
        optimal_actions[t] = (action == env.best_action)
        regrets[t] = env.q_true[env.best_action] - env.q_true[action]

    return rewards, optimal_actions, regrets
    # ---- 代码结束 ----


def run_multiple_experiments(env_factory: Callable, strategy_factory: Callable,
                             num_runs: int = 100, num_steps: int = 1000
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    运行多次实验并平均

    参数:
        env_factory: 环境工厂函数
        strategy_factory: 策略工厂函数
        num_runs: 运行次数
        num_steps: 每次运行的步数

    返回:
        avg_rewards: 平均奖励
        avg_optimal: 平均最优动作选择率
        avg_regrets: 平均遗憾
    """
    all_rewards = np.zeros((num_runs, num_steps))
    all_optimal = np.zeros((num_runs, num_steps))
    all_regrets = np.zeros((num_runs, num_steps))

    for run in range(num_runs):
        env = env_factory()
        strategy = strategy_factory()
        rewards, optimal, regrets = run_experiment(env, strategy, num_steps)

        all_rewards[run] = rewards
        all_optimal[run] = optimal
        all_regrets[run] = regrets

    return np.mean(all_rewards, axis=0), np.mean(all_optimal, axis=0), np.mean(all_regrets, axis=0)


# ============================================================
# Part 9: 对比实验
# ============================================================

def compare_strategies():
    """对比不同探索策略"""
    print("=" * 60)
    print("对比不同探索策略")
    print("=" * 60)

    k = 10
    num_runs = 100
    num_steps = 1000

    # 定义策略
    strategies = {
        'ε-Greedy (ε=0.1)': lambda: EpsilonGreedy(k, epsilon=0.1),
        'ε-Greedy (ε=0.01)': lambda: EpsilonGreedy(k, epsilon=0.01),
        'ε-Greedy (衰减)': lambda: EpsilonGreedy(k, epsilon=0.1, decay=True),
        'UCB (c=2)': lambda: UCB(k, c=2.0),
        'Thompson Sampling': lambda: ThompsonSampling(k),
        'Softmax (τ=0.1)': lambda: Softmax(k, temperature=0.1),
        'Optimistic Init (Q=5)': lambda: OptimisticInitialization(k, initial_value=5.0),
    }

    results = {}
    for name, strategy_factory in strategies.items():
        print(f"\n运行 {name}...")
        env_factory = lambda: KArmedBandit(k=k, stationary=True)
        avg_rewards, avg_optimal, avg_regrets = run_multiple_experiments(
            env_factory, strategy_factory, num_runs, num_steps
        )
        results[name] = {
            'rewards': avg_rewards,
            'optimal': avg_optimal,
            'regrets': avg_regrets
        }

    # 绘制结果
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 平均奖励
    for name, data in results.items():
        axes[0].plot(data['rewards'], label=name, alpha=0.8)
    axes[0].set_xlabel('步数')
    axes[0].set_ylabel('平均奖励')
    axes[0].set_title('平均奖励对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 最优动作选择率
    for name, data in results.items():
        axes[1].plot(data['optimal'] * 100, label=name, alpha=0.8)
    axes[1].set_xlabel('步数')
    axes[1].set_ylabel('最优动作选择率 (%)')
    axes[1].set_title('最优动作选择率对比')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 累积遗憾
    for name, data in results.items():
        cumulative_regret = np.cumsum(data['regrets'])
        axes[2].plot(cumulative_regret, label=name, alpha=0.8)
    axes[2].set_xlabel('步数')
    axes[2].set_ylabel('累积遗憾')
    axes[2].set_title('累积遗憾对比')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_12/exploration_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 打印最终统计
    print("\n" + "=" * 60)
    print("最终统计（最后100步平均）")
    print("=" * 60)
    for name, data in results.items():
        final_reward = np.mean(data['rewards'][-100:])
        final_optimal = np.mean(data['optimal'][-100:]) * 100
        final_regret = np.sum(data['regrets'])
        print(f"{name:25s}: 奖励={final_reward:.3f}, 最优率={final_optimal:.1f}%, 累积遗憾={final_regret:.1f}")


# ============================================================
# Part 10: 非平稳环境测试
# ============================================================

def test_nonstationary():
    """测试非平稳环境下的策略表现"""
    print("\n" + "=" * 60)
    print("非平稳环境测试")
    print("=" * 60)

    k = 10
    num_runs = 100
    num_steps = 10000

    strategies = {
        'ε-Greedy (ε=0.1)': lambda: EpsilonGreedy(k, epsilon=0.1),
        'ε-Greedy (衰减)': lambda: EpsilonGreedy(k, epsilon=0.1, decay=True),
        'UCB (c=2)': lambda: UCB(k, c=2.0),
    }

    results = {}
    for name, strategy_factory in strategies.items():
        print(f"\n运行 {name}...")
        env_factory = lambda: KArmedBandit(k=k, stationary=False, random_walk_std=0.01)
        avg_rewards, avg_optimal, avg_regrets = run_multiple_experiments(
            env_factory, strategy_factory, num_runs, num_steps
        )
        results[name] = {
            'rewards': avg_rewards,
            'optimal': avg_optimal,
        }

    # 绘制结果
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 平均奖励
    for name, data in results.items():
        axes[0].plot(data['rewards'], label=name, alpha=0.8)
    axes[0].set_xlabel('步数')
    axes[0].set_ylabel('平均奖励')
    axes[0].set_title('非平稳环境：平均奖励')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 最优动作选择率
    for name, data in results.items():
        axes[1].plot(data['optimal'] * 100, label=name, alpha=0.8)
    axes[1].set_xlabel('步数')
    axes[1].set_ylabel('最优动作选择率 (%)')
    axes[1].set_title('非平稳环境：最优动作选择率')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_12/nonstationary_test.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)

    # 运行对比实验
    compare_strategies()

    # 测试非平稳环境
    test_nonstationary()

    print("\n所有实验完成！")

