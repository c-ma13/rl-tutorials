"""
Lesson 09 练习：函数逼近基础 (Function Approximation)

目标：
1. 实现多项式特征和 RBF 特征
2. 实现线性 SARSA
3. 实现线性 Q-Learning
4. 在 Mountain Car 上训练
5. 对比不同特征的效果
6. 可视化价值函数

环境：MountainCar-v0
状态：(position, velocity)
    position ∈ [-1.2, 0.6]
    velocity ∈ [-0.07, 0.07]
动作：0=左推, 1=不动, 2=右推
目标：到达右侧山顶（position ≥ 0.5）
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, List, Callable
import matplotlib.pyplot as plt
from itertools import product

# ============================================================
# Part 1: 特征提取器
# ============================================================

class PolynomialFeatures:
    """
    多项式特征。

    TODO 1: 实现多项式特征
    对于状态 s = (x, y)，生成特征:
        [1, x, y, x², y², xy, x³, y³, x²y, xy², ...]

    提示:
    - degree=2: [1, x, y, x², y², xy]
    - 需要归一化状态到 [0, 1]
    """

    def __init__(self, state_low: np.ndarray, state_high: np.ndarray, degree: int = 2):
        """
        参数:
            state_low: 状态下界
            state_high: 状态上界
            degree: 多项式阶数
        """
        self.state_low = state_low
        self.state_high = state_high
        self.degree = degree
        self.n_features = self._compute_n_features()

    def _compute_n_features(self) -> int:
        """计算特征数量"""
        # ---- 在这里实现 ----
        # 对于 d 维状态和 degree 阶多项式
        # 特征数 = C(d + degree, degree)
        d = len(self.state_low)
        # 简化：对于 2D 状态，degree=2 → 6 个特征
        if d == 2:
            return (self.degree + 1) * (self.degree + 2) // 2
        return (self.degree + 1) ** d
        # ---- 代码结束 ----

    def normalize(self, state: np.ndarray) -> np.ndarray:
        """归一化状态到 [0, 1]"""
        # ---- 在这里实现 ----
        return (state - self.state_low) / (self.state_high - self.state_low)
        # ---- 代码结束 ----

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """提取特征"""
        # ---- 在这里实现 ----
        s = self.normalize(state)
        features = []

        # 生成所有阶数 <= degree 的单项式
        for d in range(self.degree + 1):
            for powers in product(range(d + 1), repeat=len(s)):
                if sum(powers) == d:
                    feature = np.prod([s[i] ** p for i, p in enumerate(powers)])
                    features.append(feature)

        return np.array(features)
        # ---- 代码结束 ----


class RBFFeatures:
    """
    径向基函数 (Radial Basis Function) 特征。

    TODO 2: 实现 RBF 特征
    φ_i(s) = exp(-||s - c_i||² / (2σ²))

    提示:
    - 在状态空间均匀放置中心点
    - σ 控制宽度（影响泛化范围）
    """

    def __init__(
        self,
        state_low: np.ndarray,
        state_high: np.ndarray,
        n_centers_per_dim: int = 10,
        sigma: float = 0.1
    ):
        """
        参数:
            state_low, state_high: 状态边界
            n_centers_per_dim: 每个维度的中心点数量
            sigma: RBF 宽度
        """
        self.state_low = state_low
        self.state_high = state_high
        self.sigma = sigma
        self.centers = self._create_centers(n_centers_per_dim)
        self.n_features = len(self.centers)

    def _create_centers(self, n_per_dim: int) -> np.ndarray:
        """在状态空间均匀创建中心点"""
        # ---- 在这里实现 ----
        grids = []
        for low, high in zip(self.state_low, self.state_high):
            grids.append(np.linspace(low, high, n_per_dim))

        # 生成所有中心点的组合
        centers = np.array(list(product(*grids)))
        return centers
        # ---- 代码结束 ----

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """提取 RBF 特征"""
        # ---- 在这里实现 ----
        # φ_i(s) = exp(-||s - c_i||² / (2σ²))
        distances = np.linalg.norm(self.centers - state, axis=1)
        features = np.exp(-distances ** 2 / (2 * self.sigma ** 2))
        return features
        # ---- 代码结束 ----


# ============================================================
# Part 2: 线性函数逼近器
# ============================================================

class LinearApproximator:
    """
    线性函数逼近：q̂(s, a, w) = w^T φ(s, a)

    TODO 3: 实现线性函数逼近器
    """

    def __init__(self, feature_extractor: Callable, n_actions: int):
        """
        参数:
            feature_extractor: 特征提取函数 φ(s)
            n_actions: 动作数量
        """
        self.feature_extractor = feature_extractor
        self.n_actions = n_actions
        self.n_features = feature_extractor.n_features

        # 权重：每个动作一组权重
        self.w = np.zeros((n_actions, self.n_features))

    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        """获取 (s, a) 的特征"""
        # ---- 在这里实现 ----
        # 方法 1：直接用状态特征（动作通过索引区分）
        return self.feature_extractor(state)
        # ---- 代码结束 ----

    def predict(self, state: np.ndarray, action: int) -> float:
        """预测 q̂(s, a)"""
        # ---- 在这里实现 ----
        features = self.get_features(state, action)
        return np.dot(self.w[action], features)
        # ---- 代码结束 ----

    def predict_all(self, state: np.ndarray) -> np.ndarray:
        """预测所有动作的 Q 值"""
        # ---- 在这里实现 ----
        features = self.feature_extractor(state)
        return np.dot(self.w, features)  # shape: (n_actions,)
        # ---- 代码结束 ----

    def update(self, state: np.ndarray, action: int, td_error: float, alpha: float):
        """梯度更新：w ← w + α δ φ(s, a)"""
        # ---- 在这里实现 ----
        features = self.get_features(state, action)
        self.w[action] += alpha * td_error * features
        # ---- 代码结束 ----


# ============================================================
# Part 3: 线性 SARSA
# ============================================================

def linear_sarsa(
    env,
    feature_extractor: Callable,
    num_episodes: int = 500,
    alpha: float = 0.01,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995
) -> Tuple[LinearApproximator, List[float]]:
    """
    线性 SARSA 算法。

    TODO 4: 实现线性 SARSA
    算法:
        对每个 episode：
            S ← 起始, A ← ε-greedy
            重复:
                执行 A, 观察 R, S'
                A' ← ε-greedy
                δ ← R + γ q̂(S', A') - q̂(S, A)
                w ← w + α δ φ(S, A)
                S ← S', A ← A'

    参数:
        env, feature_extractor, num_episodes, alpha, gamma, epsilon参数

    返回:
        approximator: 训练好的线性逼近器
        reward_history: 每个 episode 的总奖励
    """
    n_actions = env.action_space.n
    approximator = LinearApproximator(feature_extractor, n_actions)
    reward_history = []
    epsilon = epsilon_start

    # ---- 在这里实现 ----
    for ep in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state)

        # ε-greedy 选动作
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = approximator.predict_all(state)
            action = np.argmax(q_values)

        total_reward = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state)
            done = terminated or truncated
            total_reward += reward

            # 选下一个动作
            if not done:
                if np.random.random() < epsilon:
                    next_action = env.action_space.sample()
                else:
                    q_values = approximator.predict_all(next_state)
                    next_action = np.argmax(q_values)

                # SARSA 更新
                q_current = approximator.predict(state, action)
                q_next = approximator.predict(next_state, next_action)
                td_error = reward + gamma * q_next - q_current
            else:
                # 终止状态
                q_current = approximator.predict(state, action)
                td_error = reward - q_current
                next_action = None

            approximator.update(state, action, td_error, alpha)

            state = next_state
            action = next_action

        reward_history.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    # ---- 代码结束 ----

    return approximator, reward_history


# ============================================================
# Part 4: 线性 Q-Learning
# ============================================================

def linear_q_learning(
    env,
    feature_extractor: Callable,
    num_episodes: int = 500,
    alpha: float = 0.01,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995
) -> Tuple[LinearApproximator, List[float]]:
    """
    线性 Q-Learning 算法。

    TODO 5: 实现线性 Q-Learning
    算法:
        对每个 episode：
            S ← 起始
            重复:
                A ← ε-greedy
                执行 A, 观察 R, S'
                δ ← R + γ max_a q̂(S', a) - q̂(S, A)
                w ← w + α δ φ(S, A)
                S ← S'

    注意: 可能不收敛（致命三元组）

    返回:
        approximator, reward_history
    """
    n_actions = env.action_space.n
    approximator = LinearApproximator(feature_extractor, n_actions)
    reward_history = []
    epsilon = epsilon_start

    # ---- 在这里实现 ----
    for ep in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state)
        total_reward = 0
        done = False

        while not done:
            # ε-greedy 选动作
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = approximator.predict_all(state)
                action = np.argmax(q_values)

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state)
            done = terminated or truncated
            total_reward += reward

            # Q-Learning 更新：用 max
            q_current = approximator.predict(state, action)
            if not done:
                q_next_max = np.max(approximator.predict_all(next_state))
                td_error = reward + gamma * q_next_max - q_current
            else:
                td_error = reward - q_current

            approximator.update(state, action, td_error, alpha)

            state = next_state

        reward_history.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    # ---- 代码结束 ----

    return approximator, reward_history


# ============================================================
# Part 5: 可视化
# ============================================================

def plot_value_function(approximator: LinearApproximator, env, title: str = ""):
    """
    可视化价值函数（2D 状态空间）。

    TODO 6: 实现价值函数可视化
    """
    # ---- 在这里实现 ----
    # 创建网格
    n_points = 50
    positions = np.linspace(-1.2, 0.6, n_points)
    velocities = np.linspace(-0.07, 0.07, n_points)

    # 计算每个网格点的最大 Q 值
    V = np.zeros((n_points, n_points))
    for i, p in enumerate(positions):
        for j, v in enumerate(velocities):
            state = np.array([p, v])
            q_values = approximator.predict_all(state)
            V[j, i] = np.max(q_values)  # 注意：j 对应 y 轴

    # 绘制热图
    plt.figure(figsize=(10, 6))
    plt.imshow(V, extent=[-1.2, 0.6, -0.07, 0.07], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='V(s)')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title(title)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Goal')
    plt.legend()
    plt.tight_layout()
    # ---- 代码结束 ----


def plot_learning_curves(results: dict, window: int = 10):
    """绘制学习曲线"""
    plt.figure(figsize=(12, 6))

    for name, rewards in results.items():
        # 移动平均
        smoothed = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(rewards[start:i+1]))

        plt.plot(smoothed, label=name, alpha=0.8)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Moving Avg)')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# ============================================================
# Part 6: 对比实验
# ============================================================

def compare_features(
    env,
    num_episodes: int = 500,
    num_runs: int = 5
) -> dict:
    """
    对比不同特征的效果。

    TODO 7: 实现特征对比实验
    """
    state_low = env.observation_space.low
    state_high = env.observation_space.high

    feature_configs = {
        'Polynomial (degree=2)': PolynomialFeatures(state_low, state_high, degree=2),
        'Polynomial (degree=3)': PolynomialFeatures(state_low, state_high, degree=3),
        'RBF (n=5, σ=0.2)': RBFFeatures(state_low, state_high, n_centers_per_dim=5, sigma=0.2),
        'RBF (n=10, σ=0.1)': RBFFeatures(state_low, state_high, n_centers_per_dim=10, sigma=0.1),
    }

    results = {}

    # ---- 在这里实现 ----
    for name, feature_extractor in feature_configs.items():
        print(f"训练 {name}...")
        all_rewards = np.zeros(num_episodes)

        for run in range(num_runs):
            np.random.seed(run * 42)
            _, rewards = linear_sarsa(
                env,
                feature_extractor,
                num_episodes=num_episodes,
                alpha=0.01,
                gamma=1.0
            )
            all_rewards += np.array(rewards)

        results[name] = all_rewards / num_runs
    # ---- 代码结束 ----

    return results


# ============================================================
# 主测试
# ============================================================

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')

    print("MountainCar-v0 环境:")
    print(f"  状态空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  目标: position ≥ 0.5")

    state_low = env.observation_space.low
    state_high = env.observation_space.high

    # --- 测试 1: 多项式特征 ---
    print("\n" + "=" * 60)
    print("测试 1: 多项式特征")
    print("=" * 60)

    poly_features = PolynomialFeatures(state_low, state_high, degree=2)
    test_state = np.array([0.0, 0.0])
    features = poly_features(test_state)
    print(f"特征数量: {poly_features.n_features}")
    print(f"测试状态 {test_state} 的特征: {features}")

    # --- 测试 2: RBF 特征 ---
    print("\n" + "=" * 60)
    print("测试 2: RBF 特征")
    print("=" * 60)

    rbf_features = RBFFeatures(state_low, state_high, n_centers_per_dim=5, sigma=0.2)
    features_rbf = rbf_features(test_state)
    print(f"特征数量: {rbf_features.n_features}")
    print(f"测试状态 {test_state} 的前 10 个 RBF 特征: {features_rbf[:10]}")

    # --- 测试 3: 线性 SARSA ---
    print("\n" + "=" * 60)
    print("测试 3: 线性 SARSA 训练")
    print("=" * 60)

    print("使用多项式特征（degree=2）训练...")
    approximator_sarsa, rewards_sarsa = linear_sarsa(
        env,
        poly_features,
        num_episodes=500,
        alpha=0.01,
        gamma=1.0
    )

    print(f"\n训练结果:")
    print(f"  前 50 episode 平均奖励: {np.mean(rewards_sarsa[:50]):.1f}")
    print(f"  后 50 episode 平均奖励: {np.mean(rewards_sarsa[-50:]):.1f}")
    print(f"  最佳 episode 奖励: {np.max(rewards_sarsa):.1f}")

    # --- 测试 4: 线性 Q-Learning ---
    print("\n" + "=" * 60)
    print("测试 4: 线性 Q-Learning 训练")
    print("=" * 60)

    print("使用多项式特征（degree=2）训练...")
    approximator_q, rewards_q = linear_q_learning(
        env,
        poly_features,
        num_episodes=500,
        alpha=0.01,
        gamma=1.0
    )

    print(f"\n训练结果:")
    print(f"  前 50 episode 平均奖励: {np.mean(rewards_q[:50]):.1f}")
    print(f"  后 50 episode 平均奖励: {np.mean(rewards_q[-50:]):.1f}")

    print(f"\nSARSA vs Q-Learning:")
    print(f"  SARSA 后 50 ep: {np.mean(rewards_sarsa[-50:]):.1f}")
    print(f"  Q-Learning 后 50 ep: {np.mean(rewards_q[-50:]):.1f}")

    # --- 测试 5: 特征对比 ---
    print("\n" + "=" * 60)
    print("测试 5: 不同特征对比")
    print("=" * 60)

    print("正在运行对比实验（5 次取平均）...")
    feature_results = compare_features(env, num_episodes=300, num_runs=5)

    print(f"\n{'特征类型':<30} {'前 50 ep':<12} {'后 50 ep':<12}")
    print("-" * 55)
    for name, rewards in feature_results.items():
        early = np.mean(rewards[:50])
        late = np.mean(rewards[-50:])
        print(f"{name:<30} {early:<12.1f} {late:<12.1f}")

    # --- 测试 6: 可视化 ---
    print("\n" + "=" * 60)
    print("测试 6: 可视化")
    print("=" * 60)

    print("绘制学习曲线...")
    plot_learning_curves({
        'SARSA (Poly)': rewards_sarsa,
        'Q-Learning (Poly)': rewards_q,
    })
    plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    print("  已保存: learning_curves.png")

    print("\n绘制价值函数...")
    plot_value_function(approximator_sarsa, env, "Value Function (Linear SARSA)")
    plt.savefig('value_function_sarsa.png', dpi=150, bbox_inches='tight')
    print("  已保存: value_function_sarsa.png")

    plot_value_function(approximator_q, env, "Value Function (Linear Q-Learning)")
    plt.savefig('value_function_q.png', dpi=150, bbox_inches='tight')
    print("  已保存: value_function_q.png")

    # --- 测试 7: 评估策略 ---
    print("\n" + "=" * 60)
    print("测试 7: 评估学到的策略")
    print("=" * 60)

    def evaluate_policy(approximator, env, n_eval=100):
        """评估策略"""
        total_rewards = []
        for _ in range(n_eval):
            state, _ = env.reset()
            state = np.array(state)
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < 200:
                q_values = approximator.predict_all(state)
                action = np.argmax(q_values)
                state, reward, terminated, truncated, _ = env.step(action)
                state = np.array(state)
                done = terminated or truncated
                total_reward += reward
                steps += 1

            total_rewards.append(total_reward)

        return np.mean(total_rewards), np.std(total_rewards)

    mean_sarsa, std_sarsa = evaluate_policy(approximator_sarsa, env, n_eval=100)
    mean_q, std_q = evaluate_policy(approximator_q, env, n_eval=100)

    print(f"SARSA 策略评估（100 局）:")
    print(f"  平均奖励: {mean_sarsa:.1f} ± {std_sarsa:.1f}")

    print(f"\nQ-Learning 策略评估（100 局）:")
    print(f"  平均奖励: {mean_q:.1f} ± {std_q:.1f}")

    print("\n" + "=" * 60)
    print("练习完成！")
    print("=" * 60)
    print("\n关键收获:")
    print("1. 函数逼近用参数化函数 v̂(s,w) 替代 Q 表，处理大规模/连续状态")
    print("2. 线性函数逼近：v̂(s,w) = w^T φ(s)，梯度更新 w ← w + α δ φ(s)")
    print("3. 特征工程至关重要：多项式、RBF 等不同特征影响性能")
    print("4. 线性 SARSA 收敛稳定（On-Policy），Q-Learning 可能不稳定")
    print("5. 致命三元组：函数逼近 + 自举 + Off-Policy → 可能发散")
    print("6. Mountain Car 需要泛化：相似状态应有相似价值")

    env.close()
    plt.show()
