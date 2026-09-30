"""
Lesson 10 练习：策略梯度与 REINFORCE

目标：
1. 实现线性策略网络（Softmax）
2. 实现 REINFORCE 算法
3. 实现基线（Baseline）
4. 在 CartPole 上训练
5. 对比有无基线的效果
6. 可视化学习曲线

环境：CartPole-v1
状态：(x, ẋ, θ, θ̇) - 4维连续
动作：0=左推, 1=右推
目标：保持杆子直立
成功：连续100个episode平均奖励≥195
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, List
import matplotlib.pyplot as plt

# ============================================================
# Part 1: 策略网络（线性 Softmax）
# ============================================================

class LinearPolicy:
    """
    线性策略：π(a|s; θ) = softmax(θ^T φ(s))

    TODO 1: 实现线性策略网络
    """

    def __init__(self, n_features: int, n_actions: int):
        """
        参数:
            n_features: 特征维度
            n_actions: 动作数量
        """
        self.n_features = n_features
        self.n_actions = n_actions
        # 初始化权重（小随机值）
        self.theta = np.random.randn(n_features, n_actions) * 0.01

    def get_features(self, state: np.ndarray) -> np.ndarray:
        """提取特征（这里直接用状态）"""
        # ---- 在这里实现 ----
        # 简单版本：直接用状态作为特征
        return state
        # ---- 代码结束 ----

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        前向传播：计算动作概率

        返回: probs, shape (n_actions,)
        """
        # ---- 在这里实现 ----
        features = self.get_features(state)
        logits = np.dot(features, self.theta)  # shape: (n_actions,)

        # Softmax（数值稳定版本）
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)

        return probs
        # ---- 代码结束 ----

    def sample_action(self, state: np.ndarray) -> int:
        """根据策略采样动作"""
        # ---- 在这里实现 ----
        probs = self.forward(state)
        action = np.random.choice(self.n_actions, p=probs)
        return action
        # ---- 代码结束 ----

    def log_prob(self, state: np.ndarray, action: int) -> float:
        """计算 log π(a|s)"""
        # ---- 在这里实现 ----
        probs = self.forward(state)
        return np.log(probs[action] + 1e-8)  # 加小常数避免 log(0)
        # ---- 代码结束 ----

    def grad_log_prob(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        计算 ∇_θ log π(a|s)

        对于 Softmax 策略：
        ∇_θ log π(a|s) = φ(s) (e_a - π(·|s))
        其中 e_a 是 one-hot 向量
        """
        # ---- 在这里实现 ----
        features = self.get_features(state)
        probs = self.forward(state)

        # ∇_θ log π(a|s) = φ(s) ⊗ (e_a - π)
        grad = np.zeros((self.n_features, self.n_actions))
        for a in range(self.n_actions):
            if a == action:
                grad[:, a] = features * (1 - probs[a])
            else:
                grad[:, a] = -features * probs[a]

        return grad
        # ---- 代码结束 ----


# ============================================================
# Part 2: 基线网络（价值函数）
# ============================================================

class ValueBaseline:
    """
    线性价值函数：V(s; w) = w^T φ(s)

    TODO 2: 实现基线网络
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.w = np.zeros(n_features)

    def get_features(self, state: np.ndarray) -> np.ndarray:
        """提取特征"""
        # ---- 在这里实现 ----
        return state
        # ---- 代码结束 ----

    def predict(self, state: np.ndarray) -> float:
        """预测 V(s)"""
        # ---- 在这里实现 ----
        features = self.get_features(state)
        return np.dot(self.w, features)
        # ---- 代码结束 ----

    def update(self, state: np.ndarray, target: float, alpha: float):
        """
        梯度更新：w ← w + α (target - V(s)) ∇_w V(s)
        """
        # ---- 在这里实现 ----
        features = self.get_features(state)
        prediction = self.predict(state)
        error = target - prediction
        self.w += alpha * error * features
        # ---- 代码结束 ----


# ============================================================
# Part 3: REINFORCE 算法（无基线）
# ============================================================

def reinforce(
    env,
    policy: LinearPolicy,
    num_episodes: int = 1000,
    alpha: float = 0.01,
    gamma: float = 0.99
) -> Tuple[LinearPolicy, List[float]]:
    """
    REINFORCE 算法（无基线）。

    TODO 3: 实现 REINFORCE
    算法:
        对每个 episode：
            1. 生成轨迹 τ
            2. 对每个时刻 t：
                G_t = Σ_{k=t}^T γ^{k-t} R_{k+1}
                θ ← θ + α G_t ∇_θ log π(A_t|S_t)

    参数:
        env, policy, num_episodes, alpha, gamma

    返回:
        policy: 训练好的策略
        reward_history: 每个 episode 的总奖励
    """
    reward_history = []

    # ---- 在这里实现 ----
    for ep in range(num_episodes):
        # 生成 episode
        states = []
        actions = []
        rewards = []

        state, _ = env.reset()
        done = False

        while not done:
            action = policy.sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            done = terminated or truncated

        # 计算回报（从后往前）
        T = len(rewards)
        returns = np.zeros(T)
        G = 0
        for t in range(T - 1, -1, -1):
            G = rewards[t] + gamma * G
            returns[t] = G

        # 策略梯度更新
        for t in range(T):
            grad = policy.grad_log_prob(states[t], actions[t])
            policy.theta += alpha * returns[t] * grad

        reward_history.append(sum(rewards))
    # ---- 代码结束 ----

    return policy, reward_history


# ============================================================
# Part 4: REINFORCE with Baseline
# ============================================================

def reinforce_with_baseline(
    env,
    policy: LinearPolicy,
    baseline: ValueBaseline,
    num_episodes: int = 1000,
    alpha_theta: float = 0.01,
    alpha_w: float = 0.1,
    gamma: float = 0.99
) -> Tuple[LinearPolicy, ValueBaseline, List[float]]:
    """
    REINFORCE with Baseline。

    TODO 4: 实现带基线的 REINFORCE
    算法:
        对每个 episode：
            1. 生成轨迹 τ
            2. 对每个时刻 t：
                G_t = Σ_{k=t}^T γ^{k-t} R_{k+1}
                δ_t = G_t - V(S_t)
                θ ← θ + α_θ δ_t ∇_θ log π(A_t|S_t)
                w ← w + α_w δ_t ∇_w V(S_t)

    参数:
        env, policy, baseline, num_episodes, alpha_theta, alpha_w, gamma

    返回:
        policy, baseline, reward_history
    """
    reward_history = []

    # ---- 在这里实现 ----
    for ep in range(num_episodes):
        # 生成 episode
        states = []
        actions = []
        rewards = []

        state, _ = env.reset()
        done = False

        while not done:
            action = policy.sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            done = terminated or truncated

        # 计算回报
        T = len(rewards)
        returns = np.zeros(T)
        G = 0
        for t in range(T - 1, -1, -1):
            G = rewards[t] + gamma * G
            returns[t] = G

        # 更新策略和基线
        for t in range(T):
            # 优势 = 回报 - 基线
            advantage = returns[t] - baseline.predict(states[t])

            # 更新策略
            grad = policy.grad_log_prob(states[t], actions[t])
            policy.theta += alpha_theta * advantage * grad

            # 更新基线
            baseline.update(states[t], returns[t], alpha_w)

        reward_history.append(sum(rewards))
    # ---- 代码结束 ----

    return policy, baseline, reward_history


# ============================================================
# Part 5: 评估和可视化
# ============================================================

def evaluate_policy(policy: LinearPolicy, env, n_eval: int = 100) -> Tuple[float, float]:
    """
    评估策略。

    TODO 5: 实现策略评估
    """
    # ---- 在这里实现 ----
    total_rewards = []

    for _ in range(n_eval):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = policy.sample_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        total_rewards.append(total_reward)

    return np.mean(total_rewards), np.std(total_rewards)
    # ---- 代码结束 ----


def plot_learning_curves(results: dict, window: int = 100):
    """绘制学习曲线"""
    plt.figure(figsize=(12, 6))

    for name, rewards in results.items():
        # 移动平均
        smoothed = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(rewards[start:i+1]))

        plt.plot(smoothed, label=name, alpha=0.8)

    plt.axhline(y=195, color='r', linestyle='--', label='Success Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Moving Avg)')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_policy_distribution(policy: LinearPolicy, env):
    """
    可视化策略分布（2D 切片）。

    TODO 6: 实现策略可视化
    """
    # ---- 在这里实现 ----
    # 固定 x 和 θ，变化 ẋ 和 θ̇
    n_points = 20
    x_dot_range = np.linspace(-3, 3, n_points)
    theta_dot_range = np.linspace(-3, 3, n_points)

    action_probs = np.zeros((n_points, n_points, 2))

    for i, x_dot in enumerate(x_dot_range):
        for j, theta_dot in enumerate(theta_dot_range):
            state = np.array([0.0, x_dot, 0.0, theta_dot])
            probs = policy.forward(state)
            action_probs[j, i] = probs

    # 绘制动作 0（左推）的概率
    plt.figure(figsize=(10, 8))
    plt.imshow(action_probs[:, :, 0], extent=[-3, 3, -3, 3],
               origin='lower', aspect='auto', cmap='RdBu_r', vmin=0, vmax=1)
    plt.colorbar(label='P(Left | state)')
    plt.xlabel('Cart Velocity (ẋ)')
    plt.ylabel('Pole Angular Velocity (θ̇)')
    plt.title('Policy Distribution (x=0, θ=0)')
    plt.tight_layout()
    # ---- 代码结束 ----


# ============================================================
# Part 6: 超参数实验
# ============================================================

def hyperparameter_experiment(
    env,
    num_episodes: int = 500,
    num_runs: int = 5
) -> dict:
    """
    超参数敏感性实验。

    TODO 7: 实现超参数实验
    测试不同的学习率和折扣因子
    """
    configs = {
        'α=0.001, γ=0.99': (0.001, 0.99),
        'α=0.01, γ=0.99': (0.01, 0.99),
        'α=0.05, γ=0.99': (0.05, 0.99),
        'α=0.01, γ=0.95': (0.01, 0.95),
    }

    results = {}

    # ---- 在这里实现 ----
    for name, (alpha, gamma) in configs.items():
        print(f"测试 {name}...")
        all_rewards = np.zeros(num_episodes)

        for run in range(num_runs):
            np.random.seed(run * 42)
            policy = LinearPolicy(n_features=4, n_actions=2)
            _, rewards = reinforce(env, policy, num_episodes, alpha, gamma)
            all_rewards += np.array(rewards)

        results[name] = all_rewards / num_runs
    # ---- 代码结束 ----

    return results


# ============================================================
# 主测试
# ============================================================

if __name__ == "__main__":
    env = gym.make('CartPole-v1')

    print("CartPole-v1 环境:")
    print(f"  状态空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  成功标准: 连续100个episode平均≥195")

    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # --- 测试 1: 策略网络 ---
    print("\n" + "=" * 60)
    print("测试 1: 策略网络")
    print("=" * 60)

    policy = LinearPolicy(n_features, n_actions)
    test_state = np.array([0.0, 0.0, 0.0, 0.0])
    probs = policy.forward(test_state)
    print(f"测试状态: {test_state}")
    print(f"动作概率: {probs}")
    print(f"采样动作: {[policy.sample_action(test_state) for _ in range(10)]}")

    # --- 测试 2: REINFORCE（无基线）---
    print("\n" + "=" * 60)
    print("测试 2: REINFORCE（无基线）")
    print("=" * 60)

    print("训练中...")
    policy_no_baseline = LinearPolicy(n_features, n_actions)
    policy_no_baseline, rewards_no_baseline = reinforce(
        env, policy_no_baseline, num_episodes=1000, alpha=0.01, gamma=0.99
    )

    print(f"\n训练结果:")
    print(f"  前100 episode平均: {np.mean(rewards_no_baseline[:100]):.1f}")
    print(f"  后100 episode平均: {np.mean(rewards_no_baseline[-100:]):.1f}")
    print(f"  最佳episode: {np.max(rewards_no_baseline):.1f}")

    # 评估
    mean_reward, std_reward = evaluate_policy(policy_no_baseline, env, n_eval=100)
    print(f"\n策略评估（100局）:")
    print(f"  平均奖励: {mean_reward:.1f} ± {std_reward:.1f}")

    # --- 测试 3: REINFORCE with Baseline ---
    print("\n" + "=" * 60)
    print("测试 3: REINFORCE with Baseline")
    print("=" * 60)

    print("训练中...")
    policy_with_baseline = LinearPolicy(n_features, n_actions)
    baseline = ValueBaseline(n_features)
    policy_with_baseline, baseline, rewards_with_baseline = reinforce_with_baseline(
        env, policy_with_baseline, baseline,
        num_episodes=1000, alpha_theta=0.01, alpha_w=0.1, gamma=0.99
    )

    print(f"\n训练结果:")
    print(f"  前100 episode平均: {np.mean(rewards_with_baseline[:100]):.1f}")
    print(f"  后100 episode平均: {np.mean(rewards_with_baseline[-100:]):.1f}")
    print(f"  最佳episode: {np.max(rewards_with_baseline):.1f}")

    # 评估
    mean_reward_b, std_reward_b = evaluate_policy(policy_with_baseline, env, n_eval=100)
    print(f"\n策略评估（100局）:")
    print(f"  平均奖励: {mean_reward_b:.1f} ± {std_reward_b:.1f}")

    # --- 测试 4: 对比 ---
    print("\n" + "=" * 60)
    print("测试 4: 有无基线对比")
    print("=" * 60)

    print(f"{'方法':<25} {'后100 ep平均':<15} {'评估平均':<15}")
    print("-" * 55)
    print(f"{'REINFORCE':<25} {np.mean(rewards_no_baseline[-100:]):<15.1f} {mean_reward:<15.1f}")
    print(f"{'REINFORCE + Baseline':<25} {np.mean(rewards_with_baseline[-100:]):<15.1f} {mean_reward_b:<15.1f}")

    # --- 测试 5: 可视化 ---
    print("\n" + "=" * 60)
    print("测试 5: 可视化")
    print("=" * 60)

    print("绘制学习曲线...")
    plot_learning_curves({
        'REINFORCE': rewards_no_baseline,
        'REINFORCE + Baseline': rewards_with_baseline,
    })
    plt.savefig('reinforce_learning_curves.png', dpi=150, bbox_inches='tight')
    print("  已保存: reinforce_learning_curves.png")

    print("\n绘制策略分布...")
    plot_policy_distribution(policy_with_baseline, env)
    plt.savefig('policy_distribution.png', dpi=150, bbox_inches='tight')
    print("  已保存: policy_distribution.png")

    # --- 测试 6: 超参数实验 ---
    print("\n" + "=" * 60)
    print("测试 6: 超参数敏感性")
    print("=" * 60)

    print("正在运行超参数实验（5次取平均）...")
    hyperparam_results = hyperparameter_experiment(env, num_episodes=500, num_runs=5)

    print(f"\n{'配置':<25} {'前100 ep':<12} {'后100 ep':<12}")
    print("-" * 50)
    for name, rewards in hyperparam_results.items():
        early = np.mean(rewards[:100])
        late = np.mean(rewards[-100:])
        print(f"{name:<25} {early:<12.1f} {late:<12.1f}")

    # --- 测试 7: 收敛分析 ---
    print("\n" + "=" * 60)
    print("测试 7: 收敛分析")
    print("=" * 60)

    # 检查是否达到成功标准
    def check_success(rewards, threshold=195, window=100):
        """检查是否连续window个episode平均≥threshold"""
        for i in range(len(rewards) - window + 1):
            if np.mean(rewards[i:i+window]) >= threshold:
                return i + window
        return None

    success_no_baseline = check_success(rewards_no_baseline)
    success_with_baseline = check_success(rewards_with_baseline)

    print(f"达到成功标准（连续100 ep平均≥195）:")
    if success_no_baseline:
        print(f"  REINFORCE: Episode {success_no_baseline}")
    else:
        print(f"  REINFORCE: 未达到")

    if success_with_baseline:
        print(f"  REINFORCE + Baseline: Episode {success_with_baseline}")
    else:
        print(f"  REINFORCE + Baseline: 未达到")

    print("\n" + "=" * 60)
    print("练习完成！")
    print("=" * 60)
    print("\n关键收获:")
    print("1. 策略梯度直接优化策略 π(a|s; θ)，不需要价值函数")
    print("2. REINFORCE 用完整回报 G_t 作为梯度信号")
    print("3. 基线 b(s) 降低方差但不改变期望，加速收敛")
    print("4. 策略梯度的优势：连续动作、随机策略、收敛保证")
    print("5. 策略梯度的劣势：高方差、样本效率低、On-Policy")
    print("6. CartPole 是测试策略梯度的经典环境")

    env.close()
    plt.show()
