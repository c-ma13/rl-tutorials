"""
Lesson 11 练习：Actor-Critic 方法

目标：
1. 实现 One-Step Actor-Critic
2. 实现 A2C (Advantage Actor-Critic)
3. 实现 GAE (Generalized Advantage Estimation)
4. 在 CartPole 上训练
5. 对比 REINFORCE、Actor-Critic、A2C
6. 超参数实验

环境：CartPole-v1
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

# ============================================================
# Part 1: Actor 网络（策略）
# ============================================================

class Actor:
    """
    Actor 网络：π(a|s; θ)

    TODO 1: 实现 Actor（复用 Lesson 10 的代码）
    需要实现:
    - forward(state): 计算动作概率
    - sample_action(state): 采样动作
    - log_prob(state, action): 计算 log π(a|s)
    - grad_log_prob(state, action): 计算梯度
    - entropy(state): 计算熵
    """

    def __init__(self, n_features: int, n_actions: int):
        self.n_features = n_features
        self.n_actions = n_actions
        self.theta = np.random.randn(n_features, n_actions) * 0.01

    def forward(self, state: np.ndarray) -> np.ndarray:
        """计算动作概率"""
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def sample_action(self, state: np.ndarray) -> int:
        """采样动作"""
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def log_prob(self, state: np.ndarray, action: int) -> float:
        """计算 log π(a|s)"""
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def grad_log_prob(self, state: np.ndarray, action: int) -> np.ndarray:
        """计算 ∇_θ log π(a|s)"""
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def entropy(self, state: np.ndarray) -> float:
        """计算策略熵 H(π(·|s)) = -Σ π(a|s) log π(a|s)"""
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----


# ============================================================
# Part 2: Critic 网络（价值函数）
# ============================================================

class Critic:
    """
    Critic 网络：V(s; w)

    TODO 2: 实现 Critic
    需要实现:
    - predict(state): 预测 V(s)
    - grad(state): 计算梯度
    - update(state, td_error, alpha): 更新权重
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.w = np.zeros(n_features)

    def predict(self, state: np.ndarray) -> float:
        """预测 V(s) = w^T s"""
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def grad(self, state: np.ndarray) -> np.ndarray:
        """计算 ∇_w V(s) = s"""
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def update(self, state: np.ndarray, td_error: float, alpha: float):
        """更新：w ← w + α δ ∇_w V(s)"""
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----


# ============================================================
# Part 3: One-Step Actor-Critic
# ============================================================

def one_step_actor_critic(
    env,
    actor: Actor,
    critic: Critic,
    num_episodes: int = 1000,
    alpha_theta: float = 0.01,
    alpha_w: float = 0.1,
    gamma: float = 0.99
) -> Tuple[Actor, Critic, List[float]]:
    """
    One-Step Actor-Critic 算法。

    TODO 3: 实现 One-Step Actor-Critic
    算法:
        对每个 episode：
            S ← 起始
            重复:
                A ← π(·|S)
                执行 A, 观察 R, S'
                δ ← R + γ V(S') - V(S)      # TD 误差
                w ← w + α_w δ ∇_w V(S)      # 更新 Critic
                θ ← θ + α_θ δ ∇_θ log π(A|S) # 更新 Actor
                S ← S'

    提示:
    - 每步都更新，不需要等 episode 结束
    - TD 误差是优势的无偏估计

    参数:
        env, actor, critic, num_episodes, alpha_theta, alpha_w, gamma

    返回:
        actor, critic, reward_history
    """
    reward_history = []

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return actor, critic, reward_history


# ============================================================
# Part 4: A2C (n-step + 优势)
# ============================================================

def compute_n_step_returns(
    rewards: List[float],
    values: List[float],
    gamma: float,
    n_steps: int = 5
) -> List[float]:
    """
    计算 n-step 回报。

    TODO 4: 实现 n-step 回报计算
    G_t = Σ_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n V(S_{t+n})

    提示:
    - 对每个时刻 t，累加 n 步奖励
    - 如果 t+n 超过 episode 长度，只累加到末尾
    - 加上 bootstrap 项 γ^n V(S_{t+n})

    参数:
        rewards: [R_1, R_2, ..., R_T]
        values: [V(S_0), V(S_1), ..., V(S_T)]
        gamma: 折扣因子
        n_steps: 步数

    返回:
        returns: [G_0, G_1, ..., G_{T-1}]
    """
    T = len(rewards)
    returns = []

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return returns


def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float,
    lam: float = 0.95
) -> List[float]:
    """
    计算 GAE (Generalized Advantage Estimation)。

    TODO 5: 实现 GAE
    A_t^GAE = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
    其中 δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)

    递推公式（从后往前）:
        gae = δ_t + γλ gae

    提示:
    - 从最后一个时刻开始往前计算
    - 初始 gae = 0
    - 每步计算 TD 误差 δ_t，然后更新 gae

    参数:
        rewards, values, gamma, lam

    返回:
        advantages: [A_0, A_1, ..., A_{T-1}]
    """
    T = len(rewards)
    advantages = []
    gae = 0

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return advantages


def a2c(
    env,
    actor: Actor,
    critic: Critic,
    num_episodes: int = 1000,
    alpha_theta: float = 0.01,
    alpha_w: float = 0.1,
    gamma: float = 0.99,
    n_steps: int = 5,
    lam: float = 0.95,
    entropy_coef: float = 0.01
) -> Tuple[Actor, Critic, List[float]]:
    """
    A2C (Advantage Actor-Critic) 算法。

    TODO 6: 实现 A2C
    算法:
        对每个 episode：
            1. 收集完整轨迹
            2. 计算所有状态的价值 V(s)
            3. 计算 GAE 优势
            4. 更新 Critic（目标是 n-step 回报）
            5. 更新 Actor（策略梯度 + 熵正则化）

    提示:
    - 先收集完整 episode
    - 用 compute_gae 计算优势
    - 用 compute_n_step_returns 计算 Critic 的目标
    - 熵正则化：增加策略熵，鼓励探索

    参数:
        env, actor, critic, num_episodes, alpha_theta, alpha_w,
        gamma, n_steps, lam, entropy_coef

    返回:
        actor, critic, reward_history
    """
    reward_history = []

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return actor, critic, reward_history


# ============================================================
# Part 5: 评估和可视化
# ============================================================

def evaluate_policy(actor: Actor, env, n_eval: int = 100) -> Tuple[float, float]:
    """
    评估策略。

    TODO 7: 实现策略评估
    """
    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----


def plot_learning_curves(results: Dict[str, List[float]], window: int = 100):
    """绘制学习曲线"""
    plt.figure(figsize=(12, 6))

    for name, rewards in results.items():
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


# ============================================================
# Part 6: 超参数实验
# ============================================================

def hyperparameter_experiment(
    env,
    num_episodes: int = 500,
    num_runs: int = 5
) -> Dict[str, np.ndarray]:
    """
    超参数实验。

    TODO 8: 实现超参数实验
    测试不同的 n_steps 和 λ
    """
    configs = {
        'n=1, λ=0': (1, 0.0),
        'n=5, λ=0.95': (5, 0.95),
        'n=10, λ=0.95': (10, 0.95),
        'n=5, λ=1.0': (5, 1.0),
    }

    results = {}

    # ---- 在这里实现 ----
    pass
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

    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # --- 测试 1: One-Step Actor-Critic ---
    print("\n" + "=" * 60)
    print("测试 1: One-Step Actor-Critic")
    print("=" * 60)

    print("训练中...")
    actor_1step = Actor(n_features, n_actions)
    critic_1step = Critic(n_features)
    actor_1step, critic_1step, rewards_1step = one_step_actor_critic(
        env, actor_1step, critic_1step,
        num_episodes=1000, alpha_theta=0.01, alpha_w=0.1, gamma=0.99
    )

    print(f"\n训练结果:")
    print(f"  前100 episode平均: {np.mean(rewards_1step[:100]):.1f}")
    print(f"  后100 episode平均: {np.mean(rewards_1step[-100:]):.1f}")

    mean_1step, std_1step = evaluate_policy(actor_1step, env, n_eval=100)
    print(f"\n策略评估（100局）:")
    print(f"  平均奖励: {mean_1step:.1f} ± {std_1step:.1f}")

    # --- 测试 2: A2C ---
    print("\n" + "=" * 60)
    print("测试 2: A2C (n=5, λ=0.95)")
    print("=" * 60)

    print("训练中...")
    actor_a2c = Actor(n_features, n_actions)
    critic_a2c = Critic(n_features)
    actor_a2c, critic_a2c, rewards_a2c = a2c(
        env, actor_a2c, critic_a2c,
        num_episodes=1000, alpha_theta=0.01, alpha_w=0.1, gamma=0.99,
        n_steps=5, lam=0.95, entropy_coef=0.01
    )

    print(f"\n训练结果:")
    print(f"  前100 episode平均: {np.mean(rewards_a2c[:100]):.1f}")
    print(f"  后100 episode平均: {np.mean(rewards_a2c[-100:]):.1f}")

    mean_a2c, std_a2c = evaluate_policy(actor_a2c, env, n_eval=100)
    print(f"\n策略评估（100局）:")
    print(f"  平均奖励: {mean_a2c:.1f} ± {std_a2c:.1f}")

    # --- 测试 3: 对比 REINFORCE ---
    print("\n" + "=" * 60)
    print("测试 3: 与 REINFORCE 对比")
    print("=" * 60)

    # 导入 REINFORCE（如果可用）
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../lessons_10/exercises'))
        from lesson_10_exercise import reinforce, LinearPolicy

        print("训练 REINFORCE...")
        policy_reinforce = LinearPolicy(n_features, n_actions)
        _, rewards_reinforce = reinforce(
            env, policy_reinforce, num_episodes=1000, alpha=0.01, gamma=0.99
        )

        print(f"\n{'方法':<25} {'后100 ep平均':<15}")
        print("-" * 40)
        print(f"{'REINFORCE':<25} {np.mean(rewards_reinforce[-100:]):<15.1f}")
        print(f"{'One-Step AC':<25} {np.mean(rewards_1step[-100:]):<15.1f}")
        print(f"{'A2C':<25} {np.mean(rewards_a2c[-100:]):<15.1f}")

    except ImportError:
        print("无法导入 REINFORCE，跳过对比")
        rewards_reinforce = None

    # --- 测试 4: 可视化 ---
    print("\n" + "=" * 60)
    print("测试 4: 可视化")
    print("=" * 60)

    print("绘制学习曲线...")
    results_to_plot = {
        'One-Step AC': rewards_1step,
        'A2C': rewards_a2c,
    }
    if rewards_reinforce is not None:
        results_to_plot['REINFORCE'] = rewards_reinforce

    plot_learning_curves(results_to_plot)
    plt.savefig('actor_critic_learning_curves.png', dpi=150, bbox_inches='tight')
    print("  已保存: actor_critic_learning_curves.png")

    # --- 测试 5: GAE 可视化 ---
    print("\n" + "=" * 60)
    print("测试 5: GAE 优势估计")
    print("=" * 60)

    # 生成一个 episode 并计算不同 λ 的优势
    state, _ = env.reset()
    states, rewards, values = [], [], []
    done = False

    while not done and len(states) < 50:
        states.append(state)
        values.append(critic_a2c.predict(state))
        action = actor_a2c.sample_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated

    if done:
        values.append(0.0)
    else:
        values.append(critic_a2c.predict(state))

    # 计算不同 λ 的优势
    plt.figure(figsize=(12, 6))
    for lam in [0.0, 0.5, 0.95, 1.0]:
        advantages = compute_gae(rewards, values[:-1], gamma=0.99, lam=lam)
        plt.plot(advantages, label=f'λ={lam}', alpha=0.8)

    plt.xlabel('Time Step')
    plt.ylabel('Advantage')
    plt.title('GAE with Different λ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gae_comparison.png', dpi=150, bbox_inches='tight')
    print("  已保存: gae_comparison.png")

    # --- 测试 6: 超参数实验 ---
    print("\n" + "=" * 60)
    print("测试 6: 超参数敏感性")
    print("=" * 60)

    print("正在运行超参数实验（5次取平均）...")
    hyperparam_results = hyperparameter_experiment(env, num_episodes=500, num_runs=5)

    print(f"\n{'配置':<20} {'前100 ep':<12} {'后100 ep':<12}")
    print("-" * 45)
    for name, rewards in hyperparam_results.items():
        early = np.mean(rewards[:100])
        late = np.mean(rewards[-100:])
        print(f"{name:<20} {early:<12.1f} {late:<12.1f}")

    # --- 测试 7: 收敛分析 ---
    print("\n" + "=" * 60)
    print("测试 7: 收敛分析")
    print("=" * 60)

    def check_success(rewards, threshold=195, window=100):
        for i in range(len(rewards) - window + 1):
            if np.mean(rewards[i:i+window]) >= threshold:
                return i + window
        return None

    success_1step = check_success(rewards_1step)
    success_a2c = check_success(rewards_a2c)

    print(f"达到成功标准（连续100 ep平均≥195）:")
    if success_1step:
        print(f"  One-Step AC: Episode {success_1step}")
    else:
        print(f"  One-Step AC: 未达到")

    if success_a2c:
        print(f"  A2C: Episode {success_a2c}")
    else:
        print(f"  A2C: 未达到")

    print("\n" + "=" * 60)
    print("练习完成！")
    print("=" * 60)
    print("\n关键收获:")
    print("1. Actor-Critic = Actor(策略) + Critic(价值)，结合两者优势")
    print("2. 用 TD 误差 δ 替代回报 G_t，降低方差、可每步更新")
    print("3. 优势函数 A(s,a) = Q(s,a) - V(s) 衡量相对优势")
    print("4. A2C 用 n-step 回报和 GAE，平衡偏差-方差")
    print("5. 熵正则化鼓励探索，防止策略过早收敛")
    print("6. Actor-Critic 是现代 RL（PPO、SAC）的基础")

    env.close()
    plt.show()
