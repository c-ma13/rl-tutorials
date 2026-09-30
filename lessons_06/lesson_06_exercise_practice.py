"""
Lesson 06 练习：时序差分学习 (Temporal Difference Learning)

目标：
1. 实现 TD(0) 策略评估
2. 实现 MC 策略评估（对比用）
3. 实现 n-step TD 策略评估
4. 实现 TD(λ) 策略评估（资格迹）
5. 对比收敛曲线和 α 敏感性

环境：Random Walk（7 个状态）
  T_L ── A ── B ── C ── D ── E ── T_R
  (0)   (1)  (2)  (3)  (4)  (5)  (6)

T_L: 左终止, reward = 0
T_R: 右终止, reward = 1
其他: reward = 0
策略: 均匀随机（左右各 50%）
γ = 1.0

真实价值: V(A)=1/6, V(B)=2/6, V(C)=3/6, V(D)=4/6, V(E)=5/6
"""

import numpy as np
from typing import Tuple, List, Dict

# ============================================================
# Part 0: Random Walk 环境
# ============================================================

class RandomWalk:
    """
    7 个状态的 Random Walk 环境。

    状态索引: 0(T_L), 1(A), 2(B), 3(C), 4(D), 5(E), 6(T_R)
    动作: 0=左, 1=右
    策略: 均匀随机
    """

    def __init__(self, n_states: int = 7):
        self.n_states = n_states
        self.terminal_left = 0
        self.terminal_right = n_states - 1
        self.start = n_states // 2  # C = 状态 3

        # 非终止状态
        self.non_terminal = list(range(1, n_states - 1))

        # 真实价值（均匀随机策略, γ=1）
        n_non_term = len(self.non_terminal)
        self.true_values = np.zeros(n_states)
        for i, s in enumerate(self.non_terminal):
            self.true_values[s] = (i + 1) / (n_non_term + 1)

        # 状态名
        self.state_names = ['T_L'] + [chr(ord('A') + i) for i in range(n_non_term)] + ['T_R']

    def reset(self) -> int:
        """返回起始状态"""
        return self.start

    def step(self, state: int) -> Tuple[int, float, bool]:
        """
        在均匀随机策略下执行一步。

        返回: (next_state, reward, done)
        """
        direction = np.random.choice([-1, 1])  # 左或右
        next_state = state + direction

        # 奖励
        if next_state == self.terminal_right:
            reward = 1.0
        else:
            reward = 0.0

        done = (next_state == self.terminal_left or next_state == self.terminal_right)
        return next_state, reward, done

    def generate_episode(self) -> Tuple[List[int], List[float]]:
        """
        生成一个完整的 episode。

        返回:
            states: [S_0, S_1, ..., S_T] （包含终止状态）
            rewards: [R_1, R_2, ..., R_T]
        """
        states = []
        rewards = []

        state = self.reset()
        states.append(state)

        done = False
        while not done:
            next_state, reward, done = self.step(state)
            rewards.append(reward)
            states.append(next_state)
            state = next_state

        return states, rewards


# ============================================================
# Part 1: TD(0) 策略评估
# ============================================================

def td0_prediction(
    env: RandomWalk,
    num_episodes: int = 100,
    alpha: float = 0.1,
    gamma: float = 1.0
) -> Tuple[np.ndarray, List[float]]:
    """
    TD(0) 策略评估。

    TODO 1: 实现 TD(0)
    更新规则:
        V(S_t) ← V(S_t) + α [R_{t+1} + γV(S_{t+1}) - V(S_t)]
    终止状态: V(terminal) = 0 (不更新)

    提示:
    - 生成 episode 并逐步更新
    - 注意终止状态 V = 0
    - 每个 episode 结束后记录 RMSE

    参数:
        env: RandomWalk 环境
        num_episodes: 训练 episode 数
        alpha: 学习率
        gamma: 折扣因子

    返回:
        V: shape (n_states,), 学到的价值函数
        rmse_history: 每个 episode 后与真实值的 RMSE
    """
    V = np.zeros(env.n_states)
    V[env.non_terminal] = 0.5  # 初始化非终止状态为 0.5
    rmse_history = []

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return V, rmse_history


# ============================================================
# Part 2: MC 策略评估（对比用）
# ============================================================

def mc_prediction(
    env: RandomWalk,
    num_episodes: int = 100,
    alpha: float = 0.01,
    gamma: float = 1.0
) -> Tuple[np.ndarray, List[float]]:
    """
    常数步长 MC 策略评估（每次访问）。

    TODO 2: 实现 MC 评估
    更新规则:
        V(S_t) ← V(S_t) + α [G_t - V(S_t)]
    其中 G_t 是从 t 开始的折扣回报

    提示:
    - 先生成完整 episode
    - 从后往前计算回报
    - 更新每个访问过的状态

    参数:
        env, num_episodes, alpha, gamma

    返回:
        V, rmse_history
    """
    V = np.zeros(env.n_states)
    V[env.non_terminal] = 0.5
    rmse_history = []

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return V, rmse_history


# ============================================================
# Part 3: n-step TD 策略评估
# ============================================================

def nstep_td_prediction(
    env: RandomWalk,
    n: int = 3,
    num_episodes: int = 100,
    alpha: float = 0.1,
    gamma: float = 1.0
) -> Tuple[np.ndarray, List[float]]:
    """
    n-step TD 策略评估。

    TODO 3: 实现 n-step TD
    n-step 回报:
        G_t^(n) = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})

    更新规则:
        V(S_t) ← V(S_t) + α [G_t^(n) - V(S_t)]

    提示:
    - 生成完整 episode（简化实现）
    - 对每个 t，计算 n-step 回报
    - 如果 t+n 超过 episode 长度，退化为 MC 回报

    参数:
        env, n, num_episodes, alpha, gamma

    返回:
        V, rmse_history
    """
    V = np.zeros(env.n_states)
    V[env.non_terminal] = 0.5
    rmse_history = []

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return V, rmse_history


# ============================================================
# Part 4: TD(λ) 策略评估（资格迹）
# ============================================================

def td_lambda_prediction(
    env: RandomWalk,
    lam: float = 0.8,
    num_episodes: int = 100,
    alpha: float = 0.1,
    gamma: float = 1.0
) -> Tuple[np.ndarray, List[float]]:
    """
    TD(λ) 策略评估（后向视角，累积迹）。

    TODO 4: 实现 TD(λ)
    算法:
        对每个 episode：
            e(s) = 0  ∀s
            每一步:
                δ = R + γV(S') - V(S)        # TD 误差
                e(S) = e(S) + 1               # 累积迹
                对所有非终止状态 s：
                    V(s) += α δ e(s)
                    e(s) *= γλ

    提示:
    - 资格迹在每个 episode 开始时重置为 0
    - 被访问的状态 e(s) += 1
    - 所有状态的 e(s) 每步乘以 γλ 衰减
    - 所有状态都按 α δ e(s) 更新

    参数:
        env, lam (λ), num_episodes, alpha, gamma

    返回:
        V, rmse_history
    """
    V = np.zeros(env.n_states)
    V[env.non_terminal] = 0.5
    rmse_history = []

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return V, rmse_history


# ============================================================
# Part 5: 对比实验
# ============================================================

def run_comparison(
    env: RandomWalk,
    num_episodes: int = 100,
    num_runs: int = 50,
    alpha: float = 0.1,
    gamma: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    多次运行取平均，对比各算法的 RMSE 曲线。

    TODO 5: 实现对比实验
    提示:
    - 运行 num_runs 次，每次用不同随机种子
    - 记录每种算法的 RMSE 曲线
    - 取平均作为最终结果

    参数:
        env, num_episodes, num_runs, alpha, gamma

    返回:
        results: dict, key=算法名, value=平均 RMSE 曲线 (shape: num_episodes)
    """
    results = {
        'TD(0)': np.zeros(num_episodes),
        'MC': np.zeros(num_episodes),
        '3-step TD': np.zeros(num_episodes),
        'TD(λ=0.8)': np.zeros(num_episodes),
    }

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return results


def alpha_sensitivity(
    env: RandomWalk,
    num_episodes: int = 100,
    num_runs: int = 30,
    gamma: float = 1.0
) -> Dict[str, Dict[float, float]]:
    """
    学习率 α 敏感性分析。

    TODO 6: 实现 α 敏感性实验
    提示:
    - 对不同的 α 值运行 TD(0) 和 MC
    - 记录最终 RMSE（最后 10 个 episode 的平均）
    - 找到每种算法的最优 α

    返回:
        results: dict, key=算法名, value=dict(α → 最终平均RMSE)
    """
    alphas = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    results = {'TD(0)': {}, 'MC': {}}

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return results


# ============================================================
# 辅助函数
# ============================================================

def print_values(env: RandomWalk, V: np.ndarray, label: str = ""):
    """打印价值函数"""
    print(f"\n{label}价值函数:")
    header = "  ".join([f"{env.state_names[s]:>5}" for s in env.non_terminal])
    values = "  ".join([f"{V[s]:>5.3f}" for s in env.non_terminal])
    true_vals = "  ".join([f"{env.true_values[s]:>5.3f}" for s in env.non_terminal])

    print(f"  状态:  {header}")
    print(f"  估计值: {values}")
    print(f"  真实值: {true_vals}")

    rmse = np.sqrt(np.mean((V[env.non_terminal] - env.true_values[env.non_terminal]) ** 2))
    print(f"  RMSE: {rmse:.4f}")


# ============================================================
# 主测试
# ============================================================

if __name__ == "__main__":
    env = RandomWalk()

    print("Random Walk 环境:")
    print(f"  状态: {env.state_names}")
    print(f"  真实价值: {[f'{env.true_values[s]:.3f}' for s in env.non_terminal]}")
    print(f"  起始状态: {env.state_names[env.start]}")

    # --- 测试 1: TD(0) ---
    print("\n" + "=" * 60)
    print("测试 1: TD(0) 策略评估")
    print("=" * 60)

    V_td, rmse_td = td0_prediction(env, num_episodes=100, alpha=0.1, gamma=1.0)
    print_values(env, V_td, "TD(0) ")

    print(f"\n前 10 个 episode 的 RMSE:")
    print([f"{r:.4f}" for r in rmse_td[:10]])

    # --- 测试 2: MC ---
    print("\n" + "=" * 60)
    print("测试 2: MC 策略评估")
    print("=" * 60)

    V_mc, rmse_mc = mc_prediction(env, num_episodes=100, alpha=0.01, gamma=1.0)
    print_values(env, V_mc, "MC ")

    # --- 测试 3: n-step TD ---
    print("\n" + "=" * 60)
    print("测试 3: n-step TD 策略评估")
    print("=" * 60)

    for n in [1, 2, 3, 5, 10]:
        V_n, rmse_n = nstep_td_prediction(env, n=n, num_episodes=100, alpha=0.1, gamma=1.0)
        rmse_final = np.mean(rmse_n[-10:])
        print(f"  n={n:>2}: RMSE(最终) = {rmse_final:.4f}")

    # --- 测试 4: TD(λ) ---
    print("\n" + "=" * 60)
    print("测试 4: TD(λ) 策略评估")
    print("=" * 60)

    for lam in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        V_lam, rmse_lam = td_lambda_prediction(env, lam=lam, num_episodes=100, alpha=0.05, gamma=1.0)
        rmse_final = np.mean(rmse_lam[-10:])
        print(f"  λ={lam:.1f}: RMSE(最终) = {rmse_final:.4f}")

    print(f"\n验证: λ=0 应接近 TD(0), λ=1 应接近 MC")

    # --- 测试 5: 算法对比 ---
    print("\n" + "=" * 60)
    print("测试 5: 算法对比（多次运行取平均）")
    print("=" * 60)

    print("正在运行对比实验（50 次取平均）...")
    comparison = run_comparison(env, num_episodes=100, num_runs=50)

    print(f"\n{'算法':<15} {'RMSE(ep 10)':<15} {'RMSE(ep 50)':<15} {'RMSE(ep 100)':<15}")
    print("-" * 60)
    for name, rmse_curve in comparison.items():
        print(f"{name:<15} {rmse_curve[9]:<15.4f} {rmse_curve[49]:<15.4f} {rmse_curve[99]:<15.4f}")

    # --- 测试 6: α 敏感性 ---
    print("\n" + "=" * 60)
    print("测试 6: 学习率 α 敏感性分析")
    print("=" * 60)

    print("正在运行 α 敏感性实验...")
    alpha_results = alpha_sensitivity(env, num_episodes=100, num_runs=30)

    print(f"\n{'α':<10} {'TD(0) RMSE':<15} {'MC RMSE':<15}")
    print("-" * 40)
    for a in sorted(alpha_results['TD(0)'].keys()):
        td_r = alpha_results['TD(0)'][a]
        mc_r = alpha_results['MC'][a]
        print(f"{a:<10.2f} {td_r:<15.4f} {mc_r:<15.4f}")

    # 最优 α
    best_td_alpha = min(alpha_results['TD(0)'], key=alpha_results['TD(0)'].get)
    best_mc_alpha = min(alpha_results['MC'], key=alpha_results['MC'].get)
    print(f"\n最优 α: TD(0) = {best_td_alpha}, MC = {best_mc_alpha}")

    print("\n" + "=" * 60)
    print("练习完成！")
    print("=" * 60)
    print("\n关键收获:")
    print("1. TD(0) 每步更新，不需要等 episode 结束，收敛通常比 MC 快")
    print("2. TD 有偏但低方差，MC 无偏但高方差 —— 偏差-方差权衡")
    print("3. n-step TD 在 MC 和 TD(0) 之间插值，n=3-5 通常效果最好")
    print("4. TD(λ) 通过资格迹统一了 TD 和 MC，λ=0 退化为 TD(0)，λ=1 退化为 MC")
    print("5. α 的选择对收敛至关重要：TD 对 α 更鲁棒，MC 需要较小的 α")
    print("6. 自举（bootstrapping）是 TD 的核心优势，利用状态间的关系加速学习")
