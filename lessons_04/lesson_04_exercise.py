"""
Lesson 04 练习：价值迭代 (Value Iteration)

目标：
1. 实现标准价值迭代（同步更新）
2. 实现原地价值迭代（异步更新）
3. 实现策略提取
4. 对比价值迭代与策略迭代的效率
5. 可视化收敛过程

环境：Frozen Lake 4x4
SFFF       S = 起点 (0)
FHFH       F = 冰面
FFFH       H = 冰洞 (终止)
HFFG       G = 目标 (15, reward=1)

动作: 0=左, 1=下, 2=右, 3=上
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, List
import time

# ============================================================
# Part 0: 复用 Lesson 03 的环境接口
# ============================================================

def build_transition_matrices(env) -> Tuple[np.ndarray, np.ndarray]:
    """从 Gym 环境构建转移概率矩阵和奖励矩阵"""
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))

    for s in range(n_states):
        for a in range(n_actions):
            transitions = env.unwrapped.P[s][a]
            for prob, next_s, reward, done in transitions:
                P[s, a, next_s] += prob
                R[s, a] += prob * reward

    return P, R


# ============================================================
# Part 1: 标准价值迭代（同步更新）
# ============================================================

def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iter: int = 1000
) -> Tuple[np.ndarray, int, List[float]]:
    """
    标准价值迭代算法（同步更新）。

    TODO 1: 实现价值迭代
    算法:
        V_0 = 0
        重复:
            对每个状态 s:
                V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a) + γ V_k(s')]
        直到 ||V_{k+1} - V_k||_∞ < θ

    提示:
    - 使用两个数组 V 和 V_new
    - 计算 Q(s,a) = R[s,a] + γ Σ_{s'} P[s,a,s'] V[s']
    - V_new[s] = max_a Q(s,a)

    参数:
        P: shape (n_states, n_actions, n_states)
        R: shape (n_states, n_actions)
        gamma: 折扣因子
        theta: 收敛阈值
        max_iter: 最大迭代次数

    返回:
        V: shape (n_states,), 最优价值函数
        num_iters: 实际迭代次数
        v_history: 每次迭代的 V[0]，用于绘制收敛曲线
    """
    n_states = P.shape[0]
    V = np.zeros(n_states)
    v_history = []

    # ---- 在这里实现 ----
    for i in range(max_iter):
        V_new = np.zeros(n_states)

        for s in range(n_states):
            # 计算所有动作的 Q 值
            q_values = R[s, :] + gamma * np.sum(P[s, :, :] * V, axis=1)
            V_new[s] = np.max(q_values)

        v_history.append(V_new[0])

        # 检查收敛
        delta = np.max(np.abs(V_new - V))
        V = V_new

        if delta < theta:
            return V, i + 1, v_history

    return V, max_iter, v_history
    # ---- 代码结束 ----


# ============================================================
# Part 2: 原地价值迭代（异步更新）
# ============================================================

def value_iteration_inplace(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iter: int = 1000
) -> Tuple[np.ndarray, int, List[float]]:
    """
    原地价值迭代（异步更新）。

    TODO 2: 实现原地更新版本
    算法:
        V = 0
        重复:
            delta = 0
            对每个状态 s:
                v_old = V[s]
                V[s] = max_a Σ_{s'} P(s'|s,a) [R(s,a) + γ V[s']]
                delta = max(delta, |V[s] - v_old|)
        直到 delta < θ

    提示:
    - 只用一个数组 V，直接覆盖
    - 后面的状态会立即使用前面更新的值
    - 通常比同步更新收敛更快

    参数:
        同 value_iteration

    返回:
        V, num_iters, v_history
    """
    n_states = P.shape[0]
    V = np.zeros(n_states)
    v_history = []

    # ---- 在这里实现 ----
    for i in range(max_iter):
        delta = 0

        for s in range(n_states):
            v_old = V[s]
            q_values = R[s, :] + gamma * np.sum(P[s, :, :] * V, axis=1)
            V[s] = np.max(q_values)
            delta = max(delta, abs(V[s] - v_old))

        v_history.append(V[0])

        if delta < theta:
            return V, i + 1, v_history

    return V, max_iter, v_history
    # ---- 代码结束 ----


# ============================================================
# Part 3: 策略提取
# ============================================================

def extract_policy(
    P: np.ndarray,
    R: np.ndarray,
    V: np.ndarray,
    gamma: float = 0.99
) -> np.ndarray:
    """
    从价值函数提取最优策略。

    TODO 3: 实现策略提取
    公式: π*(s) = argmax_a Σ_{s'} P(s'|s,a) [R(s,a) + γ V(s')]

    提示:
    - 对每个状态，计算所有动作的 Q 值
    - 选择 Q 值最大的动作

    参数:
        P, R, V, gamma: 同上

    返回:
        policy: shape (n_states,), policy[s] = 最优动作
    """
    n_states, n_actions = R.shape
    policy = np.zeros(n_states, dtype=int)

    # ---- 在这里实现 ----
    for s in range(n_states):
        q_values = R[s, :] + gamma * np.sum(P[s, :, :] * V, axis=1)
        policy[s] = np.argmax(q_values)
    # ---- 代码结束 ----

    return policy


# ============================================================
# Part 4: 与策略迭代的对比
# ============================================================

def policy_iteration_simple(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.99,
    theta: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    简化的策略迭代（用于对比）。

    TODO 4: 实现策略迭代
    提示: 复用 Lesson 03 的代码逻辑
    - 策略评估：矩阵求逆法
    - 策略改进：贪心选择

    返回:
        policy, V, num_iters
    """
    n_states, n_actions = R.shape
    policy = np.random.randint(0, n_actions, size=n_states)

    # ---- 在这里实现 ----
    for i in range(100):
        # 策略评估（矩阵法）
        P_pi = np.zeros((n_states, n_states))
        r_pi = np.zeros(n_states)

        for s in range(n_states):
            a = policy[s]
            P_pi[s, :] = P[s, a, :]
            r_pi[s] = R[s, a]

        V = np.linalg.solve(np.eye(n_states) - gamma * P_pi, r_pi)

        # 策略改进
        new_policy = extract_policy(P, R, V, gamma)

        # 检查收敛
        if np.array_equal(new_policy, policy):
            return new_policy, V, i + 1

        policy = new_policy

    return policy, V, 100
    # ---- 代码结束 ----


# ============================================================
# Part 5: 可视化和测试
# ============================================================

def print_policy(policy: np.ndarray, shape: Tuple[int, int] = (4, 4)):
    """打印策略（箭头表示）"""
    action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    print("策略:")
    for row in range(shape[0]):
        line = ""
        for col in range(shape[1]):
            s = row * shape[1] + col
            line += f" {action_symbols[policy[s]]} "
        print(line)


def print_values(V: np.ndarray, shape: Tuple[int, int] = (4, 4)):
    """打印价值函数"""
    print("价值函数:")
    for row in range(shape[0]):
        line = ""
        for col in range(shape[1]):
            s = row * shape[1] + col
            line += f" {V[s]:5.3f} "
        print(line)


def compare_algorithms(P, R, gamma=0.99, theta=1e-6):
    """
    对比三种算法的性能。

    TODO 5: 实现算法对比
    提示:
    - 运行三种算法：价值迭代（同步）、价值迭代（原地）、策略迭代
    - 记录：迭代次数、运行时间、最终 V[0]
    - 验证三种算法得到相同的最优策略

    返回:
        results: dict, 包含各算法的统计信息
    """
    results = {}

    # ---- 在这里实现 ----
    # 1. 价值迭代（同步）
    start = time.time()
    V_vi, iters_vi, hist_vi = value_iteration(P, R, gamma, theta)
    time_vi = time.time() - start
    policy_vi = extract_policy(P, R, V_vi, gamma)

    results['value_iteration'] = {
        'iters': iters_vi,
        'time': time_vi,
        'V_start': V_vi[0],
        'policy': policy_vi,
        'history': hist_vi
    }

    # 2. 价值迭代（原地）
    start = time.time()
    V_vi_inplace, iters_vi_inplace, hist_vi_inplace = value_iteration_inplace(P, R, gamma, theta)
    time_vi_inplace = time.time() - start
    policy_vi_inplace = extract_policy(P, R, V_vi_inplace, gamma)

    results['value_iteration_inplace'] = {
        'iters': iters_vi_inplace,
        'time': time_vi_inplace,
        'V_start': V_vi_inplace[0],
        'policy': policy_vi_inplace,
        'history': hist_vi_inplace
    }

    # 3. 策略迭代
    start = time.time()
    policy_pi, V_pi, iters_pi = policy_iteration_simple(P, R, gamma, theta)
    time_pi = time.time() - start

    results['policy_iteration'] = {
        'iters': iters_pi,
        'time': time_pi,
        'V_start': V_pi[0],
        'policy': policy_pi
    }
    # ---- 代码结束 ----

    return results


# ============================================================
# 主测试
# ============================================================

if __name__ == "__main__":
    # 测试 1: 确定性环境
    print("=" * 60)
    print("测试 1: Frozen Lake 4x4 (确定性)")
    print("=" * 60)

    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)
    P, R = build_transition_matrices(env)

    print(f"状态数: {env.observation_space.n}")
    print(f"动作数: {env.action_space.n}")

    # 测试价值迭代
    print("\n" + "=" * 60)
    print("测试 2: 价值迭代（同步更新）")
    print("=" * 60)

    V_opt, num_iters, v_history = value_iteration(P, R, gamma=0.99, theta=1e-6)
    policy_opt = extract_policy(P, R, V_opt, gamma=0.99)

    print(f"迭代次数: {num_iters}")
    print(f"V*(start) = {V_opt[0]:.6f}")
    print()
    print_policy(policy_opt)
    print()
    print_values(V_opt)

    # 测试原地更新
    print("\n" + "=" * 60)
    print("测试 3: 价值迭代（原地更新）")
    print("=" * 60)

    V_inplace, num_iters_inplace, v_history_inplace = value_iteration_inplace(
        P, R, gamma=0.99, theta=1e-6
    )
    policy_inplace = extract_policy(P, R, V_inplace, gamma=0.99)

    print(f"迭代次数: {num_iters_inplace}")
    print(f"V*(start) = {V_inplace[0]:.6f}")
    print(f"与同步版本的差异: {np.max(np.abs(V_inplace - V_opt)):.2e}")
    print(f"策略是否相同: {np.array_equal(policy_inplace, policy_opt)}")

    # 算法对比
    print("\n" + "=" * 60)
    print("测试 4: 算法对比（确定性环境）")
    print("=" * 60)

    results = compare_algorithms(P, R, gamma=0.99, theta=1e-6)

    print(f"{'算法':<25} {'迭代次数':<10} {'时间(ms)':<12} {'V*(start)':<12}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<25} {res['iters']:<10} {res['time']*1000:<12.2f} {res['V_start']:<12.6f}")

    # 验证策略一致性
    print("\n策略一致性检查:")
    vi_policy = results['value_iteration']['policy']
    vi_inplace_policy = results['value_iteration_inplace']['policy']
    pi_policy = results['policy_iteration']['policy']

    print(f"  价值迭代 vs 价值迭代(原地): {np.array_equal(vi_policy, vi_inplace_policy)}")
    print(f"  价值迭代 vs 策略迭代: {np.array_equal(vi_policy, pi_policy)}")

    # 测试随机环境
    print("\n" + "=" * 60)
    print("测试 5: 随机环境 (is_slippery=True)")
    print("=" * 60)

    env_slippery = gym.make('FrozenLake-v1', is_slippery=True, render_mode=None)
    P_slip, R_slip = build_transition_matrices(env_slippery)

    results_slip = compare_algorithms(P_slip, R_slip, gamma=0.99, theta=1e-6)

    print(f"{'算法':<25} {'迭代次数':<10} {'时间(ms)':<12} {'V*(start)':<12}")
    print("-" * 60)
    for name, res in results_slip.items():
        print(f"{name:<25} {res['iters']:<10} {res['time']*1000:<12.2f} {res['V_start']:<12.6f}")

    print("\n最优策略（随机环境）:")
    print_policy(results_slip['value_iteration']['policy'])

    # 收敛曲线对比
    print("\n" + "=" * 60)
    print("测试 6: 收敛曲线分析")
    print("=" * 60)

    vi_hist = results['value_iteration']['history']
    vi_inplace_hist = results['value_iteration_inplace']['history']

    print(f"价值迭代（同步）前 10 次迭代的 V[0]:")
    print([f"{v:.4f}" for v in vi_hist[:10]])
    print(f"\n价值迭代（原地）前 10 次迭代的 V[0]:")
    print([f"{v:.4f}" for v in vi_inplace_hist[:10]])

    print(f"\n原地更新加速比: {len(vi_hist) / len(vi_inplace_hist):.2f}x")

    print("\n" + "=" * 60)
    print("练习完成！")
    print("=" * 60)
    print("\n关键收获:")
    print("1. 价值迭代直接应用 Bellman 最优算子，无需显式维护策略")
    print("2. 原地更新（异步 DP）通常比同步更新收敛更快")
    print("3. 价值迭代的迭代次数多于策略迭代，但每次迭代更简单")
    print("4. 三种算法（价值迭代、原地更新、策略迭代）得到相同的最优策略")
    print("5. 随机环境下收敛更慢，但算法仍然有效")

    env.close()
    env_slippery.close()
