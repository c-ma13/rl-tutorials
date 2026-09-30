"""
Lesson 03 练习：策略迭代 (Policy Iteration)

目标：
1. 从 Gym 环境构建转移概率矩阵
2. 实现策略评估（迭代法 + 矩阵法）
3. 实现策略改进
4. 实现完整的策略迭代算法
5. 在 Frozen Lake 上测试并可视化结果

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

# ============================================================
# Part 1: 环境接口 - 构建转移矩阵
# ============================================================

def build_transition_matrices(env) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 Gym 环境构建转移概率矩阵和奖励矩阵。

    TODO 1: 提取环境的转移概率
    提示:
    - env.P[s][a] 返回 [(prob, next_s, reward, done), ...]
    - P[s, a, s'] = 转移到 s' 的概率
    - R[s, a] = 期望即时奖励

    参数:
        env: Gym 环境对象

    返回:
        P: shape (n_states, n_actions, n_states), 转移概率
        R: shape (n_states, n_actions), 期望奖励
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))

    # ---- 在这里实现 ----
    for s in range(n_states):
        for a in range(n_actions):
            for prob,next_s,reward,done in env.unwrapped.P[s][a]:
                P[s][a][next_s] = prob
                R[s][a] += prob * reward
    # ---- 代码结束 ----
    return P, R


# ============================================================
# Part 2: 策略评估
# ============================================================

def policy_evaluation_iterative(
    P: np.ndarray,
    R: np.ndarray,
    policy: np.ndarray,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iter: int = 1000
) -> Tuple[np.ndarray, int]:
    """
    迭代策略评估。

    TODO 2: 实现迭代策略评估
    算法:
        v_0 = 0
        重复:
            v_{k+1}(s) = Σ_{s'} P(s'|s,π(s)) [R(s,π(s)) + γ v_k(s')]
        直到 ||v_{k+1} - v_k||_∞ < θ

    参数:
        P: shape (n_states, n_actions, n_states)
        R: shape (n_states, n_actions)
        policy: shape (n_states,), policy[s] = 动作索引
        gamma: 折扣因子
        theta: 收敛阈值
        max_iter: 最大迭代次数

    返回:
        V: shape (n_states,), 收敛后的价值函数
        num_iters: 实际迭代次数
    """
    n_states = P.shape[0]
    V = np.zeros(n_states)

    # ---- 在这里实现 ----
    for i in range(max_iter):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            a = policy[s]
            for s_prime in range(n_states):
                V_new[s] += P[s,a,s_prime] * (R[s,a] + gamma * V[s_prime])
        if np.max(np.abs(V_new-V)) < theta:
            break
        V = V_new
    return V,i
    # ---- 代码结束 ----


def policy_evaluation_exact(
    P: np.ndarray,
    R: np.ndarray,
    policy: np.ndarray,
    gamma: float = 0.99
) -> np.ndarray:
    """
    精确策略评估（矩阵求逆法）。

    TODO 3: 实现矩阵求逆法
    公式: V^π = (I - γP^π)^{-1} r^π
    其中:
        P^π[s, s'] = P(s'|s, π(s))
        r^π[s] = R(s, π(s))

    参数:
        P, R, policy, gamma: 同上

    返回:
        V: shape (n_states,)
    """
    n_states = P.shape[0]

    # ---- 在这里实现 ----
    p_pi = np.zeros((n_states,n_states))
    r_pi = np.zeros(n_states)

    for i in range(n_states):
        p_pi[i,:] = P[i,policy[i],:]
        r_pi[i] = R[i,policy[i]]
    V = np.linalg.solve(np.eye(n_states) - gamma * p_pi, r_pi)
    return V

    # ---- 代码结束 ----


# ============================================================
# Part 3: 策略改进
# ============================================================

def policy_improvement(
    P: np.ndarray,
    R: np.ndarray,
    V: np.ndarray,
    gamma: float = 0.99
) -> np.ndarray:
    """
    策略改进：贪心选择最优动作。

    TODO 4: 实现策略改进
    公式: π'(s) = argmax_a Σ_{s'} P(s'|s,a) [R(s,a) + γ V(s')]

    参数:
        P: shape (n_states, n_actions, n_states)
        R: shape (n_states, n_actions)
        V: shape (n_states,), 当前价值函数
        gamma: 折扣因子

    返回:
        policy: shape (n_states,), 改进后的策略
    """
    n_states, n_actions = R.shape
    policy = np.zeros(n_states, dtype=int)

    # ---- 在这里实现 ----
    for i in range(n_states):
        q = np.zeros(n_actions)
        for j in range(n_actions):
            q[j] = R[i,j] + gamma * np.sum(P[i,j,:] * V)
        policy[i] = np.argmax(q)
    # ---- 代码结束 ----

    return policy


# ============================================================
# Part 4: 策略迭代主算法
# ============================================================

def policy_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iter: int = 100,
    use_exact_eval: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    策略迭代算法。

    TODO 5: 实现完整的策略迭代
    算法:
        1. 初始化随机策略 π_0
        2. 重复:
            a. 策略评估: 求解 V^π_k
            b. 策略改进: π_{k+1}(s) = argmax_a Q^π_k(s,a)
            c. 如果 π_{k+1} = π_k，停止
        3. 返回 π*, V*

    参数:
        P, R, gamma, theta: 同上
        max_iter: 最大策略迭代次数
        use_exact_eval: 是否使用矩阵求逆法（True）或迭代法（False）

    返回:
        policy: shape (n_states,), 最优策略
        V: shape (n_states,), 最优价值函数
        v_history: 每次迭代的 V(start_state)，用于绘制收敛曲线
    """
    n_states = P.shape[0]

    # 初始化随机策略
    policy = np.random.randint(0, R.shape[1], size=n_states)
    v_history = []

    # ---- 在这里实现 ----
    for i in range(max_iter):
        if use_exact_eval == True:
            V = policy_evaluation_exact(P,R,policy,gamma)
        else:
            V,_ = policy_evaluation_iterative(P,R,policy,gamma,theta,max_iter)
        v_history.append(V[0])

        policy_new = policy_improvement(P,R,V,gamma)
        if np.array_equal(policy,policy_new):
            break
        policy = policy_new
    # ---- 代码结束 ----

    return policy, V, v_history


# ============================================================
# Part 5: 可视化
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


# ============================================================
# 主测试
# ============================================================

if __name__ == "__main__":
    # 创建环境
    print("=" * 60)
    print("测试 1: Frozen Lake 4x4 (确定性)")
    print("=" * 60)

    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)
    P, R = build_transition_matrices(env)

    print(f"状态数: {env.observation_space.n}")
    print(f"动作数: {env.action_space.n}")
    print(f"转移矩阵 P shape: {P.shape}")
    print(f"奖励矩阵 R shape: {R.shape}")

    # 测试策略评估
    print("\n" + "=" * 60)
    print("测试 2: 策略评估（迭代法 vs 矩阵法）")
    print("=" * 60)

    # 使用一个简单策略：总是向右或向下
    test_policy = np.array([2, 2, 2, 2,  # 第一行：右
                            1, 1, 1, 1,  # 第二行：下
                            2, 2, 2, 2,  # 第三行：右
                            1, 1, 1, 1]) # 第四行：下

    V_iter, num_iters = policy_evaluation_iterative(P, R, test_policy, gamma=0.99)
    V_exact = policy_evaluation_exact(P, R, test_policy, gamma=0.99)

    print(f"迭代法: {num_iters} 次迭代")
    print(f"V_iter[0]  = {V_iter[0]:.6f}")
    print(f"V_exact[0] = {V_exact[0]:.6f}")
    print(f"最大误差: {np.max(np.abs(V_iter - V_exact)):.2e}")

    # 完整策略迭代
    print("\n" + "=" * 60)
    print("测试 3: 策略迭代（确定性环境）")
    print("=" * 60)

    policy_opt, V_opt, v_history = policy_iteration(
        P, R, gamma=0.99, use_exact_eval=True
    )

    print_policy(policy_opt)
    print()
    print_values(V_opt)
    print(f"\nV*(start) = {V_opt[0]:.6f}")
    print(f"收敛历史: {[f'{v:.4f}' for v in v_history]}")

    # 随机环境测试
    print("\n" + "=" * 60)
    print("测试 4: 策略迭代（随机环境 is_slippery=True）")
    print("=" * 60)

    env_slippery = gym.make('FrozenLake-v1', is_slippery=True, render_mode=None)
    P_slip, R_slip = build_transition_matrices(env_slippery)

    policy_slip, V_slip, v_history_slip = policy_iteration(
        P_slip, R_slip, gamma=0.99, use_exact_eval=True
    )

    print_policy(policy_slip)
    print()
    print_values(V_slip)
    print(f"\nV*(start) = {V_slip[0]:.6f}")
    print(f"收敛历史: {[f'{v:.4f}' for v in v_history_slip]}")

    # 对比迭代法和矩阵法的效率
    print("\n" + "=" * 60)
    print("测试 5: 迭代法 vs 矩阵法效率对比")
    print("=" * 60)

    import time

    # 迭代法
    start = time.time()
    _, _, _ = policy_iteration(P, R, gamma=0.99, use_exact_eval=False)
    time_iter = time.time() - start

    # 矩阵法
    start = time.time()
    _, _, _ = policy_iteration(P, R, gamma=0.99, use_exact_eval=True)
    time_exact = time.time() - start

    print(f"迭代法耗时: {time_iter*1000:.2f} ms")
    print(f"矩阵法耗时: {time_exact*1000:.2f} ms")
    print(f"速度比: {time_iter/time_exact:.2f}x")

    print("\n" + "=" * 60)
    print("练习完成！")
    print("=" * 60)
    print("\n关键收获:")
    print("1. 策略迭代通过交替进行策略评估和策略改进找到最优策略")
    print("2. 策略改进定理保证每次迭代都不会变差")
    print("3. 通常 3-5 次迭代即可收敛，远快于暴力枚举")
    print("4. 矩阵法在小规模问题上更快，但大规模问题需要迭代法")
    print("5. 随机环境（is_slippery=True）的最优策略更保守")

    env.close()
    env_slippery.close()
