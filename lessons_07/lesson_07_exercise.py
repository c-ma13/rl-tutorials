"""
Lesson 07 练习：SARSA — On-Policy TD 控制

目标：
1. 实现 ε-greedy 动作选择
2. 实现 SARSA(0) 算法
3. 实现 SARSA(λ) 算法（资格迹）
4. 训练曲线和策略可视化
5. ε 衰减实验
6. 预留与 Q-Learning (Lesson 08) 的对比接口

环境：CliffWalking-v0 (4×12 网格)
┌────┬────┬─ ... ─┬────┬────┐
│    │    │       │    │    │  行0
├────┼────┼─ ... ─┼────┼────┤
│    │    │       │    │    │  行1
├────┼────┼─ ... ─┼────┼────┤
│    │    │       │    │    │  行2
├────┼────┼─ ... ─┼────┼────┤
│ S  │CCCC│  ...  │CCCC│ G  │  行3
└────┴────┴─ ... ─┴────┴────┘

S = 起点 (36), G = 终点 (47), C = 悬崖 (37-46)
动作: 0=上, 1=右, 2=下, 3=左
奖励: 每步 -1, 掉悬崖 -100 并回起点
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, List, Dict

# ============================================================
# Part 1: ε-greedy 动作选择
# ============================================================

def epsilon_greedy(Q: np.ndarray, state: int, n_actions: int, epsilon: float) -> int:
    """
    ε-greedy 动作选择。

    TODO 1: 实现 ε-greedy
    规则:
    - 概率 ε: 随机选择动作（均匀分布）
    - 概率 1-ε: 选择 Q 值最大的动作（贪心）
    - 如果多个动作 Q 值相同，随机选一个

    参数:
        Q: shape (n_states, n_actions)
        state: 当前状态
        n_actions: 动作数
        epsilon: 探索概率

    返回:
        action: 选择的动作
    """
    # ---- 在这里实现 ----
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        # 处理 Q 值相同的情况：随机选一个最大的
        q_values = Q[state]
        max_q = np.max(q_values)
        max_actions = np.where(q_values == max_q)[0]
        return np.random.choice(max_actions)
    # ---- 代码结束 ----


# ============================================================
# Part 2: SARSA(0) 算法
# ============================================================

def sarsa(
    env,
    num_episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon_start: float = 0.1,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995
) -> Tuple[np.ndarray, List[float]]:
    """
    SARSA(0) 算法。

    TODO 2: 实现 SARSA
    算法:
        对每个 episode：
            S ← 起始, A ← ε-greedy(Q, S)
            重复:
                执行 A, 观察 R, S'
                A' ← ε-greedy(Q, S')
                Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
                S ← S', A ← A'
            直到终止

    提示:
    - 终止时 Q(S_terminal, ·) = 0，所以 td_target = R
    - 用 (1 - terminated) 来处理终止状态
    - 每个 episode 记录总奖励

    参数:
        env: Gymnasium 环境
        num_episodes: 训练 episode 数
        alpha: 学习率
        gamma: 折扣因子
        epsilon_start/end/decay: ε 衰减参数

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
    for ep in range(num_episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, n_actions, epsilon)

        total_reward = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # 选下一个动作
            next_action = epsilon_greedy(Q, next_state, n_actions, epsilon)

            # SARSA 更新
            td_target = reward + gamma * Q[next_state, next_action] * (1 - int(terminated))
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            action = next_action

        reward_history.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    # ---- 代码结束 ----

    return Q, reward_history


# ============================================================
# Part 3: SARSA(λ) 算法（资格迹）
# ============================================================

def sarsa_lambda(
    env,
    lam: float = 0.8,
    num_episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon_start: float = 0.1,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995
) -> Tuple[np.ndarray, List[float]]:
    """
    SARSA(λ) 算法（累积迹）。

    TODO 3: 实现 SARSA(λ)
    算法:
        对每个 episode：
            e(s,a) = 0  ∀s,a
            S ← 起始, A ← ε-greedy(Q, S)
            重复:
                执行 A, 观察 R, S'
                A' ← ε-greedy(Q, S')
                δ ← R + γQ(S',A') - Q(S,A)
                e(S,A) ← e(S,A) + 1
                对所有 (s,a):
                    Q(s,a) ← Q(s,a) + α δ e(s,a)
                    e(s,a) ← γ λ e(s,a)
                S ← S', A ← A'

    提示:
    - 资格迹 e 的 shape 与 Q 相同
    - 每个 episode 开始时重置 e = 0
    - 终止状态处理同 SARSA(0)

    参数:
        env, lam(λ), num_episodes, alpha, gamma, epsilon参数

    返回:
        Q, reward_history
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    reward_history = []
    epsilon = epsilon_start

    # ---- 在这里实现 ----
    for ep in range(num_episodes):
        e = np.zeros((n_states, n_actions))  # 资格迹

        state, _ = env.reset()
        action = epsilon_greedy(Q, state, n_actions, epsilon)

        total_reward = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            next_action = epsilon_greedy(Q, next_state, n_actions, epsilon)

            # TD 误差
            td_target = reward + gamma * Q[next_state, next_action] * (1 - int(terminated))
            delta = td_target - Q[state, action]

            # 更新资格迹（累积迹）
            e[state, action] += 1

            # 更新所有 (s, a)
            Q += alpha * delta * e
            e *= gamma * lam

            state = next_state
            action = next_action

        reward_history.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    # ---- 代码结束 ----

    return Q, reward_history


# ============================================================
# Part 4: 策略提取与可视化
# ============================================================

def extract_policy(Q: np.ndarray) -> np.ndarray:
    """
    从 Q 函数提取贪心策略。

    TODO 4: 实现策略提取
    """
    # ---- 在这里实现 ----
    return np.argmax(Q, axis=1)
    # ---- 代码结束 ----


def print_policy(policy: np.ndarray, shape: Tuple[int, int] = (4, 12)):
    """打印策略（箭头表示）"""
    action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    rows, cols = shape
    cliff_states = set(range(37, 47))  # 悬崖状态

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


def print_q_values(Q: np.ndarray, state: int):
    """打印某个状态的 Q 值"""
    action_names = {0: '上', 1: '右', 2: '下', 3: '左'}
    print(f"  状态 {state} 的 Q 值:")
    for a in range(Q.shape[1]):
        print(f"    {action_names[a]}: {Q[state, a]:.2f}")


def moving_average(data: List[float], window: int = 50) -> List[float]:
    """计算移动平均"""
    result = []
    cumsum = np.cumsum(data)
    for i in range(len(data)):
        if i < window:
            result.append(cumsum[i] / (i + 1))
        else:
            result.append((cumsum[i] - cumsum[i - window]) / window)
    return result


# ============================================================
# Part 5: ε 衰减对比实验
# ============================================================

def epsilon_experiment(
    env,
    num_episodes: int = 500,
    num_runs: int = 20,
    alpha: float = 0.5,
    gamma: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    对比不同 ε 策略的效果。

    TODO 5: 实现 ε 对比实验
    测试:
    - 固定 ε = 0.1
    - 固定 ε = 0.01
    - 衰减 ε: 0.1 → 0.01
    多次运行取平均

    返回:
        results: dict, key=策略名, value=平均奖励曲线
    """
    configs = {
        'ε=0.1 (固定)': (0.1, 0.1, 1.0),
        'ε=0.01 (固定)': (0.01, 0.01, 1.0),
        'ε: 0.1→0.01 (衰减)': (0.1, 0.01, 0.995),
    }

    results = {}

    # ---- 在这里实现 ----
    for name, (eps_start, eps_end, eps_decay) in configs.items():
        all_rewards = np.zeros(num_episodes)

        for run in range(num_runs):
            np.random.seed(run * 42)
            _, rewards = sarsa(
                env,
                num_episodes=num_episodes,
                alpha=alpha,
                gamma=gamma,
                epsilon_start=eps_start,
                epsilon_end=eps_end,
                epsilon_decay=eps_decay
            )
            all_rewards += np.array(rewards)

        results[name] = all_rewards / num_runs
    # ---- 代码结束 ----

    return results


# ============================================================
# Part 6: SARSA(0) vs SARSA(λ) 对比
# ============================================================

def lambda_experiment(
    env,
    num_episodes: int = 500,
    num_runs: int = 20,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1
) -> Dict[str, np.ndarray]:
    """
    对比不同 λ 值的效果。

    TODO 6: 实现 λ 对比实验

    返回:
        results: dict, key=λ值描述, value=平均奖励曲线
    """
    lambdas = [0.0, 0.4, 0.8, 0.95]
    results = {}

    # ---- 在这里实现 ----
    for lam in lambdas:
        all_rewards = np.zeros(num_episodes)

        for run in range(num_runs):
            np.random.seed(run * 42)
            _, rewards = sarsa_lambda(
                env,
                lam=lam,
                num_episodes=num_episodes,
                alpha=alpha,
                gamma=gamma,
                epsilon_start=epsilon,
                epsilon_end=epsilon,
                epsilon_decay=1.0  # 固定 ε
            )
            all_rewards += np.array(rewards)

        results[f'λ={lam}'] = all_rewards / num_runs
    # ---- 代码结束 ----

    return results


# ============================================================
# 主测试
# ============================================================

if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')

    print("CliffWalking 环境:")
    print(f"  状态数: {env.observation_space.n}")
    print(f"  动作数: {env.action_space.n}")
    print(f"  起点: 36, 终点: 47")
    print(f"  悬崖: 37-46")

    # --- 测试 1: ε-greedy ---
    print("\n" + "=" * 60)
    print("测试 1: ε-greedy 动作选择")
    print("=" * 60)

    Q_test = np.zeros((48, 4))
    Q_test[0] = [1.0, 2.0, 0.5, 0.3]  # 动作 1(右) 是最优

    action_counts = np.zeros(4)
    for _ in range(10000):
        a = epsilon_greedy(Q_test, 0, 4, epsilon=0.1)
        action_counts[a] += 1

    print(f"Q[0] = {Q_test[0]}")
    print(f"ε=0.1 时动作分布 (10000 次):")
    for a in range(4):
        print(f"  动作 {a}: {action_counts[a]/10000*100:.1f}%")
    print(f"  期望: 动作1 ≈ {(1-0.1+0.1/4)*100:.1f}%, 其他 ≈ {0.1/4*100:.1f}%")

    # --- 测试 2: SARSA(0) ---
    print("\n" + "=" * 60)
    print("测试 2: SARSA(0) 训练")
    print("=" * 60)

    Q_sarsa, rewards_sarsa = sarsa(
        env, num_episodes=500, alpha=0.5, gamma=1.0,
        epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=0.995
    )

    policy_sarsa = extract_policy(Q_sarsa)
    print_policy(policy_sarsa)

    avg_rewards = moving_average(rewards_sarsa, window=50)
    print(f"\n训练曲线（移动平均, 窗口=50）:")
    for ep in [0, 50, 100, 200, 300, 400, 499]:
        print(f"  Episode {ep:>4d}: 平均奖励 = {avg_rewards[ep]:.1f}")

    # 关键状态的 Q 值
    print("\n关键状态 Q 值:")
    print_q_values(Q_sarsa, 36)  # 起点
    print_q_values(Q_sarsa, 24)  # 起点上方
    print_q_values(Q_sarsa, 35)  # 终点左上方

    # --- 测试 3: SARSA(λ) ---
    print("\n" + "=" * 60)
    print("测试 3: SARSA(λ=0.8) 训练")
    print("=" * 60)

    Q_sarsa_lam, rewards_sarsa_lam = sarsa_lambda(
        env, lam=0.8, num_episodes=500, alpha=0.5, gamma=1.0,
        epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=0.995
    )

    policy_sarsa_lam = extract_policy(Q_sarsa_lam)
    print_policy(policy_sarsa_lam)

    avg_lam = moving_average(rewards_sarsa_lam, window=50)
    print(f"\nSARSA(0) vs SARSA(λ=0.8) 最后 50 episode 平均:")
    print(f"  SARSA(0):    {np.mean(rewards_sarsa[-50:]):.1f}")
    print(f"  SARSA(λ=0.8): {np.mean(rewards_sarsa_lam[-50:]):.1f}")

    # --- 测试 4: ε 衰减对比 ---
    print("\n" + "=" * 60)
    print("测试 4: ε 衰减策略对比")
    print("=" * 60)

    print("正在运行对比实验（20 次取平均）...")
    eps_results = epsilon_experiment(env, num_episodes=500, num_runs=20)

    print(f"\n{'ε 策略':<25} {'前 50 ep':<12} {'后 50 ep':<12}")
    print("-" * 50)
    for name, rewards in eps_results.items():
        early = np.mean(rewards[:50])
        late = np.mean(rewards[-50:])
        print(f"{name:<25} {early:<12.1f} {late:<12.1f}")

    # --- 测试 5: λ 对比 ---
    print("\n" + "=" * 60)
    print("测试 5: SARSA(λ) 不同 λ 值对比")
    print("=" * 60)

    print("正在运行对比实验（20 次取平均）...")
    lam_results = lambda_experiment(env, num_episodes=500, num_runs=20)

    print(f"\n{'λ 值':<15} {'前 50 ep':<12} {'后 50 ep':<12}")
    print("-" * 40)
    for name, rewards in lam_results.items():
        early = np.mean(rewards[:50])
        late = np.mean(rewards[-50:])
        print(f"{name:<15} {early:<12.1f} {late:<12.1f}")

    # --- 测试 6: 策略分析 ---
    print("\n" + "=" * 60)
    print("测试 6: SARSA 策略分析")
    print("=" * 60)

    print("\nSARSA 学到的策略特点:")
    print("- SARSA 因为是 On-Policy，考虑了 ε 探索可能掉下悬崖的风险")
    print("- 所以倾向于走远离悬崖的安全路径（从上方绕行）")
    print("- 这和 Q-Learning (Lesson 08) 学到的贴崖最短路径不同")

    # 统计走安全路径 vs 贴崖路径
    safe_count = 0
    cliff_count = 0
    n_eval = 1000

    for _ in range(n_eval):
        state, _ = env.reset()
        done = False
        visited = set()
        while not done:
            action = np.argmax(Q_sarsa[state])  # 贪心评估
            visited.add(state)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        # 检查是否走了悬崖旁边（行2，即状态24-35）
        if any(s in visited for s in range(24, 36)):
            safe_count += 1  # 经过行2 = 安全路径
        # 检查是否经过行3的起点旁（除了起点和终点）
        if any(s in visited for s in range(37, 47)):
            cliff_count += 1

    print(f"\n贪心策略评估 ({n_eval} 局):")
    print(f"  经过安全区域（行2）: {safe_count/n_eval*100:.1f}%")
    print(f"  掉入悬崖: {cliff_count/n_eval*100:.1f}%")

    print("\n" + "=" * 60)
    print("练习完成！")
    print("=" * 60)
    print("\n关键收获:")
    print("1. SARSA 是 On-Policy TD 控制：用 ε-greedy 选动作并学习该策略的 Q 值")
    print("2. 名字来源: (S, A, R, S', A') 五元组")
    print("3. On-Policy 特点: 学到的策略考虑探索风险，偏好安全路径")
    print("4. SARSA(λ) 通过资格迹加速信用分配，通常收敛更快")
    print("5. ε 衰减在早期促进探索，后期提升策略质量")
    print("6. 下一课 Q-Learning 将展示 Off-Policy 的不同行为")

    env.close()
