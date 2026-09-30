"""
Lesson 05 练习：Monte Carlo 方法

目标：
1. 实现 MC 策略评估（首次访问）
2. 实现回报计算（从后往前递推）
3. 实现 On-Policy MC 控制（ε-greedy）
4. 在 Blackjack 环境上训练智能体
5. 对比不同 ε 衰减策略的效果

Blackjack 环境：
状态: (player_sum, dealer_showing, usable_ace)
     player_sum: 12-21
     dealer_showing: 1-10 (1=A)
     usable_ace: True/False
动作: 0=Stick(停牌), 1=Hit(叫牌)
奖励: +1(赢), 0(平), -1(输)
γ = 1.0（episode 自然终止，无需折扣）
"""

import numpy as np
import gymnasium as gym
from collections import defaultdict
from typing import Tuple, List, Dict

# ============================================================
# Part 1: 回报计算
# ============================================================

def compute_returns(rewards: List[float], gamma: float = 1.0) -> List[float]:
    """
    从后往前计算每个时刻的折扣回报。

    TODO 1: 实现回报递推计算
    公式: G_t = R_{t+1} + γ G_{t+1}
    从 G_{T-1} = R_T 开始，往前推

    参数:
        rewards: [R_1, R_2, ..., R_T], 长度 T
        gamma: 折扣因子

    返回:
        returns: [G_0, G_1, ..., G_{T-1}], 长度 T
    """
    T = len(rewards)
    returns = [0.0] * T

    # ---- 在这里实现 ----
    returns[T - 1] = rewards[T - 1]
    for t in range(T - 2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    # ---- 代码结束 ----

    return returns


# ============================================================
# Part 2: 生成 Episode
# ============================================================

def generate_episode(env, policy: Dict = None, epsilon: float = 0.0) -> Tuple[List, List, List]:
    """
    使用给定策略（或 ε-greedy 策略）生成一个 episode。

    TODO 2: 实现 episode 生成
    提示:
    - 如果 policy 为 None，使用完全随机策略
    - 如果 policy 不为 None 且 epsilon > 0，使用 ε-greedy
    - 记录 (states, actions, rewards)

    参数:
        env: Gymnasium 环境
        policy: dict, policy[state] = action（确定性策略部分）
        epsilon: 探索概率

    返回:
        states: [S_0, S_1, ...]
        actions: [A_0, A_1, ...]
        rewards: [R_1, R_2, ...]  (注意: R_1 是 A_0 后得到的)
    """
    states = []
    actions = []
    rewards = []

    # ---- 在这里实现 ----
    obs, info = env.reset()
    done = False

    while not done:
        state = obs
        states.append(state)

        # 选择动作
        if policy is None:
            action = env.action_space.sample()  # 完全随机
        elif np.random.random() < epsilon:
            action = env.action_space.sample()  # 探索
        else:
            action = policy.get(state, env.action_space.sample())  # 利用

        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    # ---- 代码结束 ----

    return states, actions, rewards


# ============================================================
# Part 3: MC 策略评估（首次访问）
# ============================================================

def mc_prediction_first_visit(
    env,
    policy: Dict,
    num_episodes: int = 100000,
    gamma: float = 1.0
) -> Dict:
    """
    首次访问 MC 策略评估。

    TODO 3: 实现首次访问 MC 评估
    算法:
        对每个 episode：
            生成轨迹
            计算回报
            对每个首次出现的状态：
                更新 V(s) = mean(所有该状态的回报)

    参数:
        env: 环境
        policy: dict, policy[state] = action
        num_episodes: 采样的 episode 数量
        gamma: 折扣因子

    返回:
        V: dict, V[state] = 状态价值估计
    """
    V = defaultdict(float)
    returns_count = defaultdict(int)
    returns_sum = defaultdict(float)

    # ---- 在这里实现 ----
    for episode_i in range(num_episodes):
        states, actions, rewards = generate_episode(env, policy, epsilon=0.0)
        G_list = compute_returns(rewards, gamma)

        visited = set()
        for t, state in enumerate(states):
            if state not in visited:
                visited.add(state)
                returns_sum[state] += G_list[t]
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]
    # ---- 代码结束 ----

    return dict(V)


# ============================================================
# Part 4: MC 策略评估 Q 函数（首次访问）
# ============================================================

def mc_prediction_q(
    env,
    policy: Dict,
    num_episodes: int = 100000,
    gamma: float = 1.0,
    epsilon: float = 0.1
) -> Dict:
    """
    首次访问 MC 评估 Q 函数。

    TODO 4: 实现 Q 函数的 MC 评估
    与 V 的区别：记录的是 (state, action) 对而不仅仅是 state

    参数:
        env: 环境
        policy: dict, policy[state] = action
        num_episodes: 采样数量
        gamma: 折扣因子
        epsilon: ε-greedy 探索率（保证所有 (s,a) 被访问）

    返回:
        Q: dict, Q[(state, action)] = 动作价值估计
    """
    Q = defaultdict(float)
    returns_count = defaultdict(int)
    returns_sum = defaultdict(float)

    # ---- 在这里实现 ----
    for episode_i in range(num_episodes):
        states, actions, rewards = generate_episode(env, policy, epsilon=epsilon)
        G_list = compute_returns(rewards, gamma)

        visited = set()
        for t in range(len(states)):
            sa = (states[t], actions[t])
            if sa not in visited:
                visited.add(sa)
                returns_sum[sa] += G_list[t]
                returns_count[sa] += 1
                Q[sa] = returns_sum[sa] / returns_count[sa]
    # ---- 代码结束 ----

    return dict(Q)


# ============================================================
# Part 5: On-Policy MC 控制（ε-greedy）
# ============================================================

def mc_control_epsilon_greedy(
    env,
    num_episodes: int = 500000,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.99999
) -> Tuple[Dict, Dict, List[float]]:
    """
    On-Policy 首次访问 MC 控制（ε-greedy）。

    TODO 5: 实现 MC 控制
    算法:
        初始化 Q(s,a) = 0, N(s,a) = 0
        对每个 episode：
            1. 用 ε-greedy(Q) 生成 episode
            2. 计算回报
            3. 对每个首次访问的 (s,a)：
                N(s,a) += 1
                Q(s,a) += (1/N) [G - Q(s,a)]
            4. ε 衰减

    参数:
        env: 环境
        num_episodes: 训练的 episode 数
        gamma: 折扣因子
        epsilon_start: 初始 ε
        epsilon_end: 最小 ε
        epsilon_decay: ε 衰减系数

    返回:
        Q: dict, Q[(state, action)] = 动作价值
        policy: dict, policy[state] = 最优动作
        reward_history: list, 每个 episode 的总奖励
    """
    Q = defaultdict(float)
    N = defaultdict(int)
    policy = {}
    reward_history = []
    epsilon = epsilon_start

    # ---- 在这里实现 ----
    for episode_i in range(num_episodes):
        # 1. 生成 episode
        states, actions, rewards = generate_episode(env, policy, epsilon=epsilon)
        reward_history.append(sum(rewards))

        # 2. 计算回报
        G_list = compute_returns(rewards, gamma)

        # 3. 更新 Q
        visited = set()
        for t in range(len(states)):
            sa = (states[t], actions[t])
            if sa not in visited:
                visited.add(sa)
                N[sa] += 1
                Q[sa] += (1.0 / N[sa]) * (G_list[t] - Q[sa])

        # 4. 更新策略（贪心部分）
        for t in range(len(states)):
            state = states[t]
            q_values = [Q.get((state, a), 0.0) for a in range(env.action_space.n)]
            policy[state] = int(np.argmax(q_values))

        # 5. 衰减 ε
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    # ---- 代码结束 ----

    return dict(Q), policy, reward_history


# ============================================================
# Part 6: 辅助函数
# ============================================================

def simple_policy(obs) -> int:
    """
    简单的 Blackjack 策略：点数 < 20 就叫牌，否则停牌。
    用于 MC 策略评估的测试。
    """
    player_sum, _, _ = obs
    return 1 if player_sum < 20 else 0


def create_policy_dict(policy_func, env) -> Dict:
    """将策略函数转换为 dict 形式"""
    policy = {}
    for player_sum in range(12, 22):
        for dealer_showing in range(1, 11):
            for usable_ace in [True, False]:
                state = (player_sum, dealer_showing, usable_ace)
                policy[state] = policy_func(state)
    return policy


def moving_average(data: List[float], window: int = 1000) -> List[float]:
    """计算移动平均"""
    if len(data) < window:
        return [np.mean(data[:i+1]) for i in range(len(data))]

    result = []
    cumsum = np.cumsum(data)
    for i in range(len(data)):
        if i < window:
            result.append(cumsum[i] / (i + 1))
        else:
            result.append((cumsum[i] - cumsum[i - window]) / window)
    return result


def print_policy_table(policy: Dict, usable_ace: bool = False):
    """
    打印 Blackjack 策略表。
    行: player_sum (21 → 12)
    列: dealer_showing (1=A, 2, ..., 10)
    """
    action_symbols = {0: 'S', 1: 'H'}  # S=Stick, H=Hit
    ace_str = "有可用 A" if usable_ace else "无可用 A"
    print(f"\n策略表（{ace_str}）:")
    print(f"{'':>4}", end="")
    for d in range(1, 11):
        label = 'A' if d == 1 else str(d)
        print(f" {label:>2}", end="")
    print()

    for p in range(21, 11, -1):
        print(f"{p:>3}:", end="")
        for d in range(1, 11):
            state = (p, d, usable_ace)
            a = policy.get(state, 0)
            print(f"  {action_symbols[a]}", end="")
        print()


def print_v_table(V: Dict, usable_ace: bool = False):
    """打印状态价值表"""
    ace_str = "有可用 A" if usable_ace else "无可用 A"
    print(f"\n价值函数（{ace_str}）:")
    print(f"{'':>4}", end="")
    for d in range(1, 11):
        label = 'A' if d == 1 else str(d)
        print(f" {label:>6}", end="")
    print()

    for p in range(21, 11, -1):
        print(f"{p:>3}:", end="")
        for d in range(1, 11):
            state = (p, d, usable_ace)
            v = V.get(state, 0.0)
            print(f" {v:>6.3f}", end="")
        print()


# ============================================================
# 主测试
# ============================================================

if __name__ == "__main__":
    env = gym.make('Blackjack-v1', sab=True)

    # --- 测试 1: 回报计算 ---
    print("=" * 60)
    print("测试 1: 回报计算")
    print("=" * 60)

    test_rewards = [0, 0, 0, 1]  # 最后一步得到 +1
    returns_no_discount = compute_returns(test_rewards, gamma=1.0)
    returns_discounted = compute_returns(test_rewards, gamma=0.9)

    print(f"奖励序列: {test_rewards}")
    print(f"γ=1.0 回报: {returns_no_discount}")
    print(f"γ=0.9 回报: {returns_discounted}")
    print(f"验证 G_0 (γ=0.9): 0 + 0.9*0 + 0.81*0 + 0.729*1 = {0.9**3:.3f}")

    # --- 测试 2: 生成 Episode ---
    print("\n" + "=" * 60)
    print("测试 2: 生成 Episode 示例")
    print("=" * 60)

    # 用简单策略生成几个 episode
    simple_dict = create_policy_dict(simple_policy, env)

    for i in range(5):
        states, actions, rewards = generate_episode(env, simple_dict, epsilon=0.0)
        action_names = ['Stick' if a == 0 else 'Hit' for a in actions]
        print(f"Episode {i}: states={states}, actions={action_names}, rewards={rewards}")

    # --- 测试 3: MC 策略评估 ---
    print("\n" + "=" * 60)
    print("测试 3: MC 策略评估（简单策略: <20 叫牌）")
    print("=" * 60)

    print("正在评估策略（100,000 episodes）...")
    V = mc_prediction_first_visit(env, simple_dict, num_episodes=100000, gamma=1.0)

    print(f"评估的状态数: {len(V)}")
    print_v_table(V, usable_ace=False)
    print_v_table(V, usable_ace=True)

    # 几个关键状态的分析
    print("\n关键状态价值:")
    print(f"  V(21, 1, False) = {V.get((21, 1, False), 0):.3f}  (21点 vs 庄家A)")
    print(f"  V(20, 6, False) = {V.get((20, 6, False), 0):.3f}  (20点 vs 庄家6)")
    print(f"  V(12, 10, False) = {V.get((12, 10, False), 0):.3f}  (12点 vs 庄家10)")

    # --- 测试 4: MC 控制 ---
    print("\n" + "=" * 60)
    print("测试 4: MC 控制（ε-greedy）")
    print("=" * 60)

    print("正在训练（500,000 episodes）...")
    Q, learned_policy, reward_history = mc_control_epsilon_greedy(
        env,
        num_episodes=500000,
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99999
    )

    # 打印学到的策略
    print("\n学到的策略:")
    print_policy_table(learned_policy, usable_ace=False)
    print_policy_table(learned_policy, usable_ace=True)

    # 学习曲线
    avg_rewards = moving_average(reward_history, window=10000)
    print(f"\n学习曲线（移动平均, 窗口=10000）:")
    checkpoints = [0, 50000, 100000, 200000, 300000, 400000, 499999]
    for cp in checkpoints:
        if cp < len(avg_rewards):
            print(f"  Episode {cp:>7d}: 平均奖励 = {avg_rewards[cp]:.4f}")

    # --- 测试 5: 评估学到的策略 ---
    print("\n" + "=" * 60)
    print("测试 5: 评估学到的策略")
    print("=" * 60)

    wins = 0
    losses = 0
    draws = 0
    n_eval = 100000

    for _ in range(n_eval):
        states, actions, rewards = generate_episode(env, learned_policy, epsilon=0.0)
        total_reward = sum(rewards)
        if total_reward > 0:
            wins += 1
        elif total_reward < 0:
            losses += 1
        else:
            draws += 1

    print(f"评估结果（{n_eval} 局）:")
    print(f"  胜率: {wins/n_eval*100:.1f}%")
    print(f"  负率: {losses/n_eval*100:.1f}%")
    print(f"  平局: {draws/n_eval*100:.1f}%")
    print(f"  净收益率: {(wins-losses)/n_eval*100:.1f}%")

    # 与简单策略对比
    wins_simple = 0
    losses_simple = 0

    for _ in range(n_eval):
        states, actions, rewards = generate_episode(env, simple_dict, epsilon=0.0)
        total_reward = sum(rewards)
        if total_reward > 0:
            wins_simple += 1
        elif total_reward < 0:
            losses_simple += 1

    print(f"\n简单策略（<20 叫牌）对比:")
    print(f"  胜率: {wins_simple/n_eval*100:.1f}%")
    print(f"  净收益率: {(wins_simple - (n_eval - wins_simple - (n_eval - wins_simple - losses_simple)))/n_eval*100:.1f}%")

    # --- 测试 6: 对比已知最优策略 ---
    print("\n" + "=" * 60)
    print("测试 6: 与 Blackjack 近似最优策略对比")
    print("=" * 60)

    # 近似最优策略（简化版）
    # 无可用 A：player_sum >= 17 停牌，12-16 看庄家牌
    # 有可用 A：player_sum >= 19 停牌
    print("学到的策略 vs 常见策略规则:")
    print("  无可用 A, 庄家明牌 2-6 (庄家弱):")
    for p in range(12, 18):
        state = (p, 5, False)
        a = learned_policy.get(state, -1)
        action_name = 'Stick' if a == 0 else 'Hit'
        print(f"    player_sum={p}: {action_name}")

    print("  无可用 A, 庄家明牌 10 (庄家强):")
    for p in range(12, 18):
        state = (p, 10, False)
        a = learned_policy.get(state, -1)
        action_name = 'Stick' if a == 0 else 'Hit'
        print(f"    player_sum={p}: {action_name}")

    print("\n" + "=" * 60)
    print("练习完成！")
    print("=" * 60)
    print("\n关键收获:")
    print("1. MC 方法不需要环境模型，只需要与环境交互产生 episode")
    print("2. 首次访问 MC 是 V(s) 的无偏估计，大数定律保证收敛")
    print("3. Model-free 控制需要用 Q(s,a) 而不是 V(s)，因为没有 P 无法做 argmax")
    print("4. ε-greedy 平衡探索与利用，ε 衰减使策略逐渐收敛到最优")
    print("5. MC 的高方差导致收敛速度较慢，需要大量 episode")
    print("6. Blackjack 本身是庄家有利的游戏，即使最优策略胜率也 < 50%")

    env.close()
