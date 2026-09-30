"""
Lesson 02 练习：Bellman 算子、迭代求解与策略枚举

目标：
1. 实现 V 到 Q 的转换
2. 实现迭代策略评估（Iterative Policy Evaluation）
3. 在小规模 MDP 上枚举所有策略，找到最优策略
4. 验证收缩映射性质

TinyGridWorld 设定（2x2 网格）：
+-----+-----+
| S=0 |  1  |
+-----+-----+
|  2  | G=3 |
+-----+-----+

S = 起点 (0,0), 状态索引 0
G = 终点 (1,1), 状态索引 3, reward = +1
其他移动: reward = -0.1
动作: 0=右, 1=下
撞墙: 留在原地
γ = 0.9

非终止状态 3 个 {0, 1, 2}，每个有 2 种动作选择，共 2^3 = 8 种确定性策略。
"""

import numpy as np
from itertools import product

# ============================================================
# Part 1: TinyGridWorld 环境（2x2 网格）
# ============================================================

class TinyGridWorld:
    """2x2 GridWorld 环境，用于策略枚举"""

    def __init__(self):
        self.rows = 2
        self.cols = 2
        self.n_states = 4
        self.n_actions = 2  # 0=右, 1=下

        # 特殊状态
        self.start = 0
        self.goal = 3
        self.terminal_states = {self.goal}

        # 奖励
        self.goal_reward = 1.0
        self.step_reward = -0.1

        # 折扣因子
        self.gamma = 0.9

        # 动作定义: 右=0, 下=1
        self.action_names = {0: '→', 1: '↓'}

    def idx_to_pos(self, idx):
        """状态索引转 (row, col)"""
        return idx // self.cols, idx % self.cols

    def pos_to_idx(self, row, col):
        """(row, col) 转状态索引"""
        return row * self.cols + col

    def get_next_state(self, state, action):
        """
        返回下一个状态索引。

        TODO 1: 实现状态转移
        提示:
        - action=0 (右): col+1
        - action=1 (下): row+1
        - 检查边界，越界则留在原地
        - 终止状态转移到自己
        """
        if state in self.terminal_states:
            return state

        row, col = self.idx_to_pos(state)

        # ---- 在这里实现 ----
        if action == 0:
            col = min(1, col+1)
        else:
            row = min(1, row+1)
        return self.pos_to_idx(row, col)
        # ---- 代码结束 ----

    def get_reward(self, next_state):
        """返回到达 next_state 的奖励"""
        if next_state == self.goal:
            return self.goal_reward
        return self.step_reward

    def build_transition_matrix(self, policy):
        """
        构建策略 π 下的转移矩阵 P^π 和奖励向量 r^π

        参数:
            policy: shape (n_states,), policy[s] = 动作索引

        返回:
            P: shape (n_states, n_states), P[s, s'] = P(s'|s, π(s))
            r: shape (n_states,), r[s] = 期望即时奖励
        """
        P = np.zeros((self.n_states, self.n_states))
        r = np.zeros(self.n_states)

        for s in range(self.n_states):
            if s in self.terminal_states:
                P[s, s] = 1.0
                r[s] = 0.0
            else:
                a = policy[s]
                s_next = self.get_next_state(s, a)
                P[s, s_next] = 1.0
                r[s] = self.get_reward(s_next)

        return P, r


# ============================================================
# Part 2: V 到 Q 的转换
# ============================================================

def compute_q_from_v(env, V, policy=None):
    """
    从 V 计算 Q 函数。

    TODO 2: 实现 V → Q 转换
    公式: Q(s, a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
    对于确定性转移: Q(s, a) = R(s, a, s') + γ V(s')

    参数:
        env: 环境对象
        V: shape (n_states,), 状态价值函数
        policy: 可选，用于验证 V = Σ_a π(a|s) Q(s,a)

    返回:
        Q: shape (n_states, n_actions)
    """
    Q = np.zeros((env.n_states, env.n_actions))

    # ---- 在这里实现 ----
    for i in range(env.n_states):
        if i not in env.terminal_states:
            for j in range(env.n_actions):
                next=env.get_next_state(i,j)
                Q[i,j] = env.get_reward(next) + env.gamma * V[next]
    return Q
    # ---- 代码结束 ----


# ============================================================
# Part 3: 迭代策略评估
# ============================================================

def iterative_policy_evaluation(env, policy, theta=1e-6, max_iter=1000):
    """
    迭代策略评估：反复应用 Bellman 期望算子 T^π

    TODO 3: 实现迭代策略评估
    算法:
        v_0 = 0
        重复:
            v_{k+1}(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ v_k(s')]
        直到 ||v_{k+1} - v_k||_∞ < θ

    参数:
        env: 环境对象
        policy: shape (n_states,), policy[s] = 动作索引
        theta: 收敛阈值
        max_iter: 最大迭代次数

    返回:
        V: shape (n_states,), 收敛后的价值函数
        num_iters: 实际迭代次数
    """
    V = np.zeros(env.n_states)

    # ---- 在这里实现 ----
    for iter in range(max_iter):
        Vnew = np.zeros(env.n_states)
        for i in range(env.n_states):
            if i not in env.terminal_states:
                next = env.get_next_state(i,policy[i])
                Vnew[i] = env.get_reward(next) + env.gamma * V[next]
        if np.max(np.abs(Vnew-V))<theta:
            # print('iterations to converge: ',iter)
            break
        else:
            V = Vnew
    return V, iter
    # ---- 代码结束 ----


def solve_bellman_exact(env, policy):
    """精确求解（矩阵法），用于验证迭代法"""
    P, r = env.build_transition_matrix(policy)
    V = np.linalg.solve(np.eye(env.n_states) - env.gamma * P, r)
    return V


# ============================================================
# Part 4: 策略枚举
# ============================================================

def enumerate_all_policies(env):
    """
    枚举所有确定性策略。

    TODO 4: 实现策略枚举
    提示:
    - 非终止状态: {0, 1, 2}
    - 每个状态有 2 种动作选择
    - 使用 itertools.product 生成所有组合
    - 终止状态的动作可以任意设置（不影响价值）

    返回:
        policies: list of arrays, 每个 array shape (n_states,)
    """
    non_terminal = [s for s in range(env.n_states) if s not in env.terminal_states]

    policies = []

    # ---- 在这里实现 ----
    for actions in product(range(env.n_actions),repeat=len(non_terminal)):
        policy = np.zeros(env.n_states)
        for i,s in enumerate(non_terminal):
            policy[s] = actions[i]
        for s in env.terminal_states:
            policy[s] = 0
        policies.append(policy)
    # ---- 代码结束 ----

    return policies


def find_optimal_policy(env, policies):
    """
    在所有策略中找到最优策略。

    TODO 5: 实现最优策略搜索
    提示:
    - 对每个策略求解 V^π
    - 比较 V^π(start_state)，选最大的

    返回:
        best_policy: shape (n_states,)
        best_V: shape (n_states,)
        all_values: list of V^π(start) for all policies
    """
    all_values = []
    best_value = -np.inf
    best_policy = None
    best_V = None

    # ---- 在这里实现 ----
    for policy in policies:
        V,_ = iterative_policy_evaluation(env,policy=policy)
        all_values.append(V[env.start])
        if V[env.start]>best_value:
            best_value = V[env.start]
            best_policy = policy
            best_V = V
    # ---- 代码结束 ----

    return best_policy, best_V, all_values


# ============================================================
# Part 5: 验证收缩映射性质
# ============================================================

def verify_contraction(env, policy, v1, v2):
    """
    验证 Bellman 算子的收缩性质。

    TODO 6: 验证 ||T^π v1 - T^π v2||_∞ ≤ γ ||v1 - v2||_∞

    参数:
        env: 环境对象
        policy: shape (n_states,)
        v1, v2: shape (n_states,), 两个任意价值函数

    返回:
        lhs: ||T^π v1 - T^π v2||_∞
        rhs: γ ||v1 - v2||_∞
        is_contraction: lhs ≤ rhs
    """
    # ---- 在这里实现 ----
    Tv1 = np.zeros(env.n_states)
    Tv2 = np.zeros(env.n_states)

    for s in range(env.n_states):
        if s in env.terminal_states:
            Tv1[s] = 0.0
            Tv2[s] = 0.0
        else:
            a = policy[s]
            s_next = env.get_next_state(s, a)
            reward = env.get_reward(s_next)
            Tv1[s] = reward + env.gamma * v1[s_next]
            Tv2[s] = reward + env.gamma * v2[s_next]

    lhs = np.max(np.abs(Tv1 - Tv2))
    rhs = env.gamma * np.max(np.abs(v1 - v2))
    is_contraction = lhs <= rhs + 1e-10  # 加小量避免浮点误差
    # ---- 代码结束 ----

    return lhs, rhs, is_contraction


# ============================================================
# 辅助函数
# ============================================================

def print_policy(env, policy):
    """打印策略"""
    print("策略:")
    for row in range(env.rows):
        line = ""
        for col in range(env.cols):
            s = env.pos_to_idx(row, col)
            if s in env.terminal_states:
                line += "  G  "
            else:
                line += f"  {env.action_names[policy[s]]}  "
        print(line)


def print_values(env, V):
    """打印价值函数"""
    print("价值函数:")
    for row in range(env.rows):
        line = ""
        for col in range(env.cols):
            s = env.pos_to_idx(row, col)
            line += f" {V[s]:5.2f} "
        print(line)


# ============================================================
# 主测试
# ============================================================

if __name__ == "__main__":
    env = TinyGridWorld()

    # --- 测试 1: 验证 V → Q 转换 ---
    print("=" * 60)
    print("测试 1: V → Q 转换")
    print("=" * 60)

    # 使用一个简单策略: 状态 0→右, 1→下, 2→右
    test_policy = np.array([0, 1, 0, 0])  # 终止状态动作任意
    V_test = solve_bellman_exact(env, test_policy)
    Q_test = compute_q_from_v(env, V_test, test_policy)

    print(f"策略: {test_policy[:3]}")  # 只显示非终止状态
    print(f"V = {V_test}")
    print(f"Q =")
    print(Q_test)

    # 验证 V(s) = Σ_a π(a|s) Q(s,a)
    print("\n验证 V(s) = Q(s, π(s)):")
    for s in range(env.n_states):
        if s not in env.terminal_states:
            a = test_policy[s]
            print(f"  状态 {s}: V={V_test[s]:.4f}, Q(s,{a})={Q_test[s,a]:.4f}, 差异={abs(V_test[s] - Q_test[s,a]):.2e}")

    # --- 测试 2: 迭代策略评估 vs 精确解 ---
    print("\n" + "=" * 60)
    print("测试 2: 迭代策略评估 vs 精确解")
    print("=" * 60)

    V_iter, num_iters = iterative_policy_evaluation(env, test_policy, theta=1e-6)
    V_exact = solve_bellman_exact(env, test_policy)

    print(f"迭代次数: {num_iters}")
    print(f"V_iter  = {V_iter}")
    print(f"V_exact = {V_exact}")
    print(f"最大误差: {np.max(np.abs(V_iter - V_exact)):.2e}")

    # --- 测试 3: 策略枚举 ---
    print("\n" + "=" * 60)
    print("测试 3: 策略枚举找最优策略")
    print("=" * 60)

    all_policies = enumerate_all_policies(env)
    print(f"总策略数: {len(all_policies)}")

    best_policy, best_V, all_values = find_optimal_policy(env, all_policies)

    print(f"\n所有策略的 V(start):")
    for i, (policy, v_start) in enumerate(zip(all_policies, all_values)):
        actions_str = "".join([env.action_names[policy[s]] for s in range(3)])
        print(f"  策略 {i}: [{actions_str}] → V(0) = {v_start:.4f}")

    print(f"\n最优策略:")
    print_policy(env, best_policy)
    print_values(env, best_V)
    print(f"V*(start) = {best_V[env.start]:.4f}")

    # --- 测试 4: 验证 Bellman 最优方程 ---
    print("\n" + "=" * 60)
    print("测试 4: 验证 Bellman 最优方程")
    print("=" * 60)

    # TODO 7: 验证 V*(s) = max_a Σ_{s'} P(s'|s,a)[R + γ V*(s')]
    # 提示: 对每个非终止状态，计算所有动作的 Q*(s,a)，检查 V*(s) = max_a Q*(s,a)

    print("验证 V*(s) = max_a Q*(s,a):")
    Q_star = compute_q_from_v(env, best_V)

    # ---- 在这里实现验证 ----
    pass
    # ---- 代码结束 ----

    # --- 测试 5: 验证收缩映射性质 ---
    print("\n" + "=" * 60)
    print("测试 5: 验证收缩映射性质")
    print("=" * 60)

    # 随机生成两个价值函数
    np.random.seed(42)
    v1 = np.random.randn(env.n_states)
    v2 = np.random.randn(env.n_states)

    lhs, rhs, is_contraction = verify_contraction(env, test_policy, v1, v2)

    print(f"||v1 - v2||_∞ = {np.max(np.abs(v1 - v2)):.4f}")
    print(f"||T^π v1 - T^π v2||_∞ = {lhs:.4f}")
    print(f"γ ||v1 - v2||_∞ = {rhs:.4f}")
    print(f"收缩性质满足: {is_contraction}")
    print(f"收缩比率: {lhs / np.max(np.abs(v1 - v2)):.4f} (理论上限 γ={env.gamma})")

    print("\n" + "=" * 60)
    print("练习完成！")
    print("=" * 60)
    print("\n关键收获:")
    print("1. V → Q 转换让我们能从状态价值推导动作价值")
    print("2. 迭代策略评估收敛到精确解，验证了收缩映射定理")
    print("3. 小规模 MDP 可以枚举所有策略找最优解")
    print("4. Bellman 算子的收缩性质保证了算法收敛")
    print(f"5. γ={env.gamma} 控制收敛速度：γ 越小收敛越快")
