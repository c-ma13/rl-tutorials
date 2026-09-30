"""
Lesson 01 练习：GridWorld 环境与 Bellman 方程

目标：
1. 实现一个 4x4 GridWorld 环境
2. 实现随机策略下的价值函数精确求解（矩阵法）
3. 可视化价值函数

GridWorld 设定：
+----+----+----+----+
| S  |    |    |    |     S = 起点 (0,0)
+----+----+----+----+     G = 终点 (3,3), reward = +1
|    |  X |    |    |     X = 陷阱 (1,1), reward = -1
+----+----+----+----+     其他移动: reward = -0.04
|    |    |    |    |     撞墙: 留在原地
+----+----+----+----+     gamma = 0.9
|    |    |    |  G |
+----+----+----+----+

动作: 0=上, 1=下, 2=左, 3=右

请按照 TODO 提示完成代码，然后运行验证。
"""

import numpy as np

# ============================================================
# Part 1: 实现 GridWorld 环境
# ============================================================

class GridWorld:
    """4x4 GridWorld 环境"""

    def __init__(self):
        self.rows = 4
        self.cols = 4
        self.n_states = self.rows * self.cols  # 16 个状态
        self.n_actions = 4  # 上下左右

        # 特殊状态
        self.start = (0, 0)
        self.goal = (3, 3)
        self.trap = (1, 1)
        self.terminal_states = {self.goal, self.trap}

        # 奖励
        self.goal_reward = 1.0
        self.trap_reward = -1.0
        self.step_reward = -0.04

        # 折扣因子
        self.gamma = 0.9

        # 动作定义: 上=0, 下=1, 左=2, 右=3
        self.action_deltas = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1),   # 右
        }
        self.action_names = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    def state_to_idx(self, row, col):
        """将 (row, col) 转换为状态索引"""
        return row * self.cols + col

    def idx_to_state(self, idx):
        """将状态索引转换为 (row, col)"""
        return idx // self.cols, idx % self.cols

    def get_next_state(self, row, col, action):
        """
        给定当前位置和动作，返回下一个位置。
        如果撞墙则留在原地。

        TODO 1: 实现状态转移逻辑
        提示:
        - 从 self.action_deltas 获取动作对应的位移
        - 计算新位置 (new_row, new_col)
        - 检查是否越界，越界则返回原位置
        - 返回 (new_row, new_col)
        """
        # ---- 在这里写代码 ----
        # pass
        new_row, new_col = row, col
        if action == 0:
            new_row = max(0, row-1)
        elif action == 1:
            new_row = min(3, row+1)
        elif action == 2:
            new_col = max(0, col-1)
        elif action == 3:
            new_col = min(3, col+1)
        return (new_row, new_col)
        # ---- 代码结束 ----

    def get_reward(self, row, col):
        """
        获取到达 (row, col) 的奖励

        TODO 2: 实现奖励函数
        提示:
        - 到达 goal 返回 self.goal_reward
        - 到达 trap 返回 self.trap_reward
        - 其他返回 self.step_reward
        """
        # ---- 在这里写代码 ----
        # pass
        if (row, col)==self.goal:
            return self.goal_reward
        elif (row, col)==self.trap:
            return self.trap_reward
        else:
            return self.step_reward
        # ---- 代码结束 ----

    def is_terminal(self, row, col):
        """判断是否是终止状态"""
        return (row, col) in self.terminal_states

    def build_transition_matrix(self, policy):
        """
        构建策略 π 下的状态转移矩阵 P^π 和奖励向量 r^π

        参数:
            policy: np.array, shape (n_states, n_actions)
                    policy[s, a] = π(a|s)，即在状态 s 选择动作 a 的概率

        返回:
            P_pi: np.array, shape (n_states, n_states) — 转移矩阵
            r_pi: np.array, shape (n_states,)          — 即时奖励向量

        TODO 3: 构建转移矩阵和奖励向量
        提示:
        - 遍历每个状态 s (用 idx 0..15)
        - 对于终止状态: P_pi[s, s] = 1, r_pi[s] = 0 (吸收态，不再获得奖励)
        - 对于非终止状态: 遍历每个动作 a
            - 计算下一个状态 s'
            - P_pi[s, s'] += policy[s, a] (注意 += 因为多个动作可能转移到同一个 s')
            - r_pi[s] += policy[s, a] * reward(s')
        """
        P_pi = np.zeros((self.n_states, self.n_states))
        r_pi = np.zeros(self.n_states)

        # ---- 在这里写代码 ----
        # pass
        for i in range(self.n_states):
            row, col = self.idx_to_state(i)
            if self.is_terminal(row, col):
                P_pi[i,i]=1
            else:
                for j in range(self.n_actions):
                    next_state = self.get_next_state(row, col, j)
                    P_pi[i, self.state_to_idx(*next_state)] += policy[i,j]
                    r_pi[i] += policy[i,j] * self.get_reward(*next_state)
        # ---- 代码结束 ----
        return P_pi, r_pi


# ============================================================
# Part 2: 精确求解 Bellman 方程
# ============================================================

def solve_bellman_exact(env, policy):
    """
    用矩阵法精确求解 Bellman 期望方程:
        v = r + γ P v
        v = (I - γP)^{-1} r

    参数:
        env: GridWorld 实例
        policy: np.array, shape (n_states, n_actions)

    返回:
        v: np.array, shape (n_states,) — 每个状态的价值

    TODO 4: 实现矩阵求逆法
    提示:
    - 调用 env.build_transition_matrix(policy) 获取 P_pi 和 r_pi
    - 构造 (I - γP) 矩阵
    - 求逆后乘以 r_pi
    - 使用 np.linalg.solve 比 np.linalg.inv 更数值稳定
    """
    # ---- 在这里写代码 ----
    P_pi, r_pi = env.build_transition_matrix(policy)
    # v = np.zeros(env.n_states)
    v = np.linalg.solve(np.eye(env.n_states)-env.gamma * P_pi,r_pi)
    return v
    # ---- 代码结束 ----


# ============================================================
# Part 3: 可视化
# ============================================================

def print_values(env, values):
    """将价值函数以网格形式打印"""
    print("\n状态价值函数 V^π:")
    print("+" + "--------+" * env.cols)
    for r in range(env.rows):
        row_str = "|"
        for c in range(env.cols):
            idx = env.state_to_idx(r, c)
            if (r, c) == env.goal:
                row_str += f"  G    |"
            elif (r, c) == env.trap:
                row_str += f"  X    |"
            else:
                row_str += f" {values[idx]:5.2f} |"
        print(row_str)
        print("+" + "--------+" * env.cols)


def print_policy(env, policy):
    """将策略以网格形式打印（显示最大概率的动作）"""
    print("\n策略 π:")
    print("+" + "----+" * env.cols)
    for r in range(env.rows):
        row_str = "|"
        for c in range(env.cols):
            idx = env.state_to_idx(r, c)
            if (r, c) in env.terminal_states:
                row_str += " .  |"
            else:
                best_a = np.argmax(policy[idx])
                row_str += f" {env.action_names[best_a]}  |"
        print(row_str)
        print("+" + "----+" * env.cols)


# ============================================================
# Part 4: 主程序 — 运行并验证
# ============================================================

if __name__ == "__main__":
    env = GridWorld()

    # --- 测试 1: 随机策略 ---
    print("=" * 50)
    print("测试 1: 随机策略 (每个动作概率 0.25)")
    print("=" * 50)

    # 均匀随机策略: 每个状态下四个动作等概率
    random_policy = np.ones((env.n_states, env.n_actions)) / env.n_actions

    v_random = solve_bellman_exact(env, random_policy)
    print_values(env, v_random)

    # 验证: 起点 (0,0) 的价值应该是负的（因为随机走很慢，每步 -0.04）
    print(f"\nV(start) = {v_random[env.state_to_idx(0, 0)]:.4f}")
    print(f"V(0,1)   = {v_random[env.state_to_idx(0, 1)]:.4f}")
    print(f"V(2,3)   = {v_random[env.state_to_idx(2, 3)]:.4f}")

    # --- 测试 2: 自定义确定性策略 ---
    print("\n" + "=" * 50)
    print("测试 2: 手动设计的 '聪明' 策略")
    print("=" * 50)

    # TODO 5 (可选挑战): 手动设计一个好的确定性策略
    # 让每个状态尽可能朝 goal 方向走，同时避开 trap
    # 提示: 确定性策略中，某个动作概率为 1，其余为 0
    smart_policy = np.zeros((env.n_states, env.n_actions))
    for i in range(env.n_states):
        row, col = env.idx_to_state(i)
        if row == 3:
            smart_policy[i,3] = 1
        elif col == 3:
            smart_policy[i,1] = 1
        # elif i == 1:
        #     smart_policy[i,3] = 1
        elif i == 4:
            smart_policy[i,1] = 1
        # else:
        #     smart_policy[i,1] = 0.5
        #     smart_policy[i,3] = 0.5
        else:
            smart_policy[i,3] = 1
    # 示例：状态 (0,0) 向右走
    # smart_policy[env.state_to_idx(0, 0), 3] = 1.0  # 右
    # ... 为其他状态设置动作 ...

    # 如果你完成了 smart_policy，取消下面的注释运行：
    v_smart = solve_bellman_exact(env, smart_policy)
    print_values(env, v_smart)
    print_policy(env, smart_policy)
    print(f"\nV(start) = {v_smart[env.state_to_idx(0, 0)]:.4f}")

    # --- 测试 3: 验证 Bellman 方程 ---
    print("\n" + "=" * 50)
    print("测试 3: 验证 Bellman 方程成立")
    print("=" * 50)

    # TODO 6: 验证 Bellman 方程
    # 对于非终止状态 s，检验:
    #   V(s) ≈ Σ_a π(a|s) Σ_s' P(s'|s,a) [R(s,a,s') + γ V(s')]
    # 提示: 遍历一个非终止状态，手动计算右边，看是否等于左边
    print("选择状态 (0,0) 手动验证 Bellman 方程:")
    s_row, s_col = 0, 0
    s_idx = env.state_to_idx(s_row, s_col)

    # ---- 在这里写验证代码 ----
    # bellman_rhs = 0.0
    # for a in range(env.n_actions):
    #     next_r, next_c = env.get_next_state(s_row, s_col, a)
    #     next_idx = env.state_to_idx(next_r, next_c)
    #     reward = env.get_reward(next_r, next_c)
    #     bellman_rhs += random_policy[s_idx, a] * (reward + env.gamma * v_random[next_idx])
    # print(f"  V(0,0) from solve   = {v_random[s_idx]:.6f}")
    # print(f"  V(0,0) from Bellman = {bellman_rhs:.6f}")
    # print(f"  差异: {abs(v_random[s_idx] - bellman_rhs):.2e}")
    # ---- 代码结束 ----

    print("\n练习完成！")
    print("下一步: 尝试完成 TODO 5 设计一个好策略，比较它和随机策略的 V(start)")
