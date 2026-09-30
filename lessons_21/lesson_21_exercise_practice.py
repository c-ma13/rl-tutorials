"""
Lesson 21: 离线 RL (Offline RL)
练习代码 - 练习版本

本练习实现离线 RL 基础算法，包括：
1. GridWorld 环境
2. 收集离线数据集（使用次优策略）
3. 朴素离线 Q-learning
4. CQL (Conservative Q-Learning)
5. 对比分析
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict, deque
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: GridWorld 环境
# ============================================================

class GridWorld:
    """
    简单的网格世界环境

    - 智能体从起点移动到终点
    - 有障碍物
    - 奖励：到达终点 +10，每步 -0.1，撞墙 -1
    """

    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        # 障碍物
        self.obstacles = {(1, 1), (1, 2), (2, 1), (3, 3)}

        self.actions = [0, 1, 2, 3]  # 上下左右
        self.action_effects = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        self.action_names = ['↑', '↓', '←', '→']

        self.reset()

    def reset(self):
        """重置环境"""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        执行动作

        返回:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
        """
        dx, dy = self.action_effects[action]
        next_x = self.state[0] + dx
        next_y = self.state[1] + dy
        next_state = (next_x, next_y)

        # 检查边界
        if not (0 <= next_x < self.size and 0 <= next_y < self.size):
            # 撞墙
            return self.state, -1.0, False

        # 检查障碍物
        if next_state in self.obstacles:
            # 撞障碍物
            return self.state, -1.0, False

        # 移动成功
        self.state = next_state

        # 检查是否到达终点
        if self.state == self.goal:
            return self.state, 10.0, True
        else:
            return self.state, -0.1, False

    def render(self, q_table=None, policy=None):
        """可视化环境"""
        grid = np.zeros((self.size, self.size))

        # 标记障碍物
        for obs in self.obstacles:
            grid[obs] = -1

        # 标记起点和终点
        grid[self.start] = 0.5
        grid[self.goal] = 1.0

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(grid, cmap='RdYlGn', vmin=-1, vmax=1)

        # 绘制网格线
        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)

        # 标注
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) in self.obstacles:
                    ax.text(j, i, 'X', ha='center', va='center',
                           fontsize=20, color='white', weight='bold')
                elif (i, j) == self.start:
                    ax.text(j, i, 'S', ha='center', va='center',
                           fontsize=20, color='blue', weight='bold')
                elif (i, j) == self.goal:
                    ax.text(j, i, 'G', ha='center', va='center',
                           fontsize=20, color='green', weight='bold')
                elif policy is not None and (i, j) not in self.obstacles:
                    # 绘制策略箭头
                    best_action = np.argmax(q_table[(i, j)])
                    ax.text(j, i, self.action_names[best_action],
                           ha='center', va='center', fontsize=16)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('GridWorld 环境', fontsize=16)

        return fig


# ============================================================
# Part 2: 收集离线数据集
# ============================================================

def collect_offline_dataset(env, behavior_policy, num_episodes=500,
                            max_steps=50, epsilon=0.3):
    """
    使用行为策略收集离线数据集

    参数:
        env: 环境
        behavior_policy: 行为策略（Q 表或 'random'）
        num_episodes: 收集的 episode 数量
        max_steps: 每个 episode 的最大步数
        epsilon: ε-greedy 探索率

    返回:
        dataset: 数据集 [(s, a, r, s', done), ...]
        returns: 每个 episode 的回报
    """
    dataset = []
    returns = []

    for ep in range(num_episodes):
        state = env.reset()
        episode_return = 0

        for step in range(max_steps):
            # 选择动作（ε-greedy）
            if behavior_policy == 'random' or np.random.random() < epsilon:
                action = np.random.choice(env.actions)
            else:
                # 使用行为策略
                q_values = behavior_policy[state]
                action = np.argmax(q_values)

            # 执行动作
            next_state, reward, done = env.step(action)

            # 存储转移
            dataset.append((state, action, reward, next_state, done))
            episode_return += reward

            state = next_state

            if done:
                break

        returns.append(episode_return)

    return dataset, returns


# ============================================================
# Part 3: 朴素离线 Q-learning
# ============================================================

class OfflineQLearning:
    """朴素离线 Q-learning"""

    def __init__(self, n_actions, learning_rate=0.1, gamma=0.95):
        """
        初始化

        参数:
            n_actions: 动作数量
            learning_rate: 学习率
            gamma: 折扣因子
        """
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def update(self, dataset, num_iterations=1000):
        """
        在离线数据集上训练

        参数:
            dataset: 离线数据集
            num_iterations: 训练迭代次数
        """
        # TODO 1: 实现离线 Q-learning 更新
        # 提示：
        # 1. 对于每次迭代，从数据集中随机采样
        # 2. 计算 TD 目标: y = r + γ max_a' Q(s', a')
        # 3. 更新: Q(s, a) ← Q(s, a) + α[y - Q(s, a)]
        # 4. 注意处理 done 的情况
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def get_action(self, state):
        """获取贪心动作"""
        return np.argmax(self.q_table[state])


# ============================================================
# Part 4: CQL (Conservative Q-Learning)
# ============================================================

class CQL:
    """Conservative Q-Learning (表格版本)"""

    def __init__(self, n_actions, learning_rate=0.1, gamma=0.95, alpha=1.0):
        """
        初始化 CQL

        参数:
            n_actions: 动作数量
            learning_rate: 学习率
            gamma: 折扣因子
            alpha: 保守系数（CQL 惩罚权重）
        """
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.alpha = alpha
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def update(self, dataset, num_iterations=1000, batch_size=32):
        """
        在离线数据集上训练（使用 CQL）

        参数:
            dataset: 离线数据集
            num_iterations: 训练迭代次数
            batch_size: 批大小
        """
        # TODO 2: 实现 CQL 更新
        # 提示：
        # 1. 采样 batch
        # 2. 计算标准 TD 目标
        # 3. 计算 CQL 惩罚项:
        #    penalty = log(Σ_a exp(Q(s, a))) - Q(s, a_data)
        #    使用 log-sum-exp 技巧: logsumexp = max_q + log(Σ exp(q - max_q))
        # 4. 组合更新: Q(s,a) += lr * (td_error - alpha * penalty)
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def get_action(self, state):
        """获取贪心动作"""
        return np.argmax(self.q_table[state])


# ============================================================
# Part 5: 评估策略
# ============================================================

def evaluate_policy(env, q_table, num_episodes=100, max_steps=50):
    """
    评估策略性能

    参数:
        env: 环境
        q_table: Q 表
        num_episodes: 评估 episode 数量
        max_steps: 每个 episode 的最大步数

    返回:
        mean_return: 平均回报
        success_rate: 成功率
    """
    # TODO 3: 实现策略评估
    # 提示：
    # 1. 运行多个 episode
    # 2. 使用贪心策略（不探索）
    # 3. 记录回报和是否成功
    # 4. 计算平均回报和成功率
    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----


# ============================================================
# Part 6: 训练行为策略（用于收集数据）
# ============================================================

def train_behavior_policy(env, num_episodes=1000, lr=0.1, gamma=0.95,
                          epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
    """训练一个次优的行为策略（用于收集离线数据）"""
    q_table = defaultdict(lambda: np.zeros(len(env.actions)))
    epsilon = epsilon_start

    for ep in range(num_episodes):
        state = env.reset()

        for step in range(50):
            # ε-greedy
            if np.random.random() < epsilon:
                action = np.random.choice(env.actions)
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done = env.step(action)

            # Q-learning 更新
            if done:
                target = reward
            else:
                target = reward + gamma * np.max(q_table[next_state])

            q_table[state][action] += lr * (target - q_table[state][action])

            state = next_state

            if done:
                break

        # 衰减探索率
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return q_table


# ============================================================
# Part 7: 可视化
# ============================================================

def plot_results(behavior_returns, offline_results, cql_results):
    """绘制结果对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 数据集质量
    ax = axes[0]
    ax.hist(behavior_returns, bins=30, alpha=0.7, color='gray', edgecolor='black')
    ax.axvline(np.mean(behavior_returns), color='red', linestyle='--',
               linewidth=2, label=f'平均: {np.mean(behavior_returns):.2f}')
    ax.set_xlabel('Episode 回报')
    ax.set_ylabel('频数')
    ax.set_title('离线数据集质量分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 方法对比
    ax = axes[1]
    methods = ['行为策略', '离线 Q-learning', 'CQL']
    returns = [
        np.mean(behavior_returns),
        offline_results['return'],
        cql_results['return']
    ]
    success_rates = [
        0,  # 行为策略的成功率（从数据集估计）
        offline_results['success_rate'],
        cql_results['success_rate']
    ]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, returns, width, label='平均回报', color='skyblue')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, success_rates, width, label='成功率', color='lightcoral')

    ax.set_xlabel('方法')
    ax.set_ylabel('平均回报')
    ax2.set_ylabel('成功率')
    ax.set_title('离线 RL 方法对比')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_21/offline_rl_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def visualize_policies(env, q_tables, titles):
    """可视化多个策略"""
    n = len(q_tables)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))

    if n == 1:
        axes = [axes]

    for ax, q_table, title in zip(axes, q_tables, titles):
        # 创建网格
        grid = np.zeros((env.size, env.size))
        for obs in env.obstacles:
            grid[obs] = -1
        grid[env.start] = 0.5
        grid[env.goal] = 1.0

        ax.imshow(grid, cmap='RdYlGn', vmin=-1, vmax=1)

        # 绘制网格线
        for i in range(env.size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)

        # 标注策略
        for i in range(env.size):
            for j in range(env.size):
                if (i, j) in env.obstacles:
                    ax.text(j, i, 'X', ha='center', va='center',
                           fontsize=16, color='white', weight='bold')
                elif (i, j) == env.start:
                    ax.text(j, i, 'S', ha='center', va='center',
                           fontsize=16, color='blue', weight='bold')
                elif (i, j) == env.goal:
                    ax.text(j, i, 'G', ha='center', va='center',
                           fontsize=16, color='green', weight='bold')
                else:
                    best_action = np.argmax(q_table[(i, j)])
                    ax.text(j, i, env.action_names[best_action],
                           ha='center', va='center', fontsize=14)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_21/policies_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # ----- 步骤 1: 创建环境 -----
    print("=" * 60)
    print("步骤 1: 创建 GridWorld 环境")
    print("=" * 60)

    env = GridWorld(size=5)
    print(f"环境大小: {env.size}x{env.size}")
    print(f"起点: {env.start}, 终点: {env.goal}")
    print(f"障碍物: {env.obstacles}")

    # ----- 步骤 2: 训练行为策略 -----
    print("\n" + "=" * 60)
    print("步骤 2: 训练次优行为策略（用于收集数据）")
    print("=" * 60)

    behavior_q = train_behavior_policy(env, num_episodes=500, lr=0.1, gamma=0.95,
                                       epsilon_start=1.0, epsilon_end=0.2,
                                       epsilon_decay=0.995)

    behavior_return, behavior_success = evaluate_policy(env, behavior_q,
                                                        num_episodes=100)
    print(f"行为策略性能: 平均回报 = {behavior_return:.2f}, 成功率 = {behavior_success:.2%}")

    # ----- 步骤 3: 收集离线数据集 -----
    print("\n" + "=" * 60)
    print("步骤 3: 使用行为策略收集离线数据集")
    print("=" * 60)

    dataset, behavior_returns = collect_offline_dataset(
        env, behavior_q, num_episodes=500, max_steps=50, epsilon=0.3
    )

    print(f"数据集大小: {len(dataset)} 条转移")
    print(f"数据集平均回报: {np.mean(behavior_returns):.2f}")
    print(f"数据集回报范围: [{np.min(behavior_returns):.2f}, {np.max(behavior_returns):.2f}]")

    # ----- 步骤 4: 朴素离线 Q-learning -----
    print("\n" + "=" * 60)
    print("步骤 4: 训练朴素离线 Q-learning")
    print("=" * 60)

    offline_agent = OfflineQLearning(n_actions=len(env.actions), learning_rate=0.1, gamma=0.95)
    offline_agent.update(dataset, num_iterations=5000)

    offline_return, offline_success = evaluate_policy(env, offline_agent.q_table,
                                                      num_episodes=100)
    print(f"离线 Q-learning 性能: 平均回报 = {offline_return:.2f}, 成功率 = {offline_success:.2%}")

    # ----- 步骤 5: CQL -----
    print("\n" + "=" * 60)
    print("步骤 5: 训练 CQL (Conservative Q-Learning)")
    print("=" * 60)

    cql_agent = CQL(n_actions=len(env.actions), learning_rate=0.1, gamma=0.95, alpha=1.0)
    cql_agent.update(dataset, num_iterations=5000, batch_size=32)

    cql_return, cql_success = evaluate_policy(env, cql_agent.q_table, num_episodes=100)
    print(f"CQL 性能: 平均回报 = {cql_return:.2f}, 成功率 = {cql_success:.2%}")

    # ----- 步骤 6: 结果对比 -----
    print("\n" + "=" * 60)
    print("步骤 6: 结果对比")
    print("=" * 60)

    print(f"\n{'方法':<20} {'平均回报':<15} {'成功率':<15}")
    print("-" * 50)
    print(f"{'行为策略':<20} {behavior_return:<15.2f} {behavior_success:<15.2%}")
    print(f"{'离线 Q-learning':<20} {offline_return:<15.2f} {offline_success:<15.2%}")
    print(f"{'CQL':<20} {cql_return:<15.2f} {cql_success:<15.2%}")

    # ----- 步骤 7: 可视化 -----
    print("\n" + "=" * 60)
    print("步骤 7: 可视化结果")
    print("=" * 60)

    offline_results = {'return': offline_return, 'success_rate': offline_success}
    cql_results = {'return': cql_return, 'success_rate': cql_success}

    plot_results(behavior_returns, offline_results, cql_results)

    visualize_policies(
        env,
        [behavior_q, offline_agent.q_table, cql_agent.q_table],
        ['行为策略', '离线 Q-learning', 'CQL']
    )

    print("\n离线 RL 实验完成！")
    print("\n关键发现：")
    print("1. 朴素离线 Q-learning 可能受外推误差影响")
    print("2. CQL 通过保守估计避免选择数据集外的动作")
    print("3. CQL 在离线数据上通常更稳定和有效")
