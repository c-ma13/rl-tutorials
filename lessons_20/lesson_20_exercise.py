"""
Lesson 20: 多智能体 RL 简介 (Multi-Agent RL)
练习代码 - 参考实现

本练习实现多智能体 RL 基础算法，包括：
1. 矩阵博弈环境（石头剪刀布）
2. Independent Q-Learning (IQL)
3. 自博弈（Self-Play）训练
4. 合作网格世界（两个智能体合作抓捕）
5. 参数共享 IQL
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: 矩阵博弈环境（石头剪刀布）
# ============================================================

class RockPaperScissors:
    """石头剪刀布环境（零和博弈）"""

    def __init__(self):
        """
        动作：0=石头, 1=剪刀, 2=布
        奖励矩阵（从玩家1视角）：
              石头  剪刀  布
        石头  [ 0,   1,  -1]
        剪刀  [-1,   0,   1]
        布    [ 1,  -1,   0]
        """
        self.n_actions = 3
        self.action_names = ['石头', '剪刀', '布']
        # 玩家1 的奖励矩阵
        self.payoff_matrix = np.array([
            [0,  1, -1],
            [-1, 0,  1],
            [1, -1,  0]
        ])

    def step(self, action1, action2):
        """
        执行一轮博弈

        参数:
            action1: 玩家1 的动作
            action2: 玩家2 的动作

        返回:
            reward1: 玩家1 的奖励
            reward2: 玩家2 的奖励
        """
        reward1 = self.payoff_matrix[action1, action2]
        reward2 = -reward1  # 零和博弈
        return reward1, reward2


# ============================================================
# Part 2: Independent Q-Learning Agent
# ============================================================

class IQLAgent:
    """Independent Q-Learning 智能体"""

    def __init__(self, n_actions: int, learning_rate: float = 0.1,
                 gamma: float = 0.0, epsilon: float = 0.3):
        """
        初始化 IQL Agent

        参数:
            n_actions: 动作数量
            learning_rate: 学习率
            gamma: 折扣因子（矩阵博弈中为 0，因为是单步博弈）
            epsilon: 探索率
        """
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        # 对于矩阵博弈，状态只有一个（"dummy" 状态）
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state="default"):
        """
        选择动作（ε-greedy）

        参数:
            state: 当前状态

        返回:
            action: 选择的动作
        """
        # TODO 1: 实现 ε-greedy 动作选择
        # 提示：
        # 1. 以概率 epsilon 随机选择动作
        # 2. 否则选择 Q 值最大的动作
        # ---- 在这里实现 ----
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
        # ---- 代码结束 ----

    def update(self, state, action, reward, next_state=None, done=True):
        """
        更新 Q 表

        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        # TODO 2: 实现 Q-learning 更新
        # 提示：
        # 1. 对于矩阵博弈（单步），目标就是 reward
        # 2. 对于多步博弈，目标是 r + γ max_a' Q(s', a')
        # 3. Q(s, a) ← Q(s, a) + α[target - Q(s, a)]
        # ---- 在这里实现 ----
        if done or next_state is None:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])
        # ---- 代码结束 ----

    def get_policy(self, state="default"):
        """获取当前策略（动作概率分布）"""
        q_values = self.q_table[state]
        # 用 softmax 近似
        exp_q = np.exp(q_values - np.max(q_values))
        return exp_q / exp_q.sum()


# ============================================================
# Part 3: 自博弈训练
# ============================================================

def self_play_training(env, num_rounds=10000, lr=0.1, epsilon_start=1.0,
                       epsilon_end=0.01, epsilon_decay=0.9995):
    """
    自博弈训练

    参数:
        env: 博弈环境
        num_rounds: 训练轮数
        lr: 学习率
        epsilon_start: 初始探索率
        epsilon_end: 最终探索率
        epsilon_decay: 探索率衰减

    返回:
        agent: 训练好的智能体
        history: 训练历史
    """
    # TODO 3: 实现自博弈训练循环
    # 提示：
    # 1. 创建两个 IQL 智能体（player1 和 player2）
    # 2. 每轮：
    #    a) 两个智能体各自选择动作
    #    b) 环境返回奖励
    #    c) 两个智能体各自更新 Q 表
    # 3. 衰减探索率
    # 4. 定期记录胜率
    # ---- 在这里实现 ----
    player1 = IQLAgent(env.n_actions, learning_rate=lr, gamma=0.0,
                       epsilon=epsilon_start)
    player2 = IQLAgent(env.n_actions, learning_rate=lr, gamma=0.0,
                       epsilon=epsilon_start)

    history = {
        'p1_wins': [],
        'p2_wins': [],
        'draws': [],
        'p1_actions': [],
        'p2_actions': [],
    }

    win_count = [0, 0, 0]  # [p1_wins, p2_wins, draws]
    window = 500

    for round_idx in range(num_rounds):
        # 两个智能体选择动作
        a1 = player1.select_action()
        a2 = player2.select_action()

        # 获取奖励
        r1, r2 = env.step(a1, a2)

        # 更新 Q 表
        player1.update("default", a1, r1)
        player2.update("default", a2, r2)

        # 记录结果
        if r1 > 0:
            win_count[0] += 1
        elif r2 > 0:
            win_count[1] += 1
        else:
            win_count[2] += 1

        history['p1_actions'].append(a1)
        history['p2_actions'].append(a2)

        # 衰减探索率
        player1.epsilon = max(epsilon_end,
                              player1.epsilon * epsilon_decay)
        player2.epsilon = max(epsilon_end,
                              player2.epsilon * epsilon_decay)

        # 定期记录
        if (round_idx + 1) % window == 0:
            total = sum(win_count)
            history['p1_wins'].append(win_count[0] / total)
            history['p2_wins'].append(win_count[1] / total)
            history['draws'].append(win_count[2] / total)
            win_count = [0, 0, 0]

    return player1, player2, history
    # ---- 代码结束 ----


# ============================================================
# Part 4: 合作网格世界
# ============================================================

class CooperativeGridWorld:
    """
    合作网格世界：两个智能体合作抓捕目标

    规则：
    - 两个智能体在网格上移动
    - 目标（猎物）固定不动
    - 当两个智能体同时相邻于目标时，抓捕成功
    - 合作奖励：抓捕成功 +10，每步 -0.1
    """

    def __init__(self, size=5):
        self.size = size
        self.prey_pos = (size // 2, size // 2)  # 猎物在中心
        self.actions = [0, 1, 2, 3]  # 上下左右
        self.action_effects = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        self.reset()

    def reset(self):
        """重置环境，随机放置两个智能体"""
        while True:
            self.agent1_pos = (np.random.randint(self.size),
                               np.random.randint(self.size))
            self.agent2_pos = (np.random.randint(self.size),
                               np.random.randint(self.size))
            # 确保不在猎物位置且不重叠
            if (self.agent1_pos != self.prey_pos and
                    self.agent2_pos != self.prey_pos and
                    self.agent1_pos != self.agent2_pos):
                break
        return self._get_state()

    def _get_state(self):
        """返回联合状态"""
        return (self.agent1_pos, self.agent2_pos)

    def _move(self, pos, action):
        """移动智能体"""
        dx, dy = self.action_effects[action]
        new_x = max(0, min(self.size - 1, pos[0] + dx))
        new_y = max(0, min(self.size - 1, pos[1] + dy))
        return (new_x, new_y)

    def _is_adjacent(self, pos1, pos2):
        """检查两个位置是否相邻"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def step(self, action1, action2):
        """
        执行联合动作

        参数:
            action1: 智能体1 的动作
            action2: 智能体2 的动作

        返回:
            state: 新的联合状态
            reward: 共享奖励
            done: 是否结束
        """
        # 移动智能体
        new_pos1 = self._move(self.agent1_pos, action1)
        new_pos2 = self._move(self.agent2_pos, action2)

        # 避免碰撞
        if new_pos1 == new_pos2:
            new_pos1 = self.agent1_pos
            new_pos2 = self.agent2_pos

        # 避免占据猎物位置
        if new_pos1 == self.prey_pos:
            new_pos1 = self.agent1_pos
        if new_pos2 == self.prey_pos:
            new_pos2 = self.agent2_pos

        self.agent1_pos = new_pos1
        self.agent2_pos = new_pos2

        # 检查是否抓捕成功
        adj1 = self._is_adjacent(self.agent1_pos, self.prey_pos)
        adj2 = self._is_adjacent(self.agent2_pos, self.prey_pos)

        if adj1 and adj2:
            reward = 10.0
            done = True
        else:
            reward = -0.1
            done = False

        return self._get_state(), reward, done


# ============================================================
# Part 5: 合作 IQL Agent（参数共享思想）
# ============================================================

class CooperativeIQLAgent:
    """合作 IQL 智能体（使用联合状态）"""

    def __init__(self, n_actions: int, learning_rate: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 0.3):
        """
        初始化合作 IQL Agent

        参数:
            n_actions: 动作数量
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: 探索率
        """
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, local_state):
        """
        选择动作

        参数:
            local_state: 局部状态（自身位置 + 猎物位置等信息）

        返回:
            action: 选择的动作
        """
        # TODO 4: 实现 ε-greedy 动作选择
        # 提示：
        # 与 IQLAgent 类似，但使用局部状态作为 key
        # ---- 在这里实现 ----
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[local_state])
        # ---- 代码结束 ----

    def update(self, state, action, reward, next_state, done):
        """
        更新 Q 表（使用共享奖励）

        参数:
            state: 当前局部状态
            action: 执行的动作
            reward: 共享奖励
            next_state: 下一个局部状态
            done: 是否结束
        """
        # TODO 5: 实现 Q-learning 更新
        # 提示：
        # 1. 与标准 Q-learning 相同
        # 2. 关键区别：奖励是团队共享的
        # ---- 在这里实现 ----
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])
        # ---- 代码结束 ----


# ============================================================
# Part 6: 训练和评估
# ============================================================

def train_cooperative(env, num_episodes=3000, max_steps=50,
                      lr=0.1, gamma=0.95,
                      epsilon_start=1.0, epsilon_end=0.05,
                      epsilon_decay=0.999):
    """训练合作智能体"""
    n_actions = len(env.actions)

    agent1 = CooperativeIQLAgent(n_actions, lr, gamma, epsilon_start)
    agent2 = CooperativeIQLAgent(n_actions, lr, gamma, epsilon_start)

    episode_steps = []
    episode_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            # 构造局部状态（包含自身位置和猎物位置）
            local_state1 = (state[0], env.prey_pos)
            local_state2 = (state[1], env.prey_pos)

            # 选择动作
            a1 = agent1.select_action(local_state1)
            a2 = agent2.select_action(local_state2)

            # 执行动作
            next_state, reward, done = env.step(a1, a2)

            # 构造下一个局部状态
            next_local1 = (next_state[0], env.prey_pos)
            next_local2 = (next_state[1], env.prey_pos)

            # 更新两个智能体（共享奖励）
            agent1.update(local_state1, a1, reward, next_local1, done)
            agent2.update(local_state2, a2, reward, next_local2, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        episode_steps.append(steps)
        episode_rewards.append(total_reward)

        # 衰减探索率
        agent1.epsilon = max(epsilon_end, agent1.epsilon * epsilon_decay)
        agent2.epsilon = max(epsilon_end, agent2.epsilon * epsilon_decay)

    return agent1, agent2, episode_steps, episode_rewards


def plot_self_play_results(history):
    """绘制自博弈结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 胜率变化
    ax = axes[0]
    x = range(len(history['p1_wins']))
    ax.plot(x, history['p1_wins'], label='玩家1 胜率', color='blue')
    ax.plot(x, history['p2_wins'], label='玩家2 胜率', color='red')
    ax.plot(x, history['draws'], label='平局率', color='green')
    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='纳什均衡 (1/3)')
    ax.set_xlabel('训练窗口')
    ax.set_ylabel('比例')
    ax.set_title('自博弈：胜率变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 动作分布变化
    ax = axes[1]
    window = 500
    n_windows = len(history['p1_actions']) // window
    action_names = ['石头', '剪刀', '布']
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for a_idx in range(3):
        proportions = []
        for w in range(n_windows):
            start = w * window
            end = start + window
            actions = history['p1_actions'][start:end]
            proportions.append(actions.count(a_idx) / window)
        ax.plot(range(n_windows), proportions, label=f'P1: {action_names[a_idx]}',
                color=colors[a_idx])

    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='均匀策略 (1/3)')
    ax.set_xlabel('训练窗口')
    ax.set_ylabel('动作比例')
    ax.set_title('玩家1 动作分布变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_20/self_play_rps.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_cooperative_results(episode_steps, episode_rewards, window=50):
    """绘制合作训练结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    def smooth(data, w):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w) / w, mode='valid')

    # 步数
    ax = axes[0]
    ax.plot(episode_steps, alpha=0.2, color='blue')
    smoothed = smooth(episode_steps, window)
    ax.plot(range(window - 1, len(episode_steps)), smoothed,
            label=f'{window}-episode 平均', color='blue')
    ax.set_xlabel('Episode')
    ax.set_ylabel('抓捕步数')
    ax.set_title('合作抓捕：步数变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 奖励
    ax = axes[1]
    ax.plot(episode_rewards, alpha=0.2, color='green')
    smoothed = smooth(episode_rewards, window)
    ax.plot(range(window - 1, len(episode_rewards)), smoothed,
            label=f'{window}-episode 平均', color='green')
    ax.set_xlabel('Episode')
    ax.set_ylabel('累计奖励')
    ax.set_title('合作抓捕：奖励变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_20/cooperative_capture.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # ----- 实验 1：石头剪刀布自博弈 -----
    print("=" * 60)
    print("实验 1：石头剪刀布 - 自博弈")
    print("=" * 60)

    rps_env = RockPaperScissors()
    p1, p2, history = self_play_training(
        rps_env, num_rounds=50000, lr=0.01,
        epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9999
    )

    print(f"\n训练完成！")
    print(f"玩家1 策略（softmax）: {p1.get_policy()}")
    print(f"玩家2 策略（softmax）: {p2.get_policy()}")
    print(f"纳什均衡策略: [1/3, 1/3, 1/3]")

    plot_self_play_results(history)

    # ----- 实验 2：合作抓捕 -----
    print("\n" + "=" * 60)
    print("实验 2：合作网格世界 - 协同抓捕")
    print("=" * 60)

    coop_env = CooperativeGridWorld(size=5)
    agent1, agent2, ep_steps, ep_rewards = train_cooperative(
        coop_env, num_episodes=3000, max_steps=50,
        lr=0.1, gamma=0.95,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999
    )

    print(f"\n训练完成！")
    print(f"最后100个episode平均步数: {np.mean(ep_steps[-100:]):.1f}")
    print(f"最后100个episode平均奖励: {np.mean(ep_rewards[-100:]):.2f}")

    plot_cooperative_results(ep_steps, ep_rewards)

    print("\n多智能体 RL 实验完成！")
    print("关键发现：")
    print("1. 石头剪刀布中，自博弈趋向纳什均衡（均匀随机策略）")
    print("2. 合作抓捕中，两个智能体学会协同从两侧包围猎物")
