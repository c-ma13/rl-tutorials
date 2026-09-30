"""
Lesson 19: 模型基础 RL 简介 (Model-Based RL)
练习代码 - 练习版本

本练习实现 Dyna-Q 算法，包括：
1. Q-learning（Model-Free 基线）
2. 表格模型（确定性环境模型）
3. Dyna-Q（结合真实经验和模拟经验）
4. 在简单网格世界环境上对比
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: 简单网格世界环境
# ============================================================

class GridWorld:
    """简单的网格世界环境"""

    def __init__(self, size=5):
        """
        初始化网格世界

        参数:
            size: 网格大小（size x size）
        """
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]  # 障碍物位置
        self.state = self.start

        # 动作：上、下、左、右
        self.actions = [0, 1, 2, 3]
        self.action_effects = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }

    def reset(self):
        """重置环境"""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        执行动作

        参数:
            action: 动作（0-3）

        返回:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
        """
        # 计算下一个状态
        dx, dy = self.action_effects[action]
        next_x = self.state[0] + dx
        next_y = self.state[1] + dy
        next_state = (next_x, next_y)

        # 检查边界
        if not (0 <= next_x < self.size and 0 <= next_y < self.size):
            next_state = self.state  # 撞墙，保持原位

        # 检查障碍物
        if next_state in self.obstacles:
            next_state = self.state  # 撞障碍物，保持原位

        # 计算奖励
        if next_state == self.goal:
            reward = 1.0
            done = True
        else:
            reward = -0.01  # 每步小惩罚
            done = False

        self.state = next_state
        return next_state, reward, done

    def render(self):
        """可视化环境"""
        grid = np.zeros((self.size, self.size))
        for obs in self.obstacles:
            grid[obs] = -1  # 障碍物
        grid[self.goal] = 2  # 目标
        grid[self.state] = 1  # 当前位置
        print(grid)


# ============================================================
# Part 2: Q-learning Agent（Model-Free 基线）
# ============================================================

class QLearningAgent:
    """Q-learning 智能体"""

    def __init__(self, n_actions: int, learning_rate: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 0.1):
        """
        初始化 Q-learning Agent

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

    def select_action(self, state):
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
        pass
        # ---- 代码结束 ----

    def update(self, state, action, reward, next_state, done):
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
        # 1. 计算 TD 目标：y = r + γ max_a' Q(s', a')
        # 2. 如果 done=True，则 y = r
        # 3. 更新 Q 值：Q(s, a) ← Q(s, a) + α[y - Q(s, a)]
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----


# ============================================================
# Part 3: 表格模型
# ============================================================

class TabularModel:
    """表格模型：存储 (s, a) → (r, s')"""

    def __init__(self):
        """初始化模型"""
        self.model = {}  # {(state, action): (reward, next_state)}
        self.visited = set()  # 访问过的 (state, action) 对

    def update(self, state, action, reward, next_state):
        """
        更新模型

        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
        """
        # TODO 3: 实现模型更新
        # 提示：
        # 1. 存储 (state, action) → (reward, next_state)
        # 2. 记录访问过的 (state, action) 对
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def sample(self):
        """
        随机采样一个访问过的 (state, action) 对

        返回:
            state: 状态
            action: 动作
        """
        # TODO 4: 实现随机采样
        # 提示：
        # 1. 从 visited 中随机选择一个 (state, action)
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def predict(self, state, action):
        """
        预测下一个状态和奖励

        参数:
            state: 当前状态
            action: 动作

        返回:
            reward: 预测的奖励
            next_state: 预测的下一个状态
        """
        return self.model[(state, action)]

    def __len__(self):
        """返回模型中存储的经验数量"""
        return len(self.visited)


# ============================================================
# Part 4: Dyna-Q Agent
# ============================================================

class DynaQAgent(QLearningAgent):
    """Dyna-Q 智能体"""

    def __init__(self, n_actions: int, learning_rate: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 0.1,
                 planning_steps: int = 5):
        """
        初始化 Dyna-Q Agent

        参数:
            n_actions: 动作数量
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: 探索率
            planning_steps: 每次真实交互后的规划步数
        """
        super().__init__(n_actions, learning_rate, gamma, epsilon)
        self.planning_steps = planning_steps
        self.model = TabularModel()

    def update_with_planning(self, state, action, reward, next_state, done):
        """
        更新 Q 表并进行规划

        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        # TODO 5: 实现 Dyna-Q 更新
        # 提示：
        # 1. 用真实经验更新 Q 表（调用父类的 update）
        # 2. 更新模型
        # 3. 进行 planning_steps 次规划：
        #    a) 从模型中随机采样 (s, a)
        #    b) 用模型预测 (r, s')
        #    c) 用模拟经验更新 Q 表
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----


# ============================================================
# Part 5: 训练和评估
# ============================================================

def train_agent(agent, env, num_episodes=500):
    """
    训练智能体

    参数:
        agent: 智能体
        env: 环境
        num_episodes: 训练 episode 数

    返回:
        episode_steps: 每个 episode 的步数
    """
    episode_steps = []

    for episode in range(num_episodes):
        state = env.reset()
        steps = 0
        done = False

        while not done and steps < 100:  # 最多 100 步
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            # 更新智能体
            if isinstance(agent, DynaQAgent):
                agent.update_with_planning(state, action, reward, next_state, done)
            else:
                agent.update(state, action, reward, next_state, done)

            state = next_state
            steps += 1

        episode_steps.append(steps)

    return episode_steps


def compare_algorithms():
    """对比 Q-learning 和 Dyna-Q"""
    env = GridWorld(size=5)
    n_actions = len(env.actions)

    # Q-learning
    print("训练 Q-learning...")
    q_agent = QLearningAgent(n_actions, learning_rate=0.1, gamma=0.95, epsilon=0.1)
    q_steps = train_agent(q_agent, env, num_episodes=500)

    # Dyna-Q (n=5)
    print("训练 Dyna-Q (n=5)...")
    dyna5_agent = DynaQAgent(n_actions, learning_rate=0.1, gamma=0.95,
                             epsilon=0.1, planning_steps=5)
    dyna5_steps = train_agent(dyna5_agent, env, num_episodes=500)

    # Dyna-Q (n=50)
    print("训练 Dyna-Q (n=50)...")
    dyna50_agent = DynaQAgent(n_actions, learning_rate=0.1, gamma=0.95,
                              epsilon=0.1, planning_steps=50)
    dyna50_steps = train_agent(dyna50_agent, env, num_episodes=500)

    # 绘制结果
    plot_comparison(q_steps, dyna5_steps, dyna50_steps)


def plot_comparison(q_steps, dyna5_steps, dyna50_steps, window=10):
    """绘制对比结果"""
    plt.figure(figsize=(10, 6))

    # 平滑曲线
    def smooth(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window) / window, mode='valid')

    episodes = range(len(q_steps))
    plt.plot(episodes, q_steps, alpha=0.2, color='blue')
    plt.plot(range(window - 1, len(q_steps)),
             smooth(q_steps, window), label='Q-learning', color='blue')

    plt.plot(episodes, dyna5_steps, alpha=0.2, color='green')
    plt.plot(range(window - 1, len(dyna5_steps)),
             smooth(dyna5_steps, window), label='Dyna-Q (n=5)', color='green')

    plt.plot(episodes, dyna50_steps, alpha=0.2, color='red')
    plt.plot(range(window - 1, len(dyna50_steps)),
             smooth(dyna50_steps, window), label='Dyna-Q (n=50)', color='red')

    plt.xlabel('Episode')
    plt.ylabel('到达目标的步数')
    plt.title('Q-learning vs Dyna-Q 对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_19/dyna_q_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)

    # 对比算法
    compare_algorithms()

    print("\n训练完成！")
    print("Dyna-Q 通过规划加速了学习过程。")
    print("规划步数越多（n 越大），学习越快。")
