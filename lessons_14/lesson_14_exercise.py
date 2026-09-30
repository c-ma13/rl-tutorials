"""
Lesson 14: DQN 进阶 (Advanced DQN)
练习代码 - 参考实现

本练习实现 DQN 的改进算法，包括：
1. Double DQN：解决高估偏差
2. Dueling DQN：分离状态价值和优势函数
3. 对比实验：DQN vs Double DQN vs Dueling DQN
4. 在 CartPole-v1 环境上训练和评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: Dueling DQN 网络
# ============================================================

class DuelingDQN(nn.Module):
    """Dueling DQN 网络：分离状态价值和优势函数"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        初始化 Dueling DQN

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(DuelingDQN, self).__init__()
        # TODO 1: 实现 Dueling DQN 网络结构
        # 提示：
        # 1. 共享特征提取层：state_dim -> hidden_dim -> hidden_dim
        # 2. 价值流：hidden_dim -> hidden_dim -> 1
        # 3. 优势流：hidden_dim -> hidden_dim -> action_dim
        # 4. 所有隐藏层使用 ReLU 激活
        # ---- 在这里实现 ----
        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # ---- 代码结束 ----

    def forward(self, x):
        """
        前向传播

        参数:
            x: 状态张量，形状 (batch_size, state_dim)

        返回:
            q_values: Q 值张量，形状 (batch_size, action_dim)
        """
        # TODO 2: 实现 Dueling DQN 前向传播
        # 提示：
        # 1. 通过共享特征层提取特征
        # 2. 分别计算价值 V(s) 和优势 A(s,a)
        # 3. 组合：Q(s,a) = V(s) + (A(s,a) - max_a' A(s,a'))
        # 4. 使用 max 而不是 mean（更稳定）
        # ---- 在这里实现 ----
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # 减去最大优势（方案 2）
        q_values = value + (advantage - advantage.max(dim=1, keepdim=True)[0])
        return q_values
        # ---- 代码结束 ----


# ============================================================
# Part 2: 经验回放缓冲区（复用 Lesson 13）
# ============================================================

class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Part 3: Double DQN Agent
# ============================================================

class DoubleDQNAgent:
    """Double DQN 智能体"""

    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 10000,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 100,
                 use_dueling: bool = False):
        """
        初始化 Double DQN Agent

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            epsilon_decay: 探索率衰减步数
            buffer_size: 回放缓冲区大小
            batch_size: 训练 batch 大小
            target_update_freq: 目标网络更新频率
            use_dueling: 是否使用 Dueling 架构
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_dueling = use_dueling

        # 创建网络
        if use_dueling:
            self.online_net = DuelingDQN(state_dim, action_dim)
            self.target_net = DuelingDQN(state_dim, action_dim)
        else:
            # 使用标准 DQN 网络（从 Lesson 13 导入）
            from lessons_13.exercises.lesson_13_exercise import DQN
            self.online_net = DQN(state_dim, action_dim)
            self.target_net = DQN(state_dim, action_dim)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)

        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)

        # 训练步数
        self.train_step = 0

    def select_action(self, state, training=True):
        """选择动作（ε-greedy）"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.online_net(state_tensor)
                return q_values.argmax(1).item()

    def update_epsilon(self):
        """更新探索率（线性衰减）"""
        self.epsilon = max(self.epsilon_end,
                          self.epsilon_start - (self.epsilon_start - self.epsilon_end) *
                          (self.train_step / self.epsilon_decay))

    def train(self):
        """训练一步（Double DQN）"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # TODO 3: 实现 Double DQN 训练步骤
        # 提示：
        # 1. 从回放缓冲区采样 batch
        # 2. 转换为 PyTorch 张量
        # 3. 计算当前 Q 值：Q(s, a; θ)
        # 4. Double DQN 目标：
        #    a) 用在线网络选择动作：a* = argmax_a' Q(s', a'; θ)
        #    b) 用目标网络评估动作：y = r + γ Q(s', a*; θ⁻)
        # 5. 计算 MSE 损失
        # 6. 梯度下降更新在线网络
        # 7. 每隔 target_update_freq 步更新目标网络
        # 8. 更新 epsilon
        # ---- 在这里实现 ----
        # 采样 batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 计算当前 Q 值
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN 目标
        with torch.no_grad():
            # 用在线网络选择动作
            next_actions = self.online_net(next_states).argmax(1)
            # 用目标网络评估动作
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算损失
        loss = nn.MSELoss()(q_values, targets)

        # 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # 更新 epsilon
        self.update_epsilon()

        return loss.item()
        # ---- 代码结束 ----


# ============================================================
# Part 4: 训练和评估函数
# ============================================================

def train_agent(agent_type: str = 'double_dqn',
                env_name: str = 'CartPole-v1',
                num_episodes: int = 500,
                max_steps: int = 500,
                learning_rate: float = 1e-3,
                gamma: float = 0.99,
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.01,
                epsilon_decay: int = 10000,
                buffer_size: int = 10000,
                batch_size: int = 64,
                target_update_freq: int = 100):
    """
    训练 DQN 变体

    参数:
        agent_type: 'dqn', 'double_dqn', 'dueling_dqn', 'double_dueling_dqn'
        env_name: 环境名称
        num_episodes: 训练 episode 数
        max_steps: 每个 episode 最大步数
        其他参数：DQN 超参数

    返回:
        agent: 训练好的 agent
        episode_rewards: 每个 episode 的奖励
        episode_losses: 每个 episode 的平均损失
    """
    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 创建 agent
    if agent_type == 'dqn':
        # 标准 DQN（从 Lesson 13 导入）
        from lessons_13.exercises.lesson_13_exercise import DQNAgent
        agent = DQNAgent(state_dim, action_dim,
                        learning_rate=learning_rate,
                        gamma=gamma,
                        epsilon_start=epsilon_start,
                        epsilon_end=epsilon_end,
                        epsilon_decay=epsilon_decay,
                        buffer_size=buffer_size,
                        batch_size=batch_size,
                        target_update_freq=target_update_freq)
    elif agent_type == 'double_dqn':
        agent = DoubleDQNAgent(state_dim, action_dim,
                              learning_rate=learning_rate,
                              gamma=gamma,
                              epsilon_start=epsilon_start,
                              epsilon_end=epsilon_end,
                              epsilon_decay=epsilon_decay,
                              buffer_size=buffer_size,
                              batch_size=batch_size,
                              target_update_freq=target_update_freq,
                              use_dueling=False)
    elif agent_type == 'dueling_dqn':
        agent = DoubleDQNAgent(state_dim, action_dim,
                              learning_rate=learning_rate,
                              gamma=gamma,
                              epsilon_start=epsilon_start,
                              epsilon_end=epsilon_end,
                              epsilon_decay=epsilon_decay,
                              buffer_size=buffer_size,
                              batch_size=batch_size,
                              target_update_freq=target_update_freq,
                              use_dueling=True)
    elif agent_type == 'double_dueling_dqn':
        agent = DoubleDQNAgent(state_dim, action_dim,
                              learning_rate=learning_rate,
                              gamma=gamma,
                              epsilon_start=epsilon_start,
                              epsilon_end=epsilon_end,
                              epsilon_decay=epsilon_decay,
                              buffer_size=buffer_size,
                              batch_size=batch_size,
                              target_update_freq=target_update_freq,
                              use_dueling=True)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    episode_rewards = []
    episode_losses = []

    print("=" * 60)
    print(f"开始训练 {agent_type.upper()} on {env_name}")
    print("=" * 60)

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []

        for t in range(max_steps):
            # 选择动作
            action = agent.select_action(state, training=True)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # 训练
            loss = agent.train()
            if loss is not None:
                episode_loss.append(loss)

            state = next_state
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)

        # 打印进度
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"Avg Loss: {avg_loss:.4f}")

    env.close()
    return agent, episode_rewards, episode_losses


def evaluate_agent(agent, env_name: str = 'CartPole-v1',
                   num_episodes: int = 10, render: bool = False):
    """评估 agent"""
    env = gym.make(env_name, render_mode='human' if render else None)
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)

    env.close()

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    print(f"评估结果 ({num_episodes} episodes):")
    print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")

    return avg_reward, std_reward


# ============================================================
# Part 5: 对比实验
# ============================================================

def compare_algorithms():
    """对比不同 DQN 变体"""
    print("=" * 60)
    print("对比 DQN 变体")
    print("=" * 60)

    algorithms = ['dqn', 'double_dqn', 'dueling_dqn', 'double_dueling_dqn']
    results = {}

    for algo in algorithms:
        print(f"\n训练 {algo.upper()}...")
        agent, rewards, losses = train_agent(
            agent_type=algo,
            env_name='CartPole-v1',
            num_episodes=500,
            max_steps=500,
            learning_rate=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=10000,
            buffer_size=10000,
            batch_size=64,
            target_update_freq=100
        )
        results[algo] = {
            'agent': agent,
            'rewards': rewards,
            'losses': losses
        }

    # 绘制对比结果
    plot_comparison(results)

    # 评估所有算法
    print("\n" + "=" * 60)
    print("最终评估")
    print("=" * 60)
    for algo in algorithms:
        print(f"\n{algo.upper()}:")
        evaluate_agent(results[algo]['agent'], num_episodes=10)


def plot_comparison(results, window=50):
    """
    绘制对比结果

    参数:
        results: 字典，包含每个算法的训练结果
        window: 滑动平均窗口大小
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 绘制奖励曲线
    for algo, data in results.items():
        rewards = data['rewards']
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
            axes[0].plot(range(window - 1, len(rewards)), smoothed,
                        label=algo.upper(), alpha=0.8)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('奖励')
    axes[0].set_title('训练奖励对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 绘制损失曲线
    for algo, data in results.items():
        losses = data['losses']
        if len(losses) >= window:
            smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
            axes[1].plot(range(window - 1, len(losses)), smoothed,
                        label=algo.upper(), alpha=0.8)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('损失')
    axes[1].set_title('训练损失对比')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_14/dqn_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 运行对比实验
    compare_algorithms()

    print("\n所有实验完成！")

