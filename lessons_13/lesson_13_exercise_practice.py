"""
Lesson 13: DQN (Deep Q-Network)
练习代码 - 练习版本

本练习实现 DQN 算法，包括：
1. 深度 Q 网络（DQN）
2. 经验回放缓冲区（Replay Buffer）
3. 目标网络（Target Network）
4. ε-Greedy 探索策略
5. DQN 训练流程
6. 在 CartPole-v1 环境上训练和评估
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
# Part 1: 深度 Q 网络
# ============================================================

class DQN(nn.Module):
    """深度 Q 网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        初始化 DQN

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(DQN, self).__init__()
        # TODO 1: 实现 DQN 网络结构
        # 提示：
        # 1. 三层全连接网络：state_dim -> hidden_dim -> hidden_dim -> action_dim
        # 2. 前两层使用 ReLU 激活函数
        # 3. 输出层不使用激活函数（直接输出 Q 值）
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def forward(self, x):
        """
        前向传播

        参数:
            x: 状态张量，形状 (batch_size, state_dim)

        返回:
            q_values: Q 值张量，形状 (batch_size, action_dim)
        """
        # TODO 2: 实现前向传播
        # 提示：
        # 1. 通过三层网络传播
        # 2. 前两层使用 ReLU 激活
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----


# ============================================================
# Part 2: 经验回放缓冲区
# ============================================================

class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int):
        """
        初始化回放缓冲区

        参数:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        存储一个转移

        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
        """
        # TODO 3: 实现经验存储
        # 提示：
        # 1. 将转移 (state, action, reward, next_state, done) 存入 buffer
        # 2. deque 会自动处理容量限制
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def sample(self, batch_size: int):
        """
        随机采样一个 batch

        参数:
            batch_size: batch 大小

        返回:
            states, actions, rewards, next_states, dones: 采样的 batch
        """
        # TODO 4: 实现随机采样
        # 提示：
        # 1. 使用 random.sample 从 buffer 中随机采样
        # 2. 将采样结果解包成 5 个数组
        # 3. 转换为 numpy 数组
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)


# ============================================================
# Part 3: DQN Agent
# ============================================================

class DQNAgent:
    """DQN 智能体"""

    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 10000,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 100):
        """
        初始化 DQN Agent

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

        # 在线网络和目标网络
        self.online_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # 目标网络不需要梯度

        # 优化器
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)

        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)

        # 训练步数
        self.train_step = 0

    def select_action(self, state, training=True):
        """
        选择动作（ε-greedy）

        参数:
            state: 当前状态
            training: 是否处于训练模式

        返回:
            action: 选择的动作
        """
        # TODO 5: 实现 ε-greedy 动作选择
        # 提示：
        # 1. 如果 training=False，直接选择最优动作（贪婪）
        # 2. 如果 training=True，以概率 epsilon 随机选择动作
        # 3. 否则选择 Q 值最大的动作
        # 4. 使用 torch.no_grad() 避免计算梯度
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def update_epsilon(self):
        """更新探索率（线性衰减）"""
        # TODO 6: 实现 epsilon 线性衰减
        # 提示：
        # 1. epsilon = epsilon_start - (epsilon_start - epsilon_end) * (train_step / epsilon_decay)
        # 2. 确保 epsilon >= epsilon_end
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def train(self):
        """训练一步"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # TODO 7: 实现 DQN 训练步骤
        # 提示：
        # 1. 从回放缓冲区采样 batch
        # 2. 转换为 PyTorch 张量
        # 3. 计算当前 Q 值：Q(s, a; θ)
        # 4. 计算目标 Q 值：y = r + γ max_a' Q(s', a'; θ⁻)
        # 5. 注意终止状态：如果 done=True，则 y = r
        # 6. 计算 MSE 损失
        # 7. 梯度下降更新在线网络
        # 8. 每隔 target_update_freq 步更新目标网络
        # 9. 更新 epsilon
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----


# ============================================================
# Part 4: 训练和评估函数
# ============================================================

def train_dqn(env_name: str = 'CartPole-v1',
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
    训练 DQN

    参数:
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
    agent = DQNAgent(state_dim, action_dim,
                     learning_rate=learning_rate,
                     gamma=gamma,
                     epsilon_start=epsilon_start,
                     epsilon_end=epsilon_end,
                     epsilon_decay=epsilon_decay,
                     buffer_size=buffer_size,
                     batch_size=batch_size,
                     target_update_freq=target_update_freq)

    episode_rewards = []
    episode_losses = []

    print("=" * 60)
    print(f"开始训练 DQN on {env_name}")
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
    """
    评估 agent

    参数:
        agent: 训练好的 agent
        env_name: 环境名称
        num_episodes: 评估 episode 数
        render: 是否渲染

    返回:
        avg_reward: 平均奖励
        std_reward: 奖励标准差
    """
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

    print(f"\n评估结果 ({num_episodes} episodes):")
    print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")

    return avg_reward, std_reward


def plot_training_results(episode_rewards, episode_losses, window=50):
    """
    绘制训练结果

    参数:
        episode_rewards: episode 奖励列表
        episode_losses: episode 损失列表
        window: 滑动平均窗口大小
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 绘制奖励曲线
    axes[0].plot(episode_rewards, alpha=0.3, label='原始奖励')
    if len(episode_rewards) >= window:
        smoothed_rewards = np.convolve(episode_rewards,
                                       np.ones(window) / window,
                                       mode='valid')
        axes[0].plot(range(window - 1, len(episode_rewards)),
                    smoothed_rewards, label=f'{window}-episode 平均')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('奖励')
    axes[0].set_title('训练奖励曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 绘制损失曲线
    axes[1].plot(episode_losses, alpha=0.3, label='原始损失')
    if len(episode_losses) >= window:
        smoothed_losses = np.convolve(episode_losses,
                                      np.ones(window) / window,
                                      mode='valid')
        axes[1].plot(range(window - 1, len(episode_losses)),
                    smoothed_losses, label=f'{window}-episode 平均')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('损失')
    axes[1].set_title('训练损失曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_13/dqn_training.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 训练 DQN
    agent, episode_rewards, episode_losses = train_dqn(
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

    # 绘制训练结果
    plot_training_results(episode_rewards, episode_losses)

    # 评估 agent
    evaluate_agent(agent, env_name='CartPole-v1', num_episodes=10)

    print("\n训练完成！")
