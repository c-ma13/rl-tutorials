"""
Lesson 17: DDPG & TD3
练习代码 - 参考实现

本练习实现 TD3 算法，包括：
1. Actor 网络（确定性策略）
2. Critic 网络（Q 函数）
3. Twin Critics（双 Critic）
4. Delayed Policy Update（延迟策略更新）
5. Target Policy Smoothing（目标策略平滑）
6. 在 Pendulum-v1 环境上训练
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
# Part 1: Actor 网络（确定性策略）
# ============================================================

class Actor(nn.Module):
    """Actor 网络：输出确定性动作"""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        """
        初始化 Actor 网络

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            max_action: 动作的最大值
        """
        super(Actor, self).__init__()
        self.max_action = max_action
        # TODO 1: 实现 Actor 网络结构
        # 提示：
        # 1. 三层全连接网络：state_dim -> 256 -> 256 -> action_dim
        # 2. 前两层使用 ReLU 激活
        # 3. 输出层使用 tanh 激活（限制到 [-1, 1]）
        # 4. 乘以 max_action 缩放到实际范围
        # ---- 在这里实现 ----
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        # ---- 代码结束 ----

    def forward(self, state):
        """
        前向传播

        参数:
            state: 状态张量

        返回:
            action: 确定性动作
        """
        # TODO 2: 实现前向传播
        # 提示：
        # 1. 通过三层网络
        # 2. 输出层使用 tanh，然后乘以 max_action
        # ---- 在这里实现 ----
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))
        # ---- 代码结束 ----


# ============================================================
# Part 2: Critic 网络
# ============================================================

class Critic(nn.Module):
    """Critic 网络：评估 Q(s, a)"""

    def __init__(self, state_dim: int, action_dim: int):
        """
        初始化 Critic 网络

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
        """
        super(Critic, self).__init__()
        # TODO 3: 实现 Critic 网络结构
        # 提示：
        # 1. 输入是状态和动作的拼接：(state_dim + action_dim)
        # 2. 三层全连接网络：(state_dim + action_dim) -> 256 -> 256 -> 1
        # 3. 前两层使用 ReLU 激活
        # 4. 输出层不使用激活函数（输出 Q 值）
        # ---- 在这里实现 ----
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        # ---- 代码结束 ----

    def forward(self, state, action):
        """
        前向传播

        参数:
            state: 状态张量
            action: 动作张量

        返回:
            q_value: Q 值
        """
        # TODO 4: 实现前向传播
        # 提示：
        # 1. 拼接状态和动作
        # 2. 通过三层网络
        # ---- 在这里实现 ----
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
        # ---- 代码结束 ----


# ============================================================
# Part 3: 经验回放缓冲区
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
# Part 4: TD3 Agent
# ============================================================

class TD3Agent:
    """TD3 智能体"""

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_delay: int = 2,
                 buffer_size: int = 1000000,
                 batch_size: int = 256):
        """
        初始化 TD3 Agent

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            max_action: 动作最大值
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            gamma: 折扣因子
            tau: 软更新系数
            policy_noise: 目标策略平滑噪声
            noise_clip: 噪声裁剪范围
            policy_delay: Actor 更新延迟
            buffer_size: 回放缓冲区大小
            batch_size: 批次大小
        """
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.max_action = max_action

        # Actor 网络
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin Critics
        self.critic1 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr
        )

        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)

        # 训练步数
        self.total_steps = 0

    def select_action(self, state, noise_scale=0.1):
        """
        选择动作

        参数:
            state: 当前状态
            noise_scale: 探索噪声标准差

        返回:
            action: 选择的动作
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]

        # 添加探索噪声
        if noise_scale > 0:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def train(self):
        """训练一步"""
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        self.total_steps += 1

        # TODO 5: 实现 TD3 训练步骤
        # 提示：
        # 1. 从回放缓冲区采样 batch
        # 2. 转换为 PyTorch 张量
        # 3. 计算目标动作（带噪声）：
        #    a' = μ_θ'(s') + clip(ε, -c, c)
        # 4. 计算目标 Q 值（取两个 Critic 的最小值）：
        #    y = r + γ min(Q_φ₁'(s', a'), Q_φ₂'(s', a'))
        # 5. 更新两个 Critic
        # 6. 每隔 policy_delay 步：
        #    a) 更新 Actor
        #    b) 软更新目标网络
        # ---- 在这里实现 ----
        # 采样 batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        with torch.no_grad():
            # 目标动作（带噪声）
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, -self.max_action, self.max_action)

            # 目标 Q 值（取最小值）
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * target_q * (1 - dones)

        # 更新 Critic
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None

        # 延迟更新 Actor
        if self.total_steps % self.policy_delay == 0:
            # 更新 Actor
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            actor_loss = actor_loss.item()

        return critic_loss.item(), actor_loss
        # ---- 代码结束 ----


# ============================================================
# Part 5: 训练函数
# ============================================================

def train_td3(env_name: str = 'Pendulum-v1',
              num_episodes: int = 500,
              max_steps: int = 1000,
              warmup_steps: int = 10000,
              noise_scale: float = 0.1,
              **kwargs):
    """
    训练 TD3

    参数:
        env_name: 环境名称
        num_episodes: 训练 episode 数
        max_steps: 每个 episode 最大步数
        warmup_steps: 预热步数（随机探索）
        noise_scale: 探索噪声标准差
        **kwargs: TD3 超参数

    返回:
        agent: 训练好的 agent
        episode_rewards: 每个 episode 的奖励
    """
    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 创建 agent
    agent = TD3Agent(state_dim, action_dim, max_action, **kwargs)

    episode_rewards = []
    total_steps = 0

    print("=" * 60)
    print(f"开始训练 TD3 on {env_name}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"动作范围: [-{max_action}, {max_action}]")
    print("=" * 60)

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            total_steps += 1

            # 预热阶段：随机探索
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise_scale)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # 训练
            if total_steps >= warmup_steps:
                critic_loss, actor_loss = agent.train()

            state = next_state
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)

        # 打印进度
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Steps: {total_steps}")

    env.close()
    return agent, episode_rewards


def evaluate_agent(agent, env_name: str = 'Pendulum-v1', num_episodes: int = 10):
    """评估 agent"""
    env = gym.make(env_name)
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, noise_scale=0)  # 评估时不加噪声
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


def plot_training_results(episode_rewards, window=20):
    """绘制训练结果"""
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.3, label='原始奖励')
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards,
                               np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(episode_rewards)),
                smoothed, label=f'{window}-episode 平均')
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.title('TD3 训练曲线 (Pendulum-v1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_17/td3_pendulum.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 训练 TD3
    agent, episode_rewards = train_td3(
        env_name='Pendulum-v1',
        num_episodes=500,
        max_steps=1000,
        warmup_steps=10000,
        noise_scale=0.1,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        buffer_size=1000000,
        batch_size=256
    )

    # 绘制训练结果
    plot_training_results(episode_rewards)

    # 评估 agent
    evaluate_agent(agent, env_name='Pendulum-v1', num_episodes=10)

    print("\n训练完成！")
