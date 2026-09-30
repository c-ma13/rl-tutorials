"""
Lesson 18: SAC (Soft Actor-Critic)
练习代码 - 练习版本

本练习实现 SAC 算法，包括：
1. 高斯 Actor 网络（随机策略）
2. Critic 网络（Twin Q-networks）
3. Squashed Gaussian（压缩高斯）
4. 重参数化技巧
5. 自动温度调节
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
# Part 1: 高斯 Actor 网络（随机策略）
# ============================================================

class GaussianActor(nn.Module):
    """高斯 Actor 网络：输出动作的均值和标准差"""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        """
        初始化高斯 Actor 网络

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            max_action: 动作的最大值
        """
        super(GaussianActor, self).__init__()
        self.max_action = max_action
        # TODO 1: 实现高斯 Actor 网络结构
        # 提示：
        # 1. 共享特征层：state_dim -> 256 -> 256
        # 2. 均值层：256 -> action_dim
        # 3. log 标准差层：256 -> action_dim
        # 4. 使用 ReLU 激活
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def forward(self, state):
        """
        前向传播

        参数:
            state: 状态张量

        返回:
            mean: 动作均值
            log_std: 动作 log 标准差
        """
        x = self.fc(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        # 限制 log_std 范围，避免数值不稳定
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        """
        采样动作（Squashed Gaussian）

        参数:
            state: 状态张量

        返回:
            action: 采样的动作（经过 tanh 压缩）
            log_prob: 动作的 log 概率
        """
        # TODO 2: 实现 Squashed Gaussian 采样
        # 提示：
        # 1. 通过前向传播得到均值和 log 标准差
        # 2. 计算标准差：std = exp(log_std)
        # 3. 重参数化采样：x = mean + std * epsilon（使用 normal.rsample()）
        # 4. tanh 压缩：action = tanh(x) * max_action
        # 5. 计算 log 概率并修正：log_prob = log_prob(x) - log(1 - tanh²(x))
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----


# ============================================================
# Part 2: Critic 网络
# ============================================================

class Critic(nn.Module):
    """Critic 网络：评估 Q(s, a)"""

    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.fc(x)


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
# Part 4: SAC Agent
# ============================================================

class SACAgent:
    """SAC 智能体"""

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 alpha_lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 auto_alpha: bool = True,
                 init_alpha: float = 0.2,
                 buffer_size: int = 1000000,
                 batch_size: int = 256):
        """
        初始化 SAC Agent

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            max_action: 动作最大值
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            alpha_lr: 温度参数学习率
            gamma: 折扣因子
            tau: 软更新系数
            auto_alpha: 是否自动调节温度
            init_alpha: 初始温度
            buffer_size: 回放缓冲区大小
            batch_size: 批次大小
        """
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.auto_alpha = auto_alpha

        # Actor 网络
        self.actor = GaussianActor(state_dim, action_dim, max_action)
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

        # 温度参数
        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = init_alpha

        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, deterministic=False):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state)
                action = torch.tanh(mean) * self.max_action
            else:
                action, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]

    def train(self):
        """训练一步"""
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None

        # TODO 3: 实现 SAC 训练步骤
        # 提示：
        # 1. 从回放缓冲区采样 batch
        # 2. 转换为 PyTorch 张量
        # 3. 更新 Critic：
        #    a) 采样下一个动作：a' ~ π(·|s')
        #    b) 计算目标 Q 值：y = r + γ (min(Q₁'(s',a'), Q₂'(s',a')) - α log π(a'|s'))
        #    c) 更新两个 Critic
        # 4. 更新 Actor：
        #    a) 采样动作：a ~ π(·|s)
        #    b) 计算损失：L_π = E[α log π(a|s) - Q₁(s, a)]
        # 5. 更新温度参数 α（如果 auto_alpha=True）：
        #    L_α = E[-α (log π(a|s) + target_entropy)]
        # 6. 软更新目标网络
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----


# ============================================================
# Part 5: 训练函数
# ============================================================

def train_sac(env_name: str = 'Pendulum-v1',
              num_episodes: int = 500,
              max_steps: int = 1000,
              warmup_steps: int = 10000,
              **kwargs):
    """
    训练 SAC

    参数:
        env_name: 环境名称
        num_episodes: 训练 episode 数
        max_steps: 每个 episode 最大步数
        warmup_steps: 预热步数
        **kwargs: SAC 超参数

    返回:
        agent: 训练好的 agent
        episode_rewards: 每个 episode 的奖励
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SACAgent(state_dim, action_dim, max_action, **kwargs)

    episode_rewards = []
    total_steps = 0

    print("=" * 60)
    print(f"开始训练 SAC on {env_name}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print("=" * 60)

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            total_steps += 1

            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)

            if total_steps >= warmup_steps:
                agent.train()

            state = next_state
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            alpha_val = agent.log_alpha.exp().item() if agent.auto_alpha else agent.alpha
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Alpha: {alpha_val:.4f}")

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
            action = agent.select_action(state, deterministic=True)
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
    plt.title('SAC 训练曲线 (Pendulum-v1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_18/sac_pendulum.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    agent, episode_rewards = train_sac(
        env_name='Pendulum-v1',
        num_episodes=500,
        max_steps=1000,
        warmup_steps=10000,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        auto_alpha=True,
        buffer_size=1000000,
        batch_size=256
    )

    plot_training_results(episode_rewards)
    evaluate_agent(agent, env_name='Pendulum-v1', num_episodes=10)

    print("\n训练完成！")
