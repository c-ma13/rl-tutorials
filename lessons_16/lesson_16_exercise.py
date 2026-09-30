"""
Lesson 16: PPO (下) - 实现细节与实践
练习代码 - 参考实现

本练习实现连续动作空间的 PPO，包括：
1. 高斯策略网络（连续动作）
2. 价值网络
3. GAE 计算
4. PPO-Clip 损失
5. 在 Pendulum-v1 环境上训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: 高斯策略网络（连续动作）
# ============================================================

class GaussianPolicy(nn.Module):
    """高斯策略网络（连续动作空间）"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        初始化高斯策略网络

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(GaussianPolicy, self).__init__()
        # TODO 1: 实现高斯策略网络结构
        # 提示：
        # 1. 共享特征提取层：state_dim -> hidden_dim -> hidden_dim
        # 2. 均值层：hidden_dim -> action_dim
        # 3. log 标准差：可学习参数，形状 (action_dim,)
        # 4. 使用 Tanh 激活函数
        # ---- 在这里实现 ----
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # ---- 代码结束 ----

    def forward(self, state):
        """
        前向传播

        参数:
            state: 状态张量，形状 (batch_size, state_dim)

        返回:
            mean: 动作均值，形状 (batch_size, action_dim)
            std: 动作标准差，形状 (batch_size, action_dim)
        """
        # TODO 2: 实现前向传播
        # 提示：
        # 1. 通过共享层提取特征
        # 2. 计算均值
        # 3. 计算标准差：std = exp(log_std)
        # ---- 在这里实现 ----
        x = self.fc(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std
        # ---- 代码结束 ----

    def get_action(self, state):
        """
        采样动作

        参数:
            state: 状态张量

        返回:
            action: 采样的动作
            log_prob: 动作的 log 概率
        """
        # TODO 3: 实现动作采样
        # 提示：
        # 1. 通过前向传播得到均值和标准差
        # 2. 创建 Normal 分布
        # 3. 采样动作
        # 4. 计算 log 概率（多维动作需要求和）
        # ---- 在这里实现 ----
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
        # ---- 代码结束 ----

    def evaluate_actions(self, state, action):
        """
        评估动作

        参数:
            state: 状态张量
            action: 动作张量

        返回:
            log_prob: 动作的 log 概率
            entropy: 策略熵
        """
        # TODO 4: 实现动作评估
        # 提示：
        # 1. 通过前向传播得到均值和标准差
        # 2. 创建 Normal 分布
        # 3. 计算给定动作的 log 概率（多维求和）
        # 4. 计算策略熵（多维求和）
        # ---- 在这里实现 ----
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
        # ---- 代码结束 ----


# ============================================================
# Part 2: 价值网络
# ============================================================

class ValueNetwork(nn.Module):
    """价值网络"""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.fc(state)


# ============================================================
# Part 3: PPO Agent（连续动作）
# ============================================================

class PPOAgent:
    """PPO 智能体（连续动作空间）"""

    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # 创建网络
        self.policy_net = GaussianPolicy(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)

        # 优化器
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=learning_rate
        )

    def compute_gae(self, rewards, values, dones, next_value):
        """计算 GAE"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values, dtype=torch.float32)

        return advantages, returns

    def update(self, states, actions, old_log_probs, advantages, returns,
               epochs=10, batch_size=64):
        """
        更新策略和价值网络

        参数:
            states: 状态列表
            actions: 动作列表
            old_log_probs: 旧策略的 log 概率
            advantages: 优势函数
            returns: 回报
            epochs: 更新轮数
            batch_size: mini-batch 大小

        返回:
            policy_losses: 策略损失列表
            value_losses: 价值损失列表
        """
        # TODO 5: 实现 PPO 更新（连续动作版本）
        # 提示：
        # 1. 优势函数归一化
        # 2. 转换为张量（注意 actions 是连续值）
        # 3. 进行 K 轮更新
        # 4. 每轮随机打乱数据，分 mini-batch 更新
        # 5. 计算 PPO-Clip 损失、价值函数损失、熵正则化
        # 6. 梯度下降更新
        # ---- 在这里实现 ----
        # 优势函数归一化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转换为张量
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(np.array(old_log_probs))
        returns = torch.FloatTensor(returns)

        policy_losses = []
        value_losses = []

        # K 轮更新
        for _ in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(states))

            # Mini-batch 更新
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 评估动作
                new_log_probs, entropy = self.policy_net.evaluate_actions(
                    batch_states, batch_actions
                )

                # 重要性采样比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO-Clip 损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                   1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值函数损失
                values = self.value_net(batch_states).squeeze()
                value_loss = (values - batch_returns).pow(2).mean()

                # 熵正则化
                entropy_loss = -entropy.mean()

                # 总损失
                loss = (policy_loss + self.value_coef * value_loss +
                       self.entropy_coef * entropy_loss)

                # 梯度下降
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) +
                    list(self.value_net.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        return policy_losses, value_losses
        # ---- 代码结束 ----


# ============================================================
# Part 4: 训练函数
# ============================================================

def train_ppo_continuous(env_name: str = 'Pendulum-v1',
                         num_iterations: int = 300,
                         steps_per_iteration: int = 2048,
                         epochs: int = 10,
                         batch_size: int = 64,
                         learning_rate: float = 3e-4,
                         gamma: float = 0.99,
                         gae_lambda: float = 0.95,
                         clip_epsilon: float = 0.2,
                         value_coef: float = 0.5,
                         entropy_coef: float = 0.01,
                         max_grad_norm: float = 0.5):
    """
    训练连续动作空间的 PPO

    参数:
        env_name: 环境名称
        num_iterations: 迭代次数
        steps_per_iteration: 每次迭代采样步数
        epochs: 每次迭代更新轮数
        batch_size: mini-batch 大小
        其他参数：PPO 超参数

    返回:
        agent: 训练好的 agent
        iteration_rewards: 每次迭代的平均奖励
    """
    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    # 创建 agent
    agent = PPOAgent(state_dim, action_dim,
                     learning_rate=learning_rate,
                     gamma=gamma,
                     gae_lambda=gae_lambda,
                     clip_epsilon=clip_epsilon,
                     value_coef=value_coef,
                     entropy_coef=entropy_coef,
                     max_grad_norm=max_grad_norm)

    iteration_rewards = []

    print("=" * 60)
    print(f"开始训练 PPO (连续动作) on {env_name}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"动作范围: [{action_low}, {action_high}]")
    print("=" * 60)

    for iteration in range(num_iterations):
        # 收集数据
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        episode_rewards = []
        episode_reward = 0

        state, _ = env.reset()

        for step in range(steps_per_iteration):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # 采样动作
            with torch.no_grad():
                action, log_prob = agent.policy_net.get_action(state_tensor)
                value = agent.value_net(state_tensor).squeeze()

            action_val = action.squeeze(0).numpy()
            log_prob_val = log_prob.item()
            value_val = value.item()

            # 裁剪动作到环境范围
            clipped_action = np.clip(action_val, action_low, action_high)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(clipped_action)
            done = terminated or truncated

            # 存储数据
            states.append(state)
            actions.append(action_val)
            rewards.append(reward)
            dones.append(float(done))
            log_probs.append(log_prob_val)
            values.append(value_val)

            episode_reward += reward
            state = next_state

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                state, _ = env.reset()

        # 计算最后一个状态的价值
        with torch.no_grad():
            next_value = agent.value_net(
                torch.FloatTensor(state).unsqueeze(0)
            ).squeeze().item()

        # 计算 GAE
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

        # 更新策略和价值网络
        policy_losses, value_losses = agent.update(
            states, actions, log_probs, advantages, returns,
            epochs=epochs, batch_size=batch_size
        )

        # 记录结果
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        iteration_rewards.append(avg_reward)

        # 打印进度
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Policy Loss: {np.mean(policy_losses):.4f}, "
                  f"Value Loss: {np.mean(value_losses):.4f}, "
                  f"Std: {torch.exp(agent.policy_net.log_std).detach().numpy()}")

    env.close()
    return agent, iteration_rewards


def evaluate_agent(agent, env_name: str = 'Pendulum-v1', num_episodes: int = 10):
    """评估 agent"""
    env = gym.make(env_name)
    action_low = env.action_space.low
    action_high = env.action_space.high
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                mean, _ = agent.policy_net(state_tensor)
                action = mean.squeeze(0).numpy()  # 评估时用均值
            action = np.clip(action, action_low, action_high)
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


def plot_training_results(iteration_rewards, window=20):
    """绘制训练结果"""
    plt.figure(figsize=(10, 5))
    plt.plot(iteration_rewards, alpha=0.3, label='原始奖励')
    if len(iteration_rewards) >= window:
        smoothed = np.convolve(iteration_rewards,
                               np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(iteration_rewards)),
                smoothed, label=f'{window}-iteration 平均')
    plt.xlabel('Iteration')
    plt.ylabel('平均奖励')
    plt.title('PPO 连续动作训练曲线 (Pendulum-v1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_16/ppo_pendulum.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 训练 PPO（连续动作空间）
    agent, iteration_rewards = train_ppo_continuous(
        env_name='Pendulum-v1',
        num_iterations=300,
        steps_per_iteration=2048,
        epochs=10,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5
    )

    # 绘制训练结果
    plot_training_results(iteration_rewards)

    # 评估 agent
    evaluate_agent(agent, env_name='Pendulum-v1', num_episodes=10)

    print("\n训练完成！")
