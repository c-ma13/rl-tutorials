"""
Lesson 15: PPO (上) - Proximal Policy Optimization
练习代码 - 练习版本

本练习实现 PPO-Clip 算法，包括：
1. 策略网络（连续动作空间）
2. 价值网络
3. GAE (Generalized Advantage Estimation)
4. PPO-Clip 损失函数
5. 在 CartPole-v1 环境上训练
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
# Part 1: 策略网络
# ============================================================

class PolicyNetwork(nn.Module):
    """策略网络（离散动作空间）"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        初始化策略网络

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(PolicyNetwork, self).__init__()
        # TODO 1: 实现策略网络结构
        # 提示：
        # 1. 两层全连接网络：state_dim -> hidden_dim -> hidden_dim -> action_dim
        # 2. 使用 Tanh 激活函数
        # 3. 输出层不使用激活函数（输出 logits）
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def forward(self, state):
        """
        前向传播

        参数:
            state: 状态张量，形状 (batch_size, state_dim)

        返回:
            logits: 动作 logits，形状 (batch_size, action_dim)
        """
        # TODO 2: 实现前向传播
        # 提示：
        # 1. 通过网络得到 logits
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def get_action(self, state):
        """
        采样动作

        参数:
            state: 状态张量，形状 (batch_size, state_dim)

        返回:
            action: 采样的动作
            log_prob: 动作的 log 概率
        """
        # TODO 3: 实现动作采样
        # 提示：
        # 1. 通过前向传播得到 logits
        # 2. 创建 Categorical 分布
        # 3. 采样动作
        # 4. 计算 log 概率
        # ---- 在这里实现 ----
        pass
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
        # 1. 通过前向传播得到 logits
        # 2. 创建 Categorical 分布
        # 3. 计算给定动作的 log 概率
        # 4. 计算策略熵
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----


# ============================================================
# Part 2: 价值网络
# ============================================================

class ValueNetwork(nn.Module):
    """价值网络"""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        """
        初始化价值网络

        参数:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
        """
        super(ValueNetwork, self).__init__()
        # TODO 5: 实现价值网络结构
        # 提示：
        # 1. 两层全连接网络：state_dim -> hidden_dim -> hidden_dim -> 1
        # 2. 使用 Tanh 激活函数
        # 3. 输出层不使用激活函数（输出状态价值）
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def forward(self, state):
        """
        前向传播

        参数:
            state: 状态张量，形状 (batch_size, state_dim)

        返回:
            value: 状态价值，形状 (batch_size, 1)
        """
        return self.fc(state)


# ============================================================
# Part 3: PPO Agent
# ============================================================

class PPOAgent:
    """PPO 智能体"""

    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        """
        初始化 PPO Agent

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            gamma: 折扣因子
            gae_lambda: GAE 参数
            clip_epsilon: PPO 裁剪参数
            value_coef: 价值函数损失系数
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度裁剪阈值
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # 创建网络
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)

        # 优化器
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=learning_rate
        )

    def compute_gae(self, rewards, values, dones, next_value):
        """
        计算 GAE (Generalized Advantage Estimation)

        参数:
            rewards: 奖励列表
            values: 状态价值列表
            dones: 终止标志列表
            next_value: 最后一个状态的价值

        返回:
            advantages: 优势函数
            returns: 回报
        """
        # TODO 6: 实现 GAE 计算
        # 提示：
        # 1. 从后向前计算 TD 误差：δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        # 2. 计算 GAE：A_t = δ_t + γλ δ_{t+1} + (γλ)² δ_{t+2} + ...
        # 3. 计算回报：R_t = A_t + V(s_t)
        # 4. 注意处理终止状态（done=True 时，V(s_{t+1}) = 0）
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----

    def update(self, states, actions, old_log_probs, advantages, returns, epochs=10, batch_size=64):
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
        # TODO 7: 实现 PPO 更新
        # 提示：
        # 1. 优势函数归一化
        # 2. 转换为张量
        # 3. 进行 K 轮更新
        # 4. 每轮随机打乱数据，分 mini-batch 更新
        # 5. 计算 PPO-Clip 损失、价值函数损失、熵正则化
        # 6. 梯度下降更新
        # ---- 在这里实现 ----
        pass
        # ---- 代码结束 ----


# ============================================================
# Part 4: 训练函数
# ============================================================

def train_ppo(env_name: str = 'CartPole-v1',
              num_iterations: int = 200,
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
    训练 PPO

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
    action_dim = env.action_space.n

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
    print(f"开始训练 PPO on {env_name}")
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

            action_val = action.item()
            log_prob_val = log_prob.item()
            value_val = value.item()

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action_val)
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
            next_value = agent.value_net(torch.FloatTensor(state).unsqueeze(0)).squeeze().item()

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
                  f"Value Loss: {np.mean(value_losses):.4f}")

    env.close()
    return agent, iteration_rewards


def evaluate_agent(agent, env_name: str = 'CartPole-v1', num_episodes: int = 10):
    """评估 agent"""
    env = gym.make(env_name)
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _ = agent.policy_net.get_action(state_tensor)
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)

    env.close()

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\n评估结果 ({num_episodes} episodes):")
    print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    return avg_reward, std_reward


def plot_training_results(iteration_rewards, window=10):
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
    plt.title('PPO 训练曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_15/ppo_training.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 训练 PPO
    agent, iteration_rewards = train_ppo(
        env_name='CartPole-v1',
        num_iterations=200,
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
    evaluate_agent(agent, env_name='CartPole-v1', num_episodes=10)

    print("\n训练完成！")
