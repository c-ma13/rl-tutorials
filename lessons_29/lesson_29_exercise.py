"""
Lesson 29: 在线 RL 与 LLM Agent
练习代码 - 参考实现

本练习构建简单的 LLM Agent + RL 反馈循环，包括：
1. 简化的任务环境（数学问题求解）
2. 工具调用系统（计算器、搜索）
3. Agent 策略（基于规则 + 可学习）
4. 在线 RL 训练（REINFORCE）
5. Agent 性能评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: 任务环境（无 TODO）
# ============================================================

class MathTaskEnv:
    """
    简化的数学任务环境

    任务: 回答数学问题，可以使用工具
    工具: calculator(expression), search(query)
    """

    def __init__(self):
        self.problems = [
            {"question": "计算 15 * 23", "answer": 345, "needs_calc": True},
            {"question": "计算 100 / 4", "answer": 25, "needs_calc": True},
            {"question": "计算 7 + 8 * 3", "answer": 31, "needs_calc": True},
            {"question": "什么是质数", "answer": "只能被1和自身整除的数", "needs_calc": False},
        ]
        self.current_problem = None
        self.step_count = 0
        self.max_steps = 5

    def reset(self):
        """开始新任务"""
        self.current_problem = np.random.choice(self.problems)
        self.step_count = 0
        return self.current_problem["question"]

    def step(self, action):
        """
        执行动作

        action: dict with 'type' and 'content'
            type: 'calculator', 'search', 'answer'
            content: expression/query/answer

        返回: observation, reward, done, info
        """
        self.step_count += 1
        done = False
        reward = 0.0
        observation = ""

        if action['type'] == 'calculator':
            try:
                result = eval(action['content'])
                observation = f"计算结果: {result}"
                reward = 0.1  # 使用工具的小奖励
            except:
                observation = "计算错误"
                reward = -0.1

        elif action['type'] == 'search':
            # 模拟搜索
            if "质数" in action['content']:
                observation = "质数是只能被1和自身整除的正整数"
                reward = 0.1
            else:
                observation = "未找到相关信息"
                reward = -0.05

        elif action['type'] == 'answer':
            done = True
            # 检查答案
            correct_answer = self.current_problem["answer"]
            if isinstance(correct_answer, (int, float)):
                try:
                    user_answer = float(action['content'])
                    if abs(user_answer - correct_answer) < 0.01:
                        reward = 1.0
                        observation = "正确！"
                    else:
                        reward = -0.5
                        observation = f"错误，正确答案是 {correct_answer}"
                except:
                    reward = -0.5
                    observation = "答案格式错误"
            else:
                if correct_answer in action['content']:
                    reward = 1.0
                    observation = "正确！"
                else:
                    reward = -0.5
                    observation = f"错误，正确答案是 {correct_answer}"

        # 步数惩罚
        reward -= 0.02

        if self.step_count >= self.max_steps:
            done = True
            if reward < 0.5:  # 未正确回答
                reward = -0.5

        return observation, reward, done, {}


# ============================================================
# Part 2: Agent 策略网络（TODO 1）
# ============================================================

class AgentPolicy(nn.Module):
    """
    Agent 策略网络

    输入: 问题和历史的编码
    输出: 动作分布（选择工具或回答）
    """

    def __init__(self, state_dim=20, hidden_dim=64, num_actions=3):
        """
        num_actions: 3 (calculator, search, answer)
        """
        super().__init__()
        # TODO 1: 实现策略网络
        # 提示:
        # 1. 构建网络: state_dim -> hidden_dim -> hidden_dim -> num_actions
        # 2. 使用 ReLU 激活
        # 3. 输出 logits（不需要 softmax，后面会用）
        # ---- 在这里实现 ----
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        # ---- 代码结束 ----

    def forward(self, state):
        """前向传播，返回动作 logits"""
        return self.network(state)

    def select_action(self, state, epsilon=0.1):
        """
        选择动作（epsilon-greedy）

        返回: action_idx, log_prob
        """
        if np.random.random() < epsilon:
            # 随机探索
            action_idx = np.random.randint(0, 3)
            logits = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            log_prob = torch.log(probs[action_idx] + 1e-8)
            return action_idx, log_prob
        else:
            # 策略选择
            logits = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob


# ============================================================
# Part 3: 状态编码（无 TODO）
# ============================================================

def encode_state(question, history, state_dim=20):
    """
    将问题和历史编码为固定维度向量

    简化版本: 使用问题长度、历史长度、关键词等特征
    """
    features = np.zeros(state_dim)

    # 问题特征
    features[0] = len(question) / 100.0
    features[1] = 1.0 if '计算' in question else 0.0
    features[2] = 1.0 if '什么' in question else 0.0
    features[3] = 1.0 if any(op in question for op in ['+', '-', '*', '/']) else 0.0

    # 历史特征
    features[4] = len(history) / 10.0
    features[5] = sum(1 for h in history if h['type'] == 'calculator') / 5.0
    features[6] = sum(1 for h in history if h['type'] == 'search') / 5.0

    # 随机填充剩余维度（模拟更复杂的编码）
    features[7:] = np.random.randn(state_dim - 7) * 0.1

    return features.astype(np.float32)


# ============================================================
# Part 4: Agent 执行（TODO 2）
# ============================================================

def execute_agent_episode(env, policy, epsilon=0.1):
    """
    执行一个 episode

    返回: trajectory, total_reward
        trajectory: [(state, action_idx, log_prob, reward), ...]
    """
    # TODO 2: 实现 Agent episode 执行
    # 提示:
    # 1. env.reset() 获取问题
    # 2. 初始化 history = []
    # 3. 循环直到 done:
    #    a) encode_state(question, history)
    #    b) policy.select_action(state, epsilon)
    #    c) 根据 action_idx 构造 action dict (0=calc, 1=search, 2=answer)
    #       - 对于 calc/search: 生成简单的 content（如 question 本身）
    #       - 对于 answer: 尝试从 history 中提取答案或随机猜测
    #    d) env.step(action)
    #    e) 记录到 trajectory 和 history
    # 4. 返回 trajectory, total_reward
    # ---- 在这里实现 ----
    question = env.reset()
    history = []
    trajectory = []
    total_reward = 0.0
    done = False

    while not done:
        # 编码当前状态
        state = encode_state(question, history)
        state_tensor = torch.FloatTensor(state)

        # 选择动作
        action_idx, log_prob = policy.select_action(state_tensor, epsilon)

        # 构造动作字典
        if action_idx == 0:
            # 使用计算器：提取问题中的数学表达式
            content = question.replace("计算 ", "").replace("计算", "")
            action = {'type': 'calculator', 'content': content}
        elif action_idx == 1:
            # 使用搜索
            action = {'type': 'search', 'content': question}
        else:
            # 回答：尝试从历史中提取结果
            answer_content = '0'
            for h in history:
                if '计算结果' in h.get('obs', ''):
                    # 提取计算结果
                    answer_content = h['obs'].replace('计算结果: ', '')
                    break
                elif '质数' in h.get('obs', ''):
                    answer_content = '只能被1和自身整除的数'
                    break
            action = {'type': 'answer', 'content': answer_content}

        # 执行动作
        obs, reward, done, _ = env.step(action)

        # 记录轨迹
        trajectory.append((state, action_idx, log_prob, reward))
        history.append({'type': action['type'], 'obs': obs})
        total_reward += reward

    return trajectory, total_reward
    # ---- 代码结束 ----


# ============================================================
# Part 5: REINFORCE 训练（TODO 3）
# ============================================================

def train_agent_reinforce(env, policy, num_episodes=500, lr=1e-3,
                          gamma=0.99, epsilon_start=0.3, epsilon_end=0.05):
    """
    用 REINFORCE 训练 Agent

    返回: history (rewards, episode_lengths)
    """
    # TODO 3: 实现 REINFORCE 训练
    # 提示:
    # 1. 创建 optimizer
    # 2. 对每个 episode:
    #    a) execute_agent_episode 获取 trajectory
    #    b) 计算 returns (discounted rewards)
    #    c) 计算 policy gradient loss: -sum(log_prob * return)
    #    d) 反向传播更新
    #    e) epsilon 衰减
    # 3. 记录 rewards 和 episode_lengths
    # ---- 在这里实现 ----
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    all_rewards = []
    all_lengths = []

    for episode in range(num_episodes):
        # 计算当前 epsilon（线性衰减）
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / num_episodes)

        # 执行 episode
        trajectory, total_reward = execute_agent_episode(env, policy, epsilon)

        # 计算折扣回报
        returns = []
        G = 0.0
        for _, _, _, reward in reversed(trajectory):
            G = reward + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)

        # 归一化回报（减少方差）
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 计算 policy gradient loss
        policy_loss = []
        for (state, action_idx, log_prob, reward), G in zip(trajectory, returns):
            policy_loss.append(-log_prob * G)

        loss = torch.stack(policy_loss).sum()

        # 反向传播更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录
        all_rewards.append(total_reward)
        all_lengths.append(len(trajectory))

        # 打印进度
        if (episode + 1) % 50 == 0:
            avg_r = np.mean(all_rewards[-50:])
            avg_l = np.mean(all_lengths[-50:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"平均奖励: {avg_r:.3f}, 平均步数: {avg_l:.1f}, "
                  f"epsilon: {epsilon:.3f}")

    return all_rewards, all_lengths
    # ---- 代码结束 ----


# ============================================================
# Part 6: 评估（TODO 4）
# ============================================================

def evaluate_agent(env, policy, num_episodes=100):
    """
    评估 Agent 性能（无探索）

    返回: success_rate, avg_reward, avg_steps
    """
    # TODO 4: 实现评估
    # 提示:
    # 1. 运行 num_episodes 个 episode（epsilon=0）
    # 2. 记录成功率（reward > 0.5）、平均奖励、平均步数
    # ---- 在这里实现 ----
    total_rewards = []
    total_steps = []
    successes = 0

    for _ in range(num_episodes):
        trajectory, total_reward = execute_agent_episode(env, policy, epsilon=0.0)
        total_rewards.append(total_reward)
        total_steps.append(len(trajectory))
        if total_reward > 0.5:
            successes += 1

    success_rate = successes / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)

    return success_rate, avg_reward, avg_steps
    # ---- 代码结束 ----


# ============================================================
# Part 7: 可视化（无 TODO）
# ============================================================

def plot_training(rewards, episode_lengths):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(rewards, alpha=0.3, color='blue')
    # Smooth
    if len(rewards) >= 20:
        smoothed = np.convolve(rewards, np.ones(20) / 20, mode='valid')
        ax.plot(range(19, len(rewards)), smoothed, color='blue', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('总奖励')
    ax.set_title('训练奖励曲线')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(episode_lengths, alpha=0.3, color='green')
    if len(episode_lengths) >= 20:
        smoothed = np.convolve(episode_lengths, np.ones(20) / 20, mode='valid')
        ax.plot(range(19, len(episode_lengths)), smoothed, color='green', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('步数')
    ax.set_title('Episode 长度')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_29/agent_training.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_evaluation(success_rate, avg_reward, avg_steps):
    """绘制评估结果"""
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = ['成功率', '平均奖励', '平均步数']
    values = [success_rate, avg_reward / 2.0, avg_steps / 5.0]  # normalize for display
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('归一化值')
    ax.set_title('Agent 评估结果')
    ax.set_ylim([0, 1.2])

    for bar, val in zip(bars, [success_rate, avg_reward, avg_steps]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12)

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_29/agent_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # 步骤 1: 创建环境和策略
    print("=" * 60)
    print("步骤 1: 创建环境和策略")
    print("=" * 60)
    env = MathTaskEnv()
    policy = AgentPolicy(state_dim=20, hidden_dim=64, num_actions=3)
    print(f"策略参数量: {sum(p.numel() for p in policy.parameters())}")

    # 步骤 2: 训练 Agent
    print("\n" + "=" * 60)
    print("步骤 2: 训练 Agent (REINFORCE)")
    print("=" * 60)
    rewards, lengths = train_agent_reinforce(env, policy, num_episodes=500, lr=1e-3)
    plot_training(rewards, lengths)

    # 步骤 3: 评估
    print("\n" + "=" * 60)
    print("步骤 3: 评估 Agent")
    print("=" * 60)
    success_rate, avg_reward, avg_steps = evaluate_agent(env, policy, num_episodes=100)
    print(f"成功率: {success_rate:.2%}")
    print(f"平均奖励: {avg_reward:.4f}")
    print(f"平均步数: {avg_steps:.2f}")
    plot_evaluation(success_rate, avg_reward, avg_steps)

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print("\n关键发现：")
    print("1. LLM Agent 可以通过 RL 学习工具使用策略")
    print("2. 在线 RL 允许 Agent 从试错中学习")
    print("3. 奖励设计很重要：任务奖励 + 过程奖励 + 效率惩罚")
    print("4. 实际应用中需要更复杂的状态编码和动作空间")
