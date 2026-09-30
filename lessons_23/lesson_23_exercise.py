"""
Lesson 23: 从 RL 到 LLM 对齐
练习代码 - 参考实现

本练习分析奖励模型的数据格式和训练流程，包括：
1. 偏好数据的构造与处理
2. Bradley-Terry 奖励模型（简化版）
3. 奖励模型训练
4. 模拟 RLHF 训练流程
5. 分析奖励过度优化现象
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: 偏好数据构造
# ============================================================

class PreferenceDataset:
    """
    偏好数据集

    模拟 RLHF 中的人类偏好数据：
    - 每条数据包含: prompt, chosen_response, rejected_response
    - 用特征向量表示 prompt 和 response
    """

    def __init__(self, num_samples=1000, feature_dim=10, seed=42):
        """
        初始化偏好数据集

        参数:
            num_samples: 样本数量
            feature_dim: 特征维度
            seed: 随机种子
        """
        np.random.seed(seed)
        self.feature_dim = feature_dim
        self.num_samples = num_samples

        # TODO 1: 构造偏好数据集
        # 提示：
        # 1. 生成 prompts: 随机特征向量 (num_samples, feature_dim)
        # 2. 定义"真实"奖励函数（线性）: true_weights
        # 3. 对每个 prompt 生成两个 response（随机特征向量）
        # 4. 用真实奖励函数计算分数
        # 5. 分数高的是 chosen，低的是 rejected
        # ---- 在这里实现 ----
        # 真实奖励函数的权重（模拟人类偏好）
        self.true_weights = np.random.randn(feature_dim)
        self.true_weights /= np.linalg.norm(self.true_weights)

        # 生成数据
        self.prompts = []
        self.chosen = []
        self.rejected = []
        self.chosen_scores = []
        self.rejected_scores = []

        for _ in range(num_samples):
            # 生成 prompt
            prompt = np.random.randn(feature_dim) * 0.5

            # 生成两个 response
            response_a = prompt + np.random.randn(feature_dim) * 0.3
            response_b = prompt + np.random.randn(feature_dim) * 0.3

            # 计算真实奖励
            score_a = np.dot(response_a, self.true_weights)
            score_b = np.dot(response_b, self.true_weights)

            # 加入噪声模拟标注不一致性
            noisy_score_a = score_a + np.random.randn() * 0.1
            noisy_score_b = score_b + np.random.randn() * 0.1

            # 根据（带噪声的）分数排序
            if noisy_score_a > noisy_score_b:
                self.chosen.append(response_a)
                self.rejected.append(response_b)
                self.chosen_scores.append(score_a)
                self.rejected_scores.append(score_b)
            else:
                self.chosen.append(response_b)
                self.rejected.append(response_a)
                self.chosen_scores.append(score_b)
                self.rejected_scores.append(score_a)

            self.prompts.append(prompt)

        # 转换为 numpy 数组
        self.prompts = np.array(self.prompts)
        self.chosen = np.array(self.chosen)
        self.rejected = np.array(self.rejected)
        self.chosen_scores = np.array(self.chosen_scores)
        self.rejected_scores = np.array(self.rejected_scores)
        # ---- 代码结束 ----

    def get_batch(self, batch_size=32):
        """获取一个 batch 的数据"""
        indices = np.random.choice(self.num_samples, batch_size, replace=False)
        return (
            torch.FloatTensor(self.chosen[indices]),
            torch.FloatTensor(self.rejected[indices]),
        )

    def describe(self):
        """描述数据集"""
        print(f"数据集大小: {self.num_samples}")
        print(f"特征维度: {self.feature_dim}")
        print(f"Chosen 平均分数: {self.chosen_scores.mean():.4f}")
        print(f"Rejected 平均分数: {self.rejected_scores.mean():.4f}")
        print(f"分数差均值: {(self.chosen_scores - self.rejected_scores).mean():.4f}")

        # 计算标注一致性（chosen 分数 > rejected 分数的比例）
        consistency = np.mean(self.chosen_scores > self.rejected_scores)
        print(f"标注一致性: {consistency:.2%}")


# ============================================================
# Part 2: Bradley-Terry 奖励模型
# ============================================================

class RewardModel(nn.Module):
    """
    Bradley-Terry 奖励模型

    给定一个 response 的特征向量，输出标量奖励
    """

    def __init__(self, feature_dim=10, hidden_dim=64):
        """
        初始化奖励模型

        参数:
            feature_dim: 输入特征维度
            hidden_dim: 隐藏层维度
        """
        super(RewardModel, self).__init__()

        # TODO 2: 实现奖励模型网络
        # 提示：
        # 1. 特征层: feature_dim -> hidden_dim -> hidden_dim
        # 2. 奖励头: hidden_dim -> 1
        # 3. 使用 ReLU 激活
        # ---- 在这里实现 ----
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # ---- 代码结束 ----

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征 (batch_size, feature_dim)

        返回:
            reward: 标量奖励 (batch_size,)
        """
        return self.network(x).squeeze(-1)


def train_reward_model(reward_model, dataset, num_epochs=50,
                       batch_size=64, lr=1e-3):
    """
    训练奖励模型

    参数:
        reward_model: 奖励模型
        dataset: 偏好数据集
        num_epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率

    返回:
        losses: 训练损失记录
        accuracies: 训练准确率记录
    """
    optimizer = optim.Adam(reward_model.parameters(), lr=lr)
    losses = []
    accuracies = []

    # TODO 3: 实现奖励模型训练
    # 提示：
    # 1. 每个 epoch，随机采样 batch
    # 2. 计算 chosen 和 rejected 的奖励
    # 3. Bradley-Terry 损失: -log(sigmoid(r_chosen - r_rejected))
    # 4. 计算准确率: chosen 的奖励 > rejected 的比例
    # 5. 反向传播更新
    # ---- 在这里实现 ----
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0

        # 多次采样 batch
        n_batches = dataset.num_samples // batch_size
        for _ in range(n_batches):
            chosen, rejected = dataset.get_batch(batch_size)

            # 计算奖励
            r_chosen = reward_model(chosen)
            r_rejected = reward_model(rejected)

            # Bradley-Terry 损失
            loss = -torch.log(torch.sigmoid(r_chosen - r_rejected) + 1e-8).mean()

            # 准确率
            correct = (r_chosen > r_rejected).float().sum().item()
            epoch_correct += correct
            epoch_total += batch_size

            # 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        accuracy = epoch_correct / epoch_total
        losses.append(avg_loss)
        accuracies.append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
                  f"Accuracy: {accuracy:.2%}")

    return losses, accuracies
    # ---- 代码结束 ----


# ============================================================
# Part 3: 模拟 RLHF 训练
# ============================================================

class SimplePolicy:
    """
    简化版策略模型

    在特征空间中，策略决定 response 的方向
    """

    def __init__(self, feature_dim=10, lr=0.01):
        self.feature_dim = feature_dim
        self.lr = lr
        # 策略参数（response 的偏好方向）
        self.weights = np.random.randn(feature_dim) * 0.1

    def generate(self, prompt, noise_scale=0.2):
        """生成 response"""
        response = prompt + self.weights + np.random.randn(self.feature_dim) * noise_scale
        return response

    def update(self, gradient):
        """策略梯度更新"""
        self.weights += self.lr * gradient


def simulate_rlhf(reward_model, true_weights, feature_dim=10,
                   num_steps=200, kl_coef=0.1, batch_size=16):
    """
    模拟 RLHF 训练过程

    参数:
        reward_model: 训练好的奖励模型
        true_weights: 真实奖励权重
        feature_dim: 特征维度
        num_steps: 训练步数
        kl_coef: KL 惩罚系数
        batch_size: 批大小

    返回:
        rm_rewards: 奖励模型分数记录
        true_rewards: 真实奖励记录
        kl_divs: KL 散度记录
    """
    # TODO 4: 实现模拟 RLHF 训练
    # 提示：
    # 1. 创建策略和参考策略
    # 2. 每步：
    #    a) 生成 prompt（随机）
    #    b) 用策略生成 response
    #    c) 用参考策略生成 reference response
    #    d) 计算奖励模型分数
    #    e) 计算 KL 散度（用 response 差异近似）
    #    f) 计算总奖励: r_total = r_model - kl_coef * kl
    #    g) 计算真实奖励（用 true_weights）
    #    h) 更新策略（简单的梯度方向）
    # 3. 返回记录
    # ---- 在这里实现 ----
    policy = SimplePolicy(feature_dim, lr=0.01)
    ref_policy = SimplePolicy(feature_dim, lr=0.0)  # 参考策略不更新
    ref_policy.weights = policy.weights.copy()  # 初始化相同

    rm_rewards = []
    true_rewards = []
    kl_divs = []

    reward_model.eval()

    for step in range(num_steps):
        batch_rm_reward = 0
        batch_true_reward = 0
        batch_kl = 0
        batch_gradient = np.zeros(feature_dim)

        for _ in range(batch_size):
            # 生成 prompt
            prompt = np.random.randn(feature_dim) * 0.5

            # 生成 response
            response = policy.generate(prompt)
            ref_response = ref_policy.generate(prompt)

            # 奖励模型分数
            with torch.no_grad():
                rm_reward = reward_model(
                    torch.FloatTensor(response).unsqueeze(0)
                ).item()

            # KL 散度近似（用参数差异）
            kl = 0.5 * np.sum((policy.weights - ref_policy.weights) ** 2)

            # 总奖励
            total_reward = rm_reward - kl_coef * kl

            # 真实奖励
            true_reward = np.dot(response, true_weights)

            # 简单策略梯度（REINFORCE 风格）
            direction = response - prompt  # 策略选择的方向
            batch_gradient += total_reward * direction / np.linalg.norm(direction + 1e-8)

            batch_rm_reward += rm_reward
            batch_true_reward += true_reward
            batch_kl += kl

        # 更新策略
        policy.update(batch_gradient / batch_size * 0.01)

        # 记录
        rm_rewards.append(batch_rm_reward / batch_size)
        true_rewards.append(batch_true_reward / batch_size)
        kl_divs.append(batch_kl / batch_size)

        if (step + 1) % 50 == 0:
            print(f"Step {step+1}/{num_steps}, "
                  f"RM Reward: {rm_rewards[-1]:.4f}, "
                  f"True Reward: {true_rewards[-1]:.4f}, "
                  f"KL: {kl_divs[-1]:.4f}")

    return rm_rewards, true_rewards, kl_divs
    # ---- 代码结束 ----


# ============================================================
# Part 4: 分析奖励过度优化
# ============================================================

def analyze_overoptimization(reward_model, true_weights, feature_dim=10):
    """
    分析不同 KL 系数下的奖励过度优化现象

    参数:
        reward_model: 训练好的奖励模型
        true_weights: 真实奖励权重
        feature_dim: 特征维度
    """
    print("\n分析奖励过度优化...")

    kl_coefs = [0.0, 0.01, 0.1, 1.0]
    results = {}

    for kl_coef in kl_coefs:
        print(f"\nKL 系数: {kl_coef}")
        rm_rewards, true_rewards, kl_divs = simulate_rlhf(
            reward_model, true_weights, feature_dim,
            num_steps=150, kl_coef=kl_coef, batch_size=16
        )
        results[kl_coef] = {
            'rm_rewards': rm_rewards,
            'true_rewards': true_rewards,
            'kl_divs': kl_divs
        }

    return results


# ============================================================
# Part 5: 可视化
# ============================================================

def plot_reward_model_training(losses, accuracies):
    """绘制奖励模型训练过程"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(losses, color='blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('损失')
    ax.set_title('奖励模型训练损失')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(accuracies, color='green')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='随机猜测')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('准确率')
    ax.set_title('奖励模型偏好准确率')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_23/reward_model_training.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_rlhf_results(rm_rewards, true_rewards, kl_divs):
    """绘制 RLHF 训练结果"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(rm_rewards, color='blue', label='RM 奖励')
    ax.plot(true_rewards, color='red', label='真实奖励')
    ax.set_xlabel('训练步数')
    ax.set_ylabel('奖励')
    ax.set_title('奖励变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(kl_divs, color='orange')
    ax.set_xlabel('训练步数')
    ax.set_ylabel('KL 散度')
    ax.set_title('KL 散度变化')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.scatter(kl_divs, true_rewards, alpha=0.5, c=range(len(kl_divs)),
               cmap='viridis', s=10)
    ax.set_xlabel('KL 散度')
    ax.set_ylabel('真实奖励')
    ax.set_title('KL vs 真实奖励')
    ax.grid(True, alpha=0.3)
    plt.colorbar(ax.collections[0], ax=ax, label='训练步数')

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_23/rlhf_simulation.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_overoptimization(results):
    """绘制奖励过度优化分析"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']

    # RM 奖励 vs 真实奖励
    ax = axes[0]
    for (kl_coef, data), color in zip(results.items(), colors):
        ax.plot(data['true_rewards'], label=f'KL={kl_coef}',
                color=color, alpha=0.8)
    ax.set_xlabel('训练步数')
    ax.set_ylabel('真实奖励')
    ax.set_title('不同 KL 系数下的真实奖励')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # KL 散度
    ax = axes[1]
    for (kl_coef, data), color in zip(results.items(), colors):
        ax.plot(data['kl_divs'], label=f'KL={kl_coef}',
                color=color, alpha=0.8)
    ax.set_xlabel('训练步数')
    ax.set_ylabel('KL 散度')
    ax.set_title('不同 KL 系数下的 KL 散度')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_23/overoptimization.png',
                dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    feature_dim = 10

    # ----- 步骤 1: 构造偏好数据集 -----
    print("=" * 60)
    print("步骤 1: 构造偏好数据集")
    print("=" * 60)

    dataset = PreferenceDataset(num_samples=2000, feature_dim=feature_dim)
    dataset.describe()

    # ----- 步骤 2: 训练奖励模型 -----
    print("\n" + "=" * 60)
    print("步骤 2: 训练奖励模型")
    print("=" * 60)

    rm = RewardModel(feature_dim=feature_dim, hidden_dim=64)
    losses, accuracies = train_reward_model(rm, dataset, num_epochs=50,
                                            batch_size=64, lr=1e-3)

    plot_reward_model_training(losses, accuracies)

    # ----- 步骤 3: 模拟 RLHF 训练 -----
    print("\n" + "=" * 60)
    print("步骤 3: 模拟 RLHF 训练")
    print("=" * 60)

    rm_rewards, true_rewards, kl_divs = simulate_rlhf(
        rm, dataset.true_weights, feature_dim,
        num_steps=200, kl_coef=0.1, batch_size=16
    )

    plot_rlhf_results(rm_rewards, true_rewards, kl_divs)

    # ----- 步骤 4: 分析奖励过度优化 -----
    print("\n" + "=" * 60)
    print("步骤 4: 分析奖励过度优化")
    print("=" * 60)

    overopt_results = analyze_overoptimization(rm, dataset.true_weights, feature_dim)
    plot_overoptimization(overopt_results)

    # ----- 总结 -----
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print(f"奖励模型最终准确率: {accuracies[-1]:.2%}")
    print(f"RLHF 最终 RM 奖励: {rm_rewards[-1]:.4f}")
    print(f"RLHF 最终真实奖励: {true_rewards[-1]:.4f}")
    print(f"RLHF 最终 KL 散度: {kl_divs[-1]:.4f}")
    print("\n关键发现：")
    print("1. 奖励模型能从偏好对比数据中学习")
    print("2. KL 约束对防止过度优化至关重要")
    print("3. KL 系数需要权衡：太小可能 reward hacking，太大则无法优化")
    print("4. RM 奖励提升 ≠ 真实奖励提升（过度优化）")
