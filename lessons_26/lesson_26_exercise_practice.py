"""
Lesson 26: DPO 变体与发展
练习代码 - 练习版本

本练习实现 DPO 的多种变体并对比，包括：
1. DPO (Direct Preference Optimization)
2. IPO (Identity Preference Optimization)
3. SimPO (Simple Preference Optimization)
4. 方法对比分析
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: SimplePolicyNet（与 L25 相同的 GRU 策略网络）
# ============================================================

class SimplePolicyNet(nn.Module):
    """
    基于 GRU 的简单策略网络
    用于模拟语言模型的 token 级别生成概率
    """

    def __init__(self, vocab_size=20, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        """
        前向传播，返回每个位置的 logits
        input_ids: (batch, seq_len)
        返回: (batch, seq_len, vocab_size)
        """
        embeds = self.embedding(input_ids)
        outputs, _ = self.gru(embeds)
        logits = self.head(outputs)
        return logits

    def compute_sequence_log_prob(self, input_ids):
        """
        计算整个序列的对数概率（自回归方式）
        对每个位置 t，用位置 t-1 的输出预测位置 t 的 token
        input_ids: (batch, seq_len)
        返回: (batch,) 每个序列的总对数概率
        """
        logits = self.forward(input_ids)
        # 用位置 0..T-2 的 logits 预测位置 1..T-1 的 token
        shift_logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab_size)
        shift_labels = input_ids[:, 1:]   # (batch, seq_len-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        # 收集每个位置对应 token 的 log prob
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        # 对序列维度求和，得到整个序列的 log prob
        sequence_log_probs = token_log_probs.sum(dim=-1)
        return sequence_log_probs


# ============================================================
# Part 2: 偏好数据集生成
# ============================================================

def true_quality_score(token_ids):
    """
    真实的质量评分函数（模拟人类偏好）
    - 多样性奖励: 不同 token 越多越好
    - 重复惩罚: 连续重复 token 会被惩罚
    """
    unique_ratio = len(set(token_ids.tolist())) / len(token_ids)
    diversity_bonus = unique_ratio * 2.0

    repeat_penalty = 0.0
    for i in range(1, len(token_ids)):
        if token_ids[i] == token_ids[i - 1]:
            repeat_penalty += 0.5

    return diversity_bonus - repeat_penalty


def generate_preference_dataset(policy, num_pairs=500, vocab_size=20,
                                prompt_len=3, response_len=8):
    """
    生成合成的偏好对数据集
    对于每一对，随机生成两个序列，用 true_quality_score 判断优劣
    返回: list of (chosen_ids, rejected_ids)，均为 numpy 数组
    """
    dataset = []
    total_len = prompt_len + response_len

    for _ in range(num_pairs):
        # 生成共享 prompt
        prompt = np.random.randint(0, vocab_size, size=(prompt_len,))

        # 生成两个不同的 response
        response_a = np.random.randint(0, vocab_size, size=(response_len,))
        response_b = np.random.randint(0, vocab_size, size=(response_len,))

        seq_a = np.concatenate([prompt, response_a])
        seq_b = np.concatenate([prompt, response_b])

        # 用质量函数评估 response 部分
        score_a = true_quality_score(torch.tensor(response_a))
        score_b = true_quality_score(torch.tensor(response_b))

        if score_a > score_b:
            dataset.append((seq_a, seq_b))
        elif score_b > score_a:
            dataset.append((seq_b, seq_a))
        else:
            # 分数相同时随机选择
            if np.random.rand() > 0.5:
                dataset.append((seq_a, seq_b))
            else:
                dataset.append((seq_b, seq_a))

    return dataset


# ============================================================
# Part 3: DPO 损失（TODO 1）
# ============================================================

def compute_dpo_loss(policy, ref_policy, chosen_ids, rejected_ids, beta=0.1):
    """
    计算 DPO 损失
    L = -log σ(β * (log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)))

    参数:
        policy: 当前策略网络
        ref_policy: 参考策略网络（冻结）
        chosen_ids: 偏好序列 (batch, seq_len)
        rejected_ids: 非偏好序列 (batch, seq_len)
        beta: 温度参数

    返回:
        loss: DPO 损失
        chosen_rewards: 偏好序列的隐式奖励
        rejected_rewards: 非偏好序列的隐式奖励

    提示:
        1. policy_chosen_logps = policy.compute_sequence_log_prob(chosen_ids)
        2. 同样计算 policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
        3. ref 的计算需要 torch.no_grad()
        4. chosen_logratio = policy_chosen_logps - ref_chosen_logps
        5. rejected_logratio = policy_rejected_logps - ref_rejected_logps
        6. logits = beta * (chosen_logratio - rejected_logratio)
        7. loss = -F.logsigmoid(logits).mean()
        8. chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
        9. rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    """
    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return loss, chosen_rewards, rejected_rewards


# ============================================================
# Part 4: IPO 损失（TODO 2）
# ============================================================

def compute_ipo_loss(policy, ref_policy, chosen_ids, rejected_ids, beta=0.1):
    """
    计算 IPO 损失
    L = (log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l) - 1/(2β))²

    IPO 用平方损失替代 DPO 的 logsigmoid，避免过度优化

    参数:
        policy: 当前策略网络
        ref_policy: 参考策略网络（冻结）
        chosen_ids: 偏好序列 (batch, seq_len)
        rejected_ids: 非偏好序列 (batch, seq_len)
        beta: 正则化强度参数

    返回:
        loss: IPO 损失

    提示:
        1. 计算 log ratio 的方式与 DPO 相同
        2. loss = ((chosen_logratio - rejected_logratio) - 1/(2*beta))^2
        3. loss = loss.mean()
    """
    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return loss


# ============================================================
# Part 5: SimPO 损失（TODO 3）
# ============================================================

def compute_simpo_loss(policy, chosen_ids, rejected_ids, beta=2.0, gamma=0.5):
    """
    计算 SimPO 损失（无需参考模型）
    L = -log σ(β * (avg_logp(y_w) - avg_logp(y_l)) - γ)

    SimPO 的关键创新:
    - 不需要参考模型，大幅降低计算开销
    - 使用平均 log prob 而非总 log prob，消除长度偏差
    - 引入 margin γ 确保 chosen 和 rejected 之间有足够差距

    参数:
        policy: 当前策略网络
        chosen_ids: 偏好序列 (batch, seq_len)
        rejected_ids: 非偏好序列 (batch, seq_len)
        beta: 温度参数
        gamma: margin 参数

    返回:
        loss: SimPO 损失
        chosen_rewards: 偏好序列的隐式奖励
        rejected_rewards: 非偏好序列的隐式奖励

    提示:
        1. chosen_logps = policy.compute_sequence_log_prob(chosen_ids)
        2. chosen_avg = chosen_logps / chosen_ids.shape[1]
        3. 同理计算 rejected_avg
        4. logits = beta * (chosen_avg - rejected_avg) - gamma
        5. loss = -F.logsigmoid(logits).mean()
        6. chosen_rewards = (beta * chosen_avg).detach()
        7. rejected_rewards = (beta * rejected_avg).detach()
    """
    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return loss, chosen_rewards, rejected_rewards


# ============================================================
# Part 6: 通用训练函数
# ============================================================

def train_method(method_name, policy, ref_policy, dataset, loss_fn,
                 num_epochs=30, lr=1e-3, batch_size=32):
    """
    通用训练函数

    参数:
        method_name: 方法名称（用于打印）
        policy: 要训练的策略网络
        ref_policy: 参考策略网络（SimPO 时为 None）
        dataset: 偏好对数据集
        loss_fn: 损失函数，接受 (policy, ref_policy, chosen_batch, rejected_batch)
        num_epochs: 训练轮数
        lr: 学习率
        batch_size: 批大小

    返回:
        history: 字典，包含 losses, accuracies, reward_margins
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    history = {'losses': [], 'accuracies': [], 'reward_margins': []}

    num_samples = len(dataset)

    for epoch in range(num_epochs):
        # 随机打乱数据
        indices = np.random.permutation(num_samples)
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0
        epoch_margins = []

        for start in range(0, num_samples, batch_size):
            batch_indices = indices[start:start + batch_size]
            if len(batch_indices) < 2:
                continue

            # 构建 batch
            chosen_list = [torch.tensor(dataset[i][0], dtype=torch.long) for i in batch_indices]
            rejected_list = [torch.tensor(dataset[i][1], dtype=torch.long) for i in batch_indices]
            chosen_batch = torch.stack(chosen_list)
            rejected_batch = torch.stack(rejected_list)

            # 计算损失
            result = loss_fn(policy, ref_policy, chosen_batch, rejected_batch)

            if isinstance(result, tuple):
                loss = result[0]
                if len(result) >= 3:
                    chosen_rewards = result[1]
                    rejected_rewards = result[2]
                    # 计算准确率: chosen_reward > rejected_reward 的比例
                    correct = (chosen_rewards > rejected_rewards).float().sum().item()
                    epoch_correct += correct
                    epoch_total += len(chosen_rewards)
                    margin = (chosen_rewards - rejected_rewards).mean().item()
                    epoch_margins.append(margin)
                else:
                    epoch_total += len(batch_indices)
            else:
                loss = result
                epoch_total += len(batch_indices)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        # 记录每个 epoch 的统计
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        avg_margin = np.mean(epoch_margins) if epoch_margins else 0.0

        history['losses'].append(avg_loss)
        history['accuracies'].append(accuracy)
        history['reward_margins'].append(avg_margin)

        if (epoch + 1) % 5 == 0:
            print(f"[{method_name}] Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | "
                  f"Margin: {avg_margin:.4f}")

    return history


# ============================================================
# Part 7: 对比所有方法（TODO 4）
# ============================================================

def compare_methods(dataset, vocab_size=20):
    """
    对比 DPO, IPO, SimPO 三种方法

    参数:
        dataset: 偏好对数据集
        vocab_size: 词表大小

    返回:
        results: 字典 {method_name: history}

    提示:
        对每种方法:
        1. 创建新的 policy（和 ref_policy，DPO/IPO 需要）
        2. ref_policy 要复制 policy 的参数并冻结
        3. 定义 loss_fn（lambda 包装对应损失函数）
        4. 调用 train_method
        5. IPO 的 loss_fn 需要额外计算 reward 信号用于监控
        6. SimPO 不需要 ref_policy
    """
    results = {}

    # ---- 在这里实现 ----
    pass
    # ---- 代码结束 ----

    return results


# ============================================================
# Part 8: 可视化
# ============================================================

def plot_training_comparison(results):
    """
    绘制训练过程对比图
    三个子图: 损失曲线、准确率曲线、奖励间距曲线
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'DPO': '#2196F3', 'IPO': '#FF9800', 'SimPO': '#4CAF50'}

    # 子图 1: 损失曲线
    ax = axes[0]
    for method, history in results.items():
        ax.plot(history['losses'], label=method, color=colors.get(method, 'gray'),
                linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('损失')
    ax.set_title('训练损失对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图 2: 准确率曲线
    ax = axes[1]
    for method, history in results.items():
        ax.plot(history['accuracies'], label=method, color=colors.get(method, 'gray'),
                linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('准确率')
    ax.set_title('偏好预测准确率对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # 子图 3: 奖励间距曲线
    ax = axes[2]
    for method, history in results.items():
        ax.plot(history['reward_margins'], label=method, color=colors.get(method, 'gray'),
                linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('奖励间距')
    ax.set_title('Chosen-Rejected 奖励间距对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_26/training_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("训练对比图已保存到 e:/rl_lessons/lessons_26/training_comparison.png")


def plot_final_comparison(results):
    """
    绘制最终结果对比柱状图
    两个子图: 最终准确率、最终奖励间距
    """
    methods = list(results.keys())
    final_accs = [results[m]['accuracies'][-1] if results[m]['accuracies'] else 0
                  for m in methods]
    final_margins = [results[m]['reward_margins'][-1] if results[m]['reward_margins'] else 0
                     for m in methods]
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 子图 1: 最终准确率
    ax = axes[0]
    bars = ax.bar(methods, final_accs, color=colors[:len(methods)], alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, final_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('准确率')
    ax.set_title('最终偏好预测准确率')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    # 子图 2: 最终奖励间距
    ax = axes[1]
    bars = ax.bar(methods, final_margins, color=colors[:len(methods)], alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, final_margins):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('奖励间距')
    ax.set_title('最终 Chosen-Rejected 奖励间距')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_26/final_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("最终对比图已保存到 e:/rl_lessons/lessons_26/final_comparison.png")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Step 1: 创建初始策略并生成偏好数据集
    print("=" * 60)
    print("Lesson 26: DPO 变体与发展")
    print("=" * 60)

    init_policy = SimplePolicyNet(vocab_size=20)
    print("\n[Step 1] 生成偏好数据集...")
    dataset = generate_preference_dataset(init_policy, num_pairs=500,
                                          vocab_size=20, prompt_len=3,
                                          response_len=8)
    print(f"数据集大小: {len(dataset)} 对偏好数据")
    print(f"序列长度: {len(dataset[0][0])} (prompt 3 + response 8)")

    # Step 2: 对比三种方法
    print("\n[Step 2] 训练并对比 DPO / IPO / SimPO...")
    results = compare_methods(dataset, vocab_size=20)

    # Step 3: 可视化
    print("\n[Step 3] 生成可视化图表...")
    plot_training_comparison(results)
    plot_final_comparison(results)

    # 打印汇总表
    print("\n" + "=" * 60)
    print("方法对比汇总")
    print("=" * 60)
    print(f"{'方法':<10} {'最终损失':>10} {'最终准确率':>10} {'最终奖励间距':>12}")
    print("-" * 45)
    for method, history in results.items():
        final_loss = history['losses'][-1] if history['losses'] else 0
        final_acc = history['accuracies'][-1] if history['accuracies'] else 0
        final_margin = history['reward_margins'][-1] if history['reward_margins'] else 0
        print(f"{method:<10} {final_loss:>10.4f} {final_acc:>10.4f} {final_margin:>12.4f}")

    print("\n关键观察:")
    print("1. DPO: 使用 logsigmoid 损失，依赖参考模型约束策略偏移")
    print("2. IPO: 使用平方损失，对异常偏好对更鲁棒，防止过度优化")
    print("3. SimPO: 无需参考模型，用平均 log prob + margin 实现对齐")
