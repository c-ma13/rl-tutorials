"""
Lesson 25: DPO (Direct Preference Optimization)
练习代码 - 参考实现

本练习实现 DPO 算法，包括：
1. 简化策略网络（模拟语言模型）
2. 偏好数据集构造
3. DPO 损失函数
4. DPO 训练
5. DPO vs RLHF 对比分析
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: 简化策略网络（模拟语言模型）
# ============================================================

class SimplePolicyNet(nn.Module):
    """
    简化的策略网络（模拟语言模型）

    使用 GRU + Linear 生成 token 概率分布
    - vocab_size: 词汇表大小
    - embed_dim: 嵌入维度
    - hidden_dim: 隐藏层维度
    """

    def __init__(self, vocab_size=20, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        """
        计算每个位置的 logits

        参数:
            input_ids: (batch, seq_len) 或 (seq_len,) token 序列

        返回:
            logits: (batch, seq_len, vocab_size)
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        embeds = self.embedding(input_ids)
        output, _ = self.gru(embeds)
        logits = self.head(output)
        return logits

    def compute_sequence_log_prob(self, input_ids):
        """
        计算整个序列的 log 概率之和

        对于序列 [t0, t1, t2, ..., tN]，计算:
            sum_{i=1}^{N} log P(t_i | t_0, ..., t_{i-1})

        参数:
            input_ids: (batch, seq_len) token 序列

        返回:
            log_prob: (batch,) 每个样本的序列 log 概率
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        logits = self.forward(input_ids)  # (batch, seq_len, vocab_size)
        # 用 t_0..t_{N-1} 的 logits 预测 t_1..t_N
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (batch, seq_len-1, vocab_size)
        # 取每个目标 token 对应的 log 概率
        targets = input_ids[:, 1:]  # (batch, seq_len-1)
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len-1)
        # 对序列求和
        sequence_log_prob = token_log_probs.sum(dim=-1)  # (batch,)
        return sequence_log_prob


# ============================================================
# Part 2: 偏好数据集构造
# ============================================================

def true_quality_score(sequence, vocab_size=20):
    """
    模拟"真实的人类偏好"质量评分

    评分标准：
    - 多样性奖励：使用不同 token 越多越好
    - 重复惩罚：连续重复的 token 扣分
    - 长度归一化

    参数:
        sequence: token 序列 (numpy array 或 list)
        vocab_size: 词汇表大小

    返回:
        score: 质量分数
    """
    seq = np.array(sequence)
    if len(seq) == 0:
        return 0.0

    # 多样性奖励：不同 token 数量 / 序列长度
    unique_ratio = len(set(seq.tolist())) / len(seq)
    diversity_bonus = unique_ratio * 3.0

    # 连续重复惩罚
    repeat_count = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1])
    repeat_penalty = repeat_count * 0.8

    # 三连重复额外惩罚
    triple_count = sum(
        1 for i in range(2, len(seq))
        if seq[i] == seq[i - 1] == seq[i - 2]
    )
    triple_penalty = triple_count * 1.5

    return diversity_bonus - repeat_penalty - triple_penalty


def generate_preference_dataset(policy, num_pairs=500, seq_len=10, vocab_size=20):
    """
    使用策略网络生成偏好数据集

    对于每对数据：
    1. 随机生成 prompt（前3个 token）
    2. 用策略网络在不同温度下生成两个 response
    3. 用 true_quality_score 判断哪个更好

    参数:
        policy: 策略网络
        num_pairs: 偏好对数量
        seq_len: 总序列长度（prompt + response）
        vocab_size: 词汇表大小

    返回:
        dataset: list of (chosen_ids, rejected_ids) 其中每个是完整序列的 tensor
    """
    dataset = []
    policy.eval()
    prompt_len = 3
    response_len = seq_len - prompt_len

    with torch.no_grad():
        for _ in range(num_pairs):
            # 随机 prompt
            prompt = torch.randint(0, vocab_size, (prompt_len,))

            # 生成两个不同 response（使用不同温度引入差异）
            seq_a = _generate_sequence(policy, prompt, response_len, temperature=0.8)
            seq_b = _generate_sequence(policy, prompt, response_len, temperature=1.5)

            # 评估质量
            score_a = true_quality_score(seq_a.numpy(), vocab_size)
            score_b = true_quality_score(seq_b.numpy(), vocab_size)

            # chosen 是分数更高的，rejected 是分数更低的
            if score_a >= score_b:
                dataset.append((seq_a, seq_b))
            else:
                dataset.append((seq_b, seq_a))

    policy.train()
    return dataset


def _generate_sequence(policy, prompt, response_len, temperature=1.0):
    """
    自回归生成序列

    参数:
        policy: 策略网络
        prompt: (prompt_len,) prompt tokens
        response_len: 要生成的 token 数量
        temperature: 采样温度

    返回:
        full_sequence: (prompt_len + response_len,) 完整序列
    """
    tokens = prompt.tolist()
    for _ in range(response_len):
        input_ids = torch.LongTensor([tokens])
        logits = policy(input_ids)
        next_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        token = torch.multinomial(probs, 1).item()
        tokens.append(token)
    return torch.LongTensor(tokens)


# ============================================================
# Part 3: DPO 损失函数 (TODO 1 - 核心)
# ============================================================

def compute_dpo_loss(policy, ref_policy, chosen_ids, rejected_ids, beta=0.1):
    """
    计算 DPO 损失

    DPO 核心公式:
        Loss = -log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))

    等价于:
        Loss = -log σ(β * ((log π_θ(y_w) - log π_ref(y_w)) - (log π_θ(y_l) - log π_ref(y_l))))

    参数:
        policy: 当前策略网络 π_θ
        ref_policy: 参考策略网络 π_ref（冻结）
        chosen_ids: (batch, seq_len) 被选中的序列
        rejected_ids: (batch, seq_len) 被拒绝的序列
        beta: DPO 温度参数，控制偏离参考策略的程度

    返回:
        loss: DPO 损失标量
        chosen_rewards: (batch,) chosen 的隐式奖励
        rejected_rewards: (batch,) rejected 的隐式奖励
    """
    # TODO 1: 计算 DPO 损失
    # 提示：
    # 1. 用 policy.compute_sequence_log_prob 计算策略对 chosen 和 rejected 的 log 概率
    # 2. 用 ref_policy 计算参考策略的 log 概率（with torch.no_grad）
    # 3. 计算 log ratio: policy_logps - ref_logps（对 chosen 和 rejected 分别计算）
    # 4. 计算 DPO logits = beta * (chosen_logratio - rejected_logratio)
    # 5. 损失 = -F.logsigmoid(logits).mean()
    # 6. 隐式奖励 = beta * (policy_logps - ref_logps)
    # ---- 在这里实现 ----
    # 步骤 1: 策略的 log 概率
    policy_chosen_logps = policy.compute_sequence_log_prob(chosen_ids)
    policy_rejected_logps = policy.compute_sequence_log_prob(rejected_ids)

    # 步骤 2: 参考策略的 log 概率（不计算梯度）
    with torch.no_grad():
        ref_chosen_logps = ref_policy.compute_sequence_log_prob(chosen_ids)
        ref_rejected_logps = ref_policy.compute_sequence_log_prob(rejected_ids)

    # 步骤 3: 计算 log ratio
    chosen_logratio = policy_chosen_logps - ref_chosen_logps
    rejected_logratio = policy_rejected_logps - ref_rejected_logps

    # 步骤 4: DPO logits
    logits = beta * (chosen_logratio - rejected_logratio)

    # 步骤 5: DPO 损失
    loss = -F.logsigmoid(logits).mean()

    # 步骤 6: 隐式奖励（用于监控）
    chosen_rewards = beta * chosen_logratio.detach()
    rejected_rewards = beta * rejected_logratio.detach()

    return loss, chosen_rewards, rejected_rewards
    # ---- 代码结束 ----


# ============================================================
# Part 4: DPO 训练循环 (TODO 2)
# ============================================================

def train_dpo(policy, ref_policy, dataset, beta=0.1, lr=1e-3, num_epochs=50, batch_size=32):
    """
    DPO 训练循环

    参数:
        policy: 待训练的策略网络
        ref_policy: 参考策略网络（冻结）
        dataset: 偏好数据集 [(chosen_ids, rejected_ids), ...]
        beta: DPO 温度参数
        lr: 学习率
        num_epochs: 训练轮数
        batch_size: 批大小

    返回:
        losses: 每个 epoch 的平均损失
        accuracies: 每个 epoch 的偏好准确率（chosen_reward > rejected_reward 的比率）
        reward_margins: 每个 epoch 的平均奖励差距
    """
    # TODO 2: 实现 DPO 训练循环
    # 提示：
    # 1. 创建优化器
    # 2. 每个 epoch 中：
    #    a. 打乱数据索引
    #    b. 按 batch_size 分批
    #    c. 对每个 batch 调用 compute_dpo_loss
    #    d. 反向传播并更新
    #    e. 记录 loss、accuracy（chosen_reward > rejected_reward）、reward_margin
    # 3. 每 10 个 epoch 打印进度
    # ---- 在这里实现 ----
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    losses = []
    accuracies = []
    reward_margins = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_margin = 0.0
        epoch_count = 0

        # 打乱数据
        indices = np.random.permutation(len(dataset))

        for start in range(0, len(dataset), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_chosen = torch.stack([dataset[i][0] for i in batch_indices])
            batch_rejected = torch.stack([dataset[i][1] for i in batch_indices])

            loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
                policy, ref_policy, batch_chosen, batch_rejected, beta=beta
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            batch_len = len(batch_indices)
            epoch_loss += loss.item() * batch_len
            epoch_correct += (chosen_rewards > rejected_rewards).sum().item()
            epoch_margin += (chosen_rewards - rejected_rewards).sum().item()
            epoch_count += batch_len

        avg_loss = epoch_loss / epoch_count
        accuracy = epoch_correct / epoch_count
        avg_margin = epoch_margin / epoch_count

        losses.append(avg_loss)
        accuracies.append(accuracy)
        reward_margins.append(avg_margin)

        if (epoch + 1) % 10 == 0:
            print(f"DPO Epoch {epoch + 1}/{num_epochs}, "
                  f"Loss: {avg_loss:.4f}, "
                  f"Accuracy: {accuracy:.2%}, "
                  f"Reward Margin: {avg_margin:.4f}")

    return losses, accuracies, reward_margins
    # ---- 代码结束 ----


# ============================================================
# Part 5: Beta 敏感性分析 (TODO 3)
# ============================================================

def analyze_beta_sensitivity(dataset, betas=[0.01, 0.05, 0.1, 0.5, 1.0],
                             vocab_size=20, num_epochs=50, lr=1e-3):
    """
    分析不同 beta 值对 DPO 训练效果的影响

    beta 是 DPO 的关键超参数：
    - 小 beta: 策略可以大幅偏离参考策略，训练更激进
    - 大 beta: 策略被约束在参考策略附近，训练更保守

    参数:
        dataset: 偏好数据集
        betas: 要测试的 beta 值列表
        vocab_size: 词汇表大小
        num_epochs: 每个 beta 的训练轮数
        lr: 学习率

    返回:
        results: dict, key 是 beta 值, value 是 (losses, accuracies, reward_margins)
    """
    # TODO 3: Beta 敏感性分析
    # 提示：
    # 1. 对每个 beta 值：
    #    a. 创建一个全新的 policy（保证公平比较）
    #    b. 创建对应的 ref_policy（从同一初始化复制）
    #    c. 调用 train_dpo 训练
    #    d. 保存结果
    # ---- 在这里实现 ----
    results = {}

    for beta in betas:
        print(f"\n--- Beta = {beta} ---")

        # 每次都从全新的网络开始
        policy = SimplePolicyNet(vocab_size=vocab_size)
        ref_policy = SimplePolicyNet(vocab_size=vocab_size)
        ref_policy.load_state_dict(policy.state_dict())
        ref_policy.eval()
        for p in ref_policy.parameters():
            p.requires_grad = False

        losses, accuracies, reward_margins = train_dpo(
            policy, ref_policy, dataset,
            beta=beta, lr=lr, num_epochs=num_epochs, batch_size=32
        )

        results[beta] = (losses, accuracies, reward_margins)
        print(f"Beta={beta}: 最终准确率={accuracies[-1]:.2%}, "
              f"最终 Margin={reward_margins[-1]:.4f}")

    return results
    # ---- 代码结束 ----


# ============================================================
# Part 6: DPO vs 模拟 RLHF 对比 (TODO 4)
# ============================================================

def compare_dpo_vs_rlhf(dataset, vocab_size=20, num_epochs=50):
    """
    对比 DPO 和模拟 RLHF 的效果

    DPO 的优势：
    - 不需要单独的奖励模型（2个模型 vs 4个模型）
    - 不需要 PPO（更简单、更稳定）
    - 直接从偏好数据优化

    这里我们用一个简单的奖励模型来模拟 RLHF，
    比较两种方法在相同偏好数据上的偏好准确率。

    参数:
        dataset: 偏好数据集
        vocab_size: 词汇表大小
        num_epochs: 训练轮数

    返回:
        dpo_accuracy: DPO 最终偏好准确率
        rlhf_accuracy: 模拟 RLHF 最终偏好准确率
        dpo_losses: DPO 训练损失列表
        rlhf_losses: RLHF 奖励模型训练损失列表
    """
    # TODO 4: DPO vs RLHF 对比
    # 提示：
    # 1. 训练 DPO:
    #    a. 创建 policy 和 ref_policy
    #    b. 调用 train_dpo
    #    c. 最终准确率即 DPO 的隐式奖励准确率
    # 2. 训练简单奖励模型（模拟 RLHF）:
    #    a. 创建一个 nn.Sequential 奖励模型
    #       (Embedding -> 展平 -> Linear -> ReLU -> Linear -> 标量分数)
    #    b. 用 Bradley-Terry 损失训练
    #    c. 计算奖励模型的偏好准确率
    # ---- 在这里实现 ----

    # ===== 1. DPO 训练 =====
    print("=" * 50)
    print("训练 DPO...")
    print("=" * 50)

    policy = SimplePolicyNet(vocab_size=vocab_size)
    ref_policy = SimplePolicyNet(vocab_size=vocab_size)
    ref_policy.load_state_dict(policy.state_dict())
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    dpo_losses, dpo_accs, dpo_margins = train_dpo(
        policy, ref_policy, dataset,
        beta=0.1, lr=1e-3, num_epochs=num_epochs, batch_size=32
    )
    dpo_accuracy = dpo_accs[-1]

    # ===== 2. 模拟 RLHF: 训练奖励模型 =====
    print("\n" + "=" * 50)
    print("训练奖励模型（模拟 RLHF）...")
    print("=" * 50)

    seq_len = dataset[0][0].shape[0]

    # 简单的奖励模型
    reward_model = nn.Sequential(
        nn.Embedding(vocab_size, 32),           # (batch, seq_len, 32)
        nn.Flatten(),                            # (batch, seq_len * 32)
        nn.Linear(seq_len * 32, 64),
        nn.ReLU(),
        nn.Linear(64, 1)                         # (batch, 1) -> 标量奖励
    )

    rm_optimizer = optim.Adam(reward_model.parameters(), lr=1e-3)
    rlhf_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        indices = np.random.permutation(len(dataset))

        for start in range(0, len(dataset), 32):
            batch_idx = indices[start:start + 32]
            batch_chosen = torch.stack([dataset[i][0] for i in batch_idx])
            batch_rejected = torch.stack([dataset[i][1] for i in batch_idx])

            r_chosen = reward_model(batch_chosen).squeeze(-1)
            r_rejected = reward_model(batch_rejected).squeeze(-1)

            # Bradley-Terry 损失
            loss = -F.logsigmoid(r_chosen - r_rejected).mean()

            rm_optimizer.zero_grad()
            loss.backward()
            rm_optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)
            epoch_correct += (r_chosen > r_rejected).sum().item()

        avg_loss = epoch_loss / len(dataset)
        accuracy = epoch_correct / len(dataset)
        rlhf_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"RM Epoch {epoch + 1}/{num_epochs}, "
                  f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

    # 计算奖励模型最终偏好准确率
    reward_model.eval()
    correct = 0
    with torch.no_grad():
        for chosen, rejected in dataset:
            r_c = reward_model(chosen.unsqueeze(0)).item()
            r_r = reward_model(rejected.unsqueeze(0)).item()
            if r_c > r_r:
                correct += 1
    rlhf_accuracy = correct / len(dataset)

    print(f"\nDPO 偏好准确率: {dpo_accuracy:.2%}")
    print(f"RLHF (奖励模型) 偏好准确率: {rlhf_accuracy:.2%}")

    return dpo_accuracy, rlhf_accuracy, dpo_losses, rlhf_losses
    # ---- 代码结束 ----


# ============================================================
# Part 7: 可视化
# ============================================================

def plot_dpo_training(losses, accuracies, reward_margins):
    """
    绘制 DPO 训练过程的三个指标

    参数:
        losses: 每个 epoch 的损失
        accuracies: 每个 epoch 的偏好准确率
        reward_margins: 每个 epoch 的奖励差距
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 损失曲线
    ax = axes[0]
    ax.plot(losses, color='blue', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('DPO 损失')
    ax.set_title('DPO 训练损失')
    ax.grid(True, alpha=0.3)

    # 偏好准确率
    ax = axes[1]
    ax.plot(accuracies, color='green', alpha=0.7)
    ax.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5, label='随机基线')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('准确率')
    ax.set_title('偏好准确率 (chosen > rejected)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 奖励差距
    ax = axes[2]
    ax.plot(reward_margins, color='orange', alpha=0.7)
    ax.axhline(y=0, linestyle='--', color='gray', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('奖励差距')
    ax.set_title('隐式奖励差距 (chosen - rejected)')
    ax.grid(True, alpha=0.3)

    plt.suptitle('DPO 训练过程', fontsize=16)
    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_25/dpo_training.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: dpo_training.png")


def plot_beta_analysis(results):
    """
    绘制不同 beta 值的对比分析

    参数:
        results: dict, {beta: (losses, accuracies, reward_margins)}
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    # 损失曲线对比
    ax = axes[0]
    for (beta, (losses, _, _)), color in zip(sorted(results.items()), colors):
        ax.plot(losses, color=color, label=f'β={beta}', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('DPO 损失')
    ax.set_title('不同 β 的训练损失')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 准确率对比
    ax = axes[1]
    for (beta, (_, accs, _)), color in zip(sorted(results.items()), colors):
        ax.plot(accs, color=color, label=f'β={beta}', alpha=0.8)
    ax.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('准确率')
    ax.set_title('不同 β 的偏好准确率')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 最终指标柱状图
    ax = axes[2]
    betas_sorted = sorted(results.keys())
    final_accs = [results[b][1][-1] for b in betas_sorted]
    final_margins = [results[b][2][-1] for b in betas_sorted]
    x = np.arange(len(betas_sorted))
    width = 0.35

    bars1 = ax.bar(x - width / 2, final_accs, width, label='最终准确率', color='steelblue')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width / 2, final_margins, width, label='最终奖励差距', color='coral')

    ax.set_xlabel('β 值')
    ax.set_ylabel('准确率', color='steelblue')
    ax2.set_ylabel('奖励差距', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in betas_sorted])
    ax.set_title('β 敏感性分析')

    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.suptitle('DPO Beta 超参数分析', fontsize=16)
    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_25/dpo_beta_analysis.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: dpo_beta_analysis.png")


def plot_dpo_vs_rlhf_comparison(dpo_acc, rlhf_acc):
    """
    绘制 DPO vs RLHF 的对比柱状图

    参数:
        dpo_acc: DPO 的最终偏好准确率
        rlhf_acc: RLHF 的最终偏好准确率
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 准确率对比
    ax = axes[0]
    methods = ['DPO', 'RLHF\n(奖励模型)']
    accs = [dpo_acc, rlhf_acc]
    colors = ['#2196F3', '#FF9800']
    bars = ax.bar(methods, accs, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5, label='随机基线')
    ax.set_ylabel('偏好准确率')
    ax.set_title('偏好准确率对比')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱子上方标注数值
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

    # 复杂度对比
    ax = axes[1]
    categories = ['需要模型数量', '训练稳定性\n(1-5分)', '实现复杂度\n(1-5分, 越低越好)']
    dpo_scores = [2, 4.5, 1.5]
    rlhf_scores = [4, 3.0, 4.5]

    x = np.arange(len(categories))
    width = 0.3
    ax.bar(x - width / 2, dpo_scores, width, label='DPO', color='#2196F3', edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, rlhf_scores, width, label='RLHF', color='#FF9800', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('分数')
    ax.set_title('DPO vs RLHF 特性对比')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('DPO vs RLHF 综合对比', fontsize=16)
    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_25/dpo_vs_rlhf.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: dpo_vs_rlhf.png")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # ===== 步骤 1: 创建偏好数据集 =====
    print("=" * 60)
    print("步骤 1: 创建偏好数据集")
    print("=" * 60)

    # 创建初始策略（随机初始化，模拟 SFT 后的模型）
    init_policy = SimplePolicyNet(vocab_size=20)
    dataset = generate_preference_dataset(init_policy, num_pairs=500, seq_len=10)
    print(f"偏好数据集大小: {len(dataset)} 对")
    print(f"序列长度: {dataset[0][0].shape[0]}")

    # 展示几个样本
    print("\n偏好对示例:")
    for i in range(3):
        chosen, rejected = dataset[i]
        c_score = true_quality_score(chosen.numpy())
        r_score = true_quality_score(rejected.numpy())
        print(f"  Chosen:   {chosen.numpy()} (质量分: {c_score:.2f})")
        print(f"  Rejected: {rejected.numpy()} (质量分: {r_score:.2f})")
        print()

    # ===== 步骤 2: DPO 训练 =====
    print("=" * 60)
    print("步骤 2: DPO 训练")
    print("=" * 60)

    # 创建策略和参考策略
    policy = SimplePolicyNet(vocab_size=20)
    ref_policy = SimplePolicyNet(vocab_size=20)
    ref_policy.load_state_dict(policy.state_dict())
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    losses, accuracies, reward_margins = train_dpo(
        policy, ref_policy, dataset,
        beta=0.1, lr=1e-3, num_epochs=50, batch_size=32
    )

    plot_dpo_training(losses, accuracies, reward_margins)

    # ===== 步骤 3: Beta 敏感性分析 =====
    print("\n" + "=" * 60)
    print("步骤 3: Beta 敏感性分析")
    print("=" * 60)

    beta_results = analyze_beta_sensitivity(
        dataset, betas=[0.01, 0.05, 0.1, 0.5, 1.0],
        vocab_size=20, num_epochs=50
    )

    plot_beta_analysis(beta_results)

    # ===== 步骤 4: DPO vs RLHF 对比 =====
    print("\n" + "=" * 60)
    print("步骤 4: DPO vs RLHF 对比")
    print("=" * 60)

    dpo_acc, rlhf_acc, dpo_losses, rlhf_losses = compare_dpo_vs_rlhf(
        dataset, vocab_size=20, num_epochs=50
    )

    plot_dpo_vs_rlhf_comparison(dpo_acc, rlhf_acc)

    # ===== 总结 =====
    print("\n" + "=" * 60)
    print("DPO 训练完成！总结")
    print("=" * 60)
    print(f"DPO 最终训练损失: {losses[-1]:.4f}")
    print(f"DPO 最终偏好准确率: {accuracies[-1]:.2%}")
    print(f"DPO 最终奖励差距: {reward_margins[-1]:.4f}")
    print(f"\nBeta 敏感性分析:")
    for beta in sorted(beta_results.keys()):
        _, accs, margins = beta_results[beta]
        print(f"  β={beta:<5}: 准确率={accs[-1]:.2%}, 奖励差距={margins[-1]:.4f}")
    print(f"\nDPO vs RLHF 对比:")
    print(f"  DPO 偏好准确率:  {dpo_acc:.2%}")
    print(f"  RLHF 偏好准确率: {rlhf_acc:.2%}")
    print(f"\n关键要点:")
    print(f"  1. DPO 直接从偏好数据优化，无需训练奖励模型")
    print(f"  2. DPO 只需要 2 个模型（策略 + 参考策略），RLHF 需要 4 个")
    print(f"  3. beta 控制策略偏离参考模型的程度，需要合理调节")
    print(f"  4. DPO 训练更稳定，不存在 PPO 的超参数敏感问题")
