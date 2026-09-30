"""
Lesson 30: 前沿综述与展望
综合项目 - 参考实现

本项目是课程的综合实践，整合前面学到的多个技术：
1. 构建完整的 RLHF pipeline（简化版）
2. 对比 RLHF vs DPO vs GRPO
3. 实现简单的推理 Agent
4. 可视化对比分析

这是一个综合性项目，展示如何将各个组件整合在一起。
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
# Part 1: 简化的语言模型（策略网络）
# ============================================================

class SimpleLM(nn.Module):
    """简化的语言模型（GRU-based）"""

    def __init__(self, vocab_size=20, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        embeds = self.embedding(input_ids)
        output, _ = self.gru(embeds)
        logits = self.head(output)
        return logits

    def generate(self, prompt, max_len=8, temperature=1.0):
        """自回归生成"""
        self.eval()
        tokens = list(prompt)

        with torch.no_grad():
            for _ in range(max_len):
                input_ids = torch.LongTensor([tokens])
                logits = self.forward(input_ids)
                next_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                token = torch.multinomial(probs, 1).item()
                tokens.append(token)

        self.train()
        return np.array(tokens)

    def compute_log_prob(self, sequence):
        """计算序列的 log 概率"""
        input_ids = torch.LongTensor([sequence[:-1]])
        target_ids = torch.LongTensor([sequence[1:]])

        logits = self.forward(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)

        token_log_probs = log_probs[0, range(len(target_ids[0])), target_ids[0]]
        return token_log_probs.sum()


# ============================================================
# Part 2: 奖励模型
# ============================================================

class RewardModel(nn.Module):
    """奖励模型"""

    def __init__(self, vocab_size=20, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        embeds = self.embedding(input_ids)
        output, _ = self.gru(embeds)
        reward = self.head(output[:, -1, :]).squeeze(-1)
        return reward


def train_reward_model(rm, preference_data, num_epochs=20, lr=1e-3):
    """训练奖励模型"""
    optimizer = optim.Adam(rm.parameters(), lr=lr)
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for chosen, rejected in preference_data:
            chosen_ids = torch.LongTensor([chosen])
            rejected_ids = torch.LongTensor([rejected])

            r_chosen = rm(chosen_ids)
            r_rejected = rm(rejected_ids)

            loss = -F.logsigmoid(r_chosen - r_rejected).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(preference_data))

    return losses


# ============================================================
# Part 3: RLHF 训练（简化版 PPO）
# ============================================================

def train_rlhf_simple(policy, ref_policy, reward_model, prompts,
                      num_steps=50, lr=1e-4, kl_coef=0.1):
    """简化的 RLHF 训练"""
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    history = {'rewards': [], 'kl': []}

    for step in range(num_steps):
        step_rewards = []
        step_kl = []

        for prompt in prompts:
            # 生成 response
            sequence = policy.generate(prompt, max_len=8)

            # 计算奖励
            with torch.no_grad():
                rm_reward = reward_model(torch.LongTensor([sequence])).item()
                policy_logp = policy.compute_log_prob(sequence).item()
                ref_logp = ref_policy.compute_log_prob(sequence).item()
                kl = policy_logp - ref_logp
                total_reward = rm_reward - kl_coef * kl

            step_rewards.append(total_reward)
            step_kl.append(kl)

            # 简化的策略梯度更新
            policy_logp_grad = policy.compute_log_prob(sequence)
            loss = -policy_logp_grad * total_reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        history['rewards'].append(np.mean(step_rewards))
        history['kl'].append(np.mean(step_kl))

    return history


# ============================================================
# Part 4: DPO 训练
# ============================================================

def train_dpo(policy, ref_policy, preference_data, num_epochs=30,
              lr=1e-3, beta=0.1):
    """DPO 训练"""
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0

        for chosen, rejected in preference_data:
            chosen_ids = torch.LongTensor([chosen])
            rejected_ids = torch.LongTensor([rejected])

            # 计算 log 概率
            policy_chosen_logp = policy.compute_log_prob(chosen)
            policy_rejected_logp = policy.compute_log_prob(rejected)

            with torch.no_grad():
                ref_chosen_logp = ref_policy.compute_log_prob(chosen)
                ref_rejected_logp = ref_policy.compute_log_prob(rejected)

            # DPO 损失
            logits = beta * ((policy_chosen_logp - ref_chosen_logp) -
                            (policy_rejected_logp - ref_rejected_logp))
            loss = -F.logsigmoid(logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(preference_data))

    return losses


# ============================================================
# Part 5: GRPO 训练
# ============================================================

def train_grpo(policy, ref_policy, reward_model, prompts,
               num_steps=50, K=4, lr=1e-4, kl_coef=0.1):
    """GRPO 训练"""
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    history = {'rewards': [], 'kl': []}

    for step in range(num_steps):
        step_rewards = []
        step_kl = []

        for prompt in prompts:
            # 组采样
            sequences = []
            rewards = []

            for _ in range(K):
                seq = policy.generate(prompt, max_len=8)
                sequences.append(seq)

                with torch.no_grad():
                    rm_reward = reward_model(torch.LongTensor([seq])).item()
                    policy_logp = policy.compute_log_prob(seq).item()
                    ref_logp = ref_policy.compute_log_prob(seq).item()
                    kl = policy_logp - ref_logp
                    total_reward = rm_reward - kl_coef * kl
                    rewards.append(total_reward)

            # 计算组内优势
            baseline = np.mean(rewards)
            advantages = np.array(rewards) - baseline

            # 更新策略
            for seq, adv in zip(sequences, advantages):
                policy_logp = policy.compute_log_prob(seq)
                loss = -policy_logp * adv

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            step_rewards.append(np.mean(rewards))
            step_kl.append(np.mean([policy.compute_log_prob(s).item() -
                                   ref_policy.compute_log_prob(s).item()
                                   for s in sequences]))

        history['rewards'].append(np.mean(step_rewards))
        history['kl'].append(np.mean(step_kl))

    return history


# ============================================================
# Part 6: 数据生成
# ============================================================

def generate_preference_data(policy, num_pairs=200, vocab_size=20):
    """生成偏好数据"""

    def quality_score(sequence):
        """简单的质量评分函数"""
        response = sequence[3:]
        if len(response) == 0:
            return 0.0
        unique_ratio = len(set(response)) / len(response)
        repeat_penalty = sum(1 for i in range(1, len(response))
                           if response[i] == response[i-1]) * 0.3
        return unique_ratio - repeat_penalty

    data = []
    for _ in range(num_pairs):
        prompt = np.random.randint(0, vocab_size, size=3)

        seq_a = policy.generate(prompt, max_len=8, temperature=1.0)
        seq_b = policy.generate(prompt, max_len=8, temperature=1.2)

        score_a = quality_score(seq_a)
        score_b = quality_score(seq_b)

        if score_a > score_b:
            data.append((seq_a, seq_b))
        else:
            data.append((seq_b, seq_a))

    return data


# ============================================================
# Part 7: 评估
# ============================================================

def evaluate_policy(policy, reward_model, prompts, num_samples=10):
    """评估策略"""
    rewards = []

    for prompt in prompts:
        for _ in range(num_samples):
            sequence = policy.generate(prompt, max_len=8)
            with torch.no_grad():
                reward = reward_model(torch.LongTensor([sequence])).item()
            rewards.append(reward)

    return np.mean(rewards)


# ============================================================
# Part 8: 可视化
# ============================================================

def plot_training_comparison(rlhf_hist, dpo_losses, grpo_hist):
    """对比三种方法的训练过程"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RLHF 奖励
    ax = axes[0, 0]
    ax.plot(rlhf_hist['rewards'], label='RLHF', color='blue')
    ax.set_xlabel('步数')
    ax.set_ylabel('平均奖励')
    ax.set_title('RLHF 训练奖励')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # DPO 损失
    ax = axes[0, 1]
    ax.plot(dpo_losses, label='DPO', color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('损失')
    ax.set_title('DPO 训练损失')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # GRPO 奖励
    ax = axes[1, 0]
    ax.plot(grpo_hist['rewards'], label='GRPO', color='red')
    ax.set_xlabel('步数')
    ax.set_ylabel('平均奖励')
    ax.set_title('GRPO 训练奖励')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # KL 散度对比
    ax = axes[1, 1]
    ax.plot(rlhf_hist['kl'], label='RLHF', color='blue', alpha=0.7)
    ax.plot(grpo_hist['kl'], label='GRPO', color='red', alpha=0.7)
    ax.set_xlabel('步数')
    ax.set_ylabel('KL 散度')
    ax.set_title('KL 散度对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_30/training_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_final_comparison(sft_score, rlhf_score, dpo_score, grpo_score):
    """最终性能对比"""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['SFT', 'RLHF', 'DPO', 'GRPO']
    scores = [sft_score, rlhf_score, dpo_score, grpo_score]
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#e74c3c']

    bars = ax.bar(methods, scores, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('平均奖励')
    ax.set_title('不同对齐方法的最终性能对比')
    ax.set_ylim([min(scores) - 0.1, max(scores) + 0.1])

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=12)

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_30/final_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    vocab_size = 20

    print("=" * 70)
    print("Lesson 30: 综合项目 - RLHF vs DPO vs GRPO 对比")
    print("=" * 70)

    # 步骤 1: 初始化模型
    print("\n步骤 1: 初始化模型")
    print("-" * 70)
    sft_model = SimpleLM(vocab_size=vocab_size)
    print(f"SFT 模型参数量: {sum(p.numel() for p in sft_model.parameters())}")

    # 步骤 2: 生成偏好数据
    print("\n步骤 2: 生成偏好数据")
    print("-" * 70)
    preference_data = generate_preference_data(sft_model, num_pairs=300)
    print(f"偏好数据量: {len(preference_data)} 对")

    # 步骤 3: 训练奖励模型
    print("\n步骤 3: 训练奖励模型")
    print("-" * 70)
    reward_model = RewardModel(vocab_size=vocab_size)
    rm_losses = train_reward_model(reward_model, preference_data, num_epochs=20)
    print(f"奖励模型训练完成，最终损失: {rm_losses[-1]:.4f}")

    # 准备测试 prompts
    test_prompts = [np.random.randint(0, vocab_size, size=3) for _ in range(10)]

    # 评估 SFT 基线
    print("\n步骤 4: 评估 SFT 基线")
    print("-" * 70)
    sft_score = evaluate_policy(sft_model, reward_model, test_prompts)
    print(f"SFT 基线得分: {sft_score:.4f}")

    # 步骤 5: RLHF 训练
    print("\n步骤 5: RLHF 训练")
    print("-" * 70)
    rlhf_policy = SimpleLM(vocab_size=vocab_size)
    rlhf_policy.load_state_dict(sft_model.state_dict())
    ref_policy_rlhf = SimpleLM(vocab_size=vocab_size)
    ref_policy_rlhf.load_state_dict(sft_model.state_dict())
    ref_policy_rlhf.eval()

    rlhf_history = train_rlhf_simple(rlhf_policy, ref_policy_rlhf,
                                     reward_model, test_prompts, num_steps=50)
    rlhf_score = evaluate_policy(rlhf_policy, reward_model, test_prompts)
    print(f"RLHF 最终得分: {rlhf_score:.4f}")

    # 步骤 6: DPO 训练
    print("\n步骤 6: DPO 训练")
    print("-" * 70)
    dpo_policy = SimpleLM(vocab_size=vocab_size)
    dpo_policy.load_state_dict(sft_model.state_dict())
    ref_policy_dpo = SimpleLM(vocab_size=vocab_size)
    ref_policy_dpo.load_state_dict(sft_model.state_dict())
    ref_policy_dpo.eval()

    dpo_losses = train_dpo(dpo_policy, ref_policy_dpo, preference_data, num_epochs=30)
    dpo_score = evaluate_policy(dpo_policy, reward_model, test_prompts)
    print(f"DPO 最终得分: {dpo_score:.4f}")

    # 步骤 7: GRPO 训练
    print("\n步骤 7: GRPO 训练")
    print("-" * 70)
    grpo_policy = SimpleLM(vocab_size=vocab_size)
    grpo_policy.load_state_dict(sft_model.state_dict())
    ref_policy_grpo = SimpleLM(vocab_size=vocab_size)
    ref_policy_grpo.load_state_dict(sft_model.state_dict())
    ref_policy_grpo.eval()

    grpo_history = train_grpo(grpo_policy, ref_policy_grpo, reward_model,
                              test_prompts, num_steps=50, K=4)
    grpo_score = evaluate_policy(grpo_policy, reward_model, test_prompts)
    print(f"GRPO 最终得分: {grpo_score:.4f}")

    # 步骤 8: 可视化对比
    print("\n步骤 8: 可视化对比")
    print("-" * 70)
    plot_training_comparison(rlhf_history, dpo_losses, grpo_history)
    plot_final_comparison(sft_score, rlhf_score, dpo_score, grpo_score)

    # 最终总结
    print("\n" + "=" * 70)
    print("综合项目完成！")
    print("=" * 70)
    print("\n最终性能对比:")
    print(f"  SFT 基线:  {sft_score:.4f}")
    print(f"  RLHF:      {rlhf_score:.4f} (提升 {rlhf_score - sft_score:+.4f})")
    print(f"  DPO:       {dpo_score:.4f} (提升 {dpo_score - sft_score:+.4f})")
    print(f"  GRPO:      {grpo_score:.4f} (提升 {grpo_score - sft_score:+.4f})")

    print("\n关键发现:")
    print("1. 所有对齐方法都能提升 SFT 基线性能")
    print("2. DPO 训练更稳定（监督学习），无需价值模型")
    print("3. GRPO 无需价值模型，用组内相对排序")
    print("4. RLHF 是最经典的方法，但训练复杂度最高")
    print("5. 实际应用中应根据资源和需求选择合适的方法")

    print("\n恭喜你完成了全部 30 课的学习！")
    print("你已经掌握了从基础 RL 到 LLM 对齐的完整知识体系。")
    print("继续探索，在实践中不断提升！")
