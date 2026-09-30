"""
Lesson 27: GRPO (Group Relative Policy Optimization)
练习代码 - 参考实现

本练习实现简化版 GRPO 算法，包括：
1. 简化语言模型环境
2. 组采样（Group Sampling）
3. 组内相对优势计算
4. GRPO 训练（PPO-Clip 风格）
5. GRPO vs 标准 REINFORCE 对比
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

import copy


# ============================================================
# Part 1: SimplePolicyNet（策略网络）
# ============================================================
class SimplePolicyNet(nn.Module):
    """
    简化的策略网络（GRU-based 语言模型）
    用于生成 token 序列并计算 log prob
    """
    def __init__(self, vocab_size=20, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        """
        前向传播
        input_ids: (batch, seq_len) 或 (seq_len,) 的 LongTensor
        返回: logits (batch, seq_len, vocab_size) 或 (seq_len, vocab_size)
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        embeds = self.embedding(input_ids)           # (B, T, E)
        gru_out, _ = self.gru(embeds)                # (B, T, H)
        logits = self.output_head(gru_out)           # (B, T, V)

        if squeeze:
            logits = logits.squeeze(0)               # (T, V)
        return logits

    def generate(self, prompt, max_len=8, temperature=1.0):
        """
        自回归生成序列

        参数:
            prompt: numpy array，prompt token ids
            max_len: 生成的最大长度（不含 prompt）
            temperature: 采样温度

        返回:
            sequence: numpy array，完整序列（prompt + 生成部分）
            total_log_prob: float，所有生成 token 的 log prob 之和
        """
        self.eval()
        with torch.no_grad():
            tokens = list(prompt)
            total_log_prob = 0.0

            for _ in range(max_len):
                input_ids = torch.LongTensor(tokens).unsqueeze(0)   # (1, T)
                logits = self.forward(input_ids)                     # (1, T, V)
                # 取最后一个时间步的 logits
                next_logits = logits[0, -1, :] / temperature         # (V,)
                probs = F.softmax(next_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                token = dist.sample()
                log_p = dist.log_prob(token)

                tokens.append(token.item())
                total_log_prob += log_p.item()

        self.train()
        return np.array(tokens), total_log_prob

    def compute_log_prob(self, sequence):
        """
        计算给定序列在当前策略下的 log prob（带梯度）

        参数:
            sequence: numpy array，完整序列

        返回:
            total_log_prob: scalar tensor（带梯度）
        """
        input_ids = torch.LongTensor(sequence).unsqueeze(0)   # (1, T)
        logits = self.forward(input_ids)                        # (1, T, V)

        # 用 teacher forcing：对 t=0..T-2，预测 t+1
        log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)   # (T-1, V)
        targets = torch.LongTensor(sequence[1:])                # (T-1,)
        selected_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (T-1,)

        return selected_log_probs.sum()


# ============================================================
# Part 2: RewardFunction（奖励函数）
# ============================================================
class RewardFunction:
    """模拟奖励函数（代替奖励模型）"""
    def __init__(self, vocab_size=20):
        # 为每个 token 定义固定的 "质量" 分数
        np.random.seed(123)
        self.token_quality = np.random.randn(vocab_size) * 0.5

    def compute_reward(self, sequence):
        """
        计算序列奖励
        奖励 = 多样性 + token质量 - 重复惩罚
        """
        response = sequence[3:]  # 跳过 prompt
        if len(response) == 0:
            return 0.0

        # 多样性：唯一 token 比例
        unique_ratio = len(set(response)) / len(response)
        # token 质量：平均质量
        quality = np.mean([self.token_quality[t] for t in response])
        # 重复惩罚：连续重复
        repeat_penalty = sum(
            1 for i in range(1, len(response)) if response[i] == response[i - 1]
        ) * 0.3

        return unique_ratio + quality - repeat_penalty


# ============================================================
# Part 3: 组采样（Group Sampling）  [TODO 1]
# ============================================================
def group_sampling(policy, prompts, reward_fn, K=4, temperature=1.0, max_len=8):
    """
    组采样：对每个 prompt 生成 K 个 response 并计算奖励

    参数:
        policy: 策略网络
        prompts: list of numpy arrays (prompt sequences)
        reward_fn: 奖励函数
        K: 组大小
        temperature: 采样温度
        max_len: 生成最大长度

    返回:
        all_sequences: list of list of numpy arrays, shape [B][K]
        all_log_probs: list of list of floats, shape [B][K]
        all_rewards: numpy array, shape (B, K)

    提示:
        1. 对每个 prompt，生成 K 个 response（使用 policy.generate）
        2. 对每个 response，计算奖励（使用 reward_fn.compute_reward）
        3. 收集 sequences, log_probs, rewards
        4. 返回 all_sequences, all_log_probs, np.array(all_rewards)
    """
    # ---- 在这里实现 ----
    all_sequences = []
    all_log_probs = []
    all_rewards = []

    for prompt in prompts:
        seqs_k = []
        logps_k = []
        rewards_k = []

        for _ in range(K):
            # 生成一个 response
            seq, log_prob = policy.generate(prompt, max_len=max_len, temperature=temperature)
            # 计算奖励
            reward = reward_fn.compute_reward(seq)

            seqs_k.append(seq)
            logps_k.append(log_prob)
            rewards_k.append(reward)

        all_sequences.append(seqs_k)
        all_log_probs.append(logps_k)
        all_rewards.append(rewards_k)

    return all_sequences, all_log_probs, np.array(all_rewards)
    # ---- 代码结束 ----


# ============================================================
# Part 4: 计算组内相对优势  [TODO 2]
# ============================================================
def compute_group_advantages(rewards):
    """
    计算组内相对优势

    参数:
        rewards: numpy array (B, K) 奖励矩阵

    返回:
        advantages: numpy array (B, K) 归一化优势
    """
    # ---- 在这里实现 ----
    # 每个组的均值作为 baseline
    baselines = rewards.mean(axis=1, keepdims=True)  # (B, 1)
    # 优势 = 奖励 - baseline
    advantages = rewards - baselines                  # (B, K)
    # 全局归一化
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages
    # ---- 代码结束 ----


# ============================================================
# Part 5: GRPO 策略更新  [TODO 3]
# ============================================================
def grpo_update(policy, old_log_probs, sequences, advantages,
                optimizer, clip_range=0.2, ppo_epochs=3):
    """
    GRPO 策略更新（PPO-Clip 风格）

    参数:
        policy: 策略网络
        old_log_probs: list of floats - 旧 log probs (flattened B*K)
        sequences: list of numpy arrays - 所有序列 (flattened B*K)
        advantages: numpy array (B*K,) - 展平后的优势
        optimizer: 优化器
        clip_range: PPO clip 范围
        ppo_epochs: PPO 内循环次数

    返回:
        mean_loss: 平均损失
    """
    # ---- 在这里实现 ----
    adv_tensor = torch.FloatTensor(advantages)
    total_loss = 0.0
    num_updates = 0

    for _ in range(ppo_epochs):
        for i in range(len(sequences)):
            # 计算当前策略下的 log prob（带梯度）
            new_log_prob = policy.compute_log_prob(sequences[i])
            # 旧的 log prob（无梯度）
            old_lp = old_log_probs[i]

            # 重要性比率
            ratio = torch.exp(new_log_prob - old_lp)

            # PPO-Clip 目标
            adv_i = adv_tensor[i]
            surr1 = ratio * adv_i
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_i
            loss = -torch.min(surr1, surr2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_updates += 1

    mean_loss = total_loss / max(num_updates, 1)
    return mean_loss
    # ---- 代码结束 ----


# ============================================================
# Part 6: 完整 GRPO 训练流程  [TODO 4]
# ============================================================
def train_grpo(num_steps=100, batch_size=8, K=4, lr=1e-3,
               clip_range=0.2, ppo_epochs=2, temperature=0.8,
               kl_coef=0.1, vocab_size=20):
    """
    完整的 GRPO 训练流程

    返回:
        history: dict with 'mean_reward', 'max_reward', 'reward_std', 'loss', 'kl'
    """
    # ---- 在这里实现 ----
    # 1. 创建策略网络、参考策略（冻结）、奖励函数、优化器
    policy = SimplePolicyNet(vocab_size=vocab_size)
    ref_policy = copy.deepcopy(policy)
    for p in ref_policy.parameters():
        p.requires_grad = False
    ref_policy.eval()

    reward_fn = RewardFunction(vocab_size=vocab_size)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # 2. 生成固定的 prompt 集合
    prompts = [np.random.randint(0, vocab_size, 3) for _ in range(batch_size)]

    history = {
        'mean_reward': [],
        'max_reward': [],
        'reward_std': [],
        'loss': [],
        'kl': []
    }

    for step in range(num_steps):
        # 3a. 组采样
        all_sequences, all_log_probs, all_rewards = group_sampling(
            policy, prompts, reward_fn, K=K, temperature=temperature
        )

        # 3b. 计算 KL 惩罚并调整奖励
        kl_values = []
        penalized_rewards = np.copy(all_rewards)

        for b in range(batch_size):
            for k in range(K):
                seq = all_sequences[b][k]
                # 当前策略的 log prob
                with torch.no_grad():
                    policy_logp = policy.compute_log_prob(seq).item()
                    ref_logp = ref_policy.compute_log_prob(seq).item()
                kl = policy_logp - ref_logp
                kl_values.append(kl)
                penalized_rewards[b, k] = all_rewards[b, k] - kl_coef * kl

        # 3c. 计算组内相对优势
        advantages = compute_group_advantages(penalized_rewards)

        # 3d. 展平序列、log_probs、优势
        flat_sequences = []
        flat_log_probs = []
        for b in range(batch_size):
            for k in range(K):
                flat_sequences.append(all_sequences[b][k])
                flat_log_probs.append(all_log_probs[b][k])
        flat_advantages = advantages.flatten()

        # 3e. GRPO 更新
        loss = grpo_update(
            policy, flat_log_probs, flat_sequences, flat_advantages,
            optimizer, clip_range=clip_range, ppo_epochs=ppo_epochs
        )

        # 3f. 记录指标
        mean_reward = all_rewards.mean()
        max_reward = all_rewards.max()
        reward_std = all_rewards.std()
        mean_kl = np.mean(kl_values)

        history['mean_reward'].append(mean_reward)
        history['max_reward'].append(max_reward)
        history['reward_std'].append(reward_std)
        history['loss'].append(loss)
        history['kl'].append(mean_kl)

        if step % 10 == 0:
            print(f"  Step {step:3d} | "
                  f"Mean Reward: {mean_reward:.4f} | "
                  f"Max Reward: {max_reward:.4f} | "
                  f"KL: {mean_kl:.4f} | "
                  f"Loss: {loss:.4f}")

    return history
    # ---- 代码结束 ----


# ============================================================
# Part 7: REINFORCE 基线训练（对照实验）
# ============================================================
def train_reinforce(num_steps=100, batch_size=8, lr=1e-3, vocab_size=20):
    """标准 REINFORCE 训练（作为对照）"""
    policy = SimplePolicyNet(vocab_size=vocab_size)
    reward_fn = RewardFunction(vocab_size=vocab_size)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    prompts = [np.random.randint(0, vocab_size, 3) for _ in range(batch_size)]

    history = {
        'mean_reward': [],
        'max_reward': [],
        'reward_std': [],
        'loss': []
    }

    for step in range(num_steps):
        all_rewards = []
        all_log_probs = []
        all_sequences = []

        # 每个 prompt 生成一个 response
        for prompt in prompts:
            seq, log_prob = policy.generate(prompt, max_len=8, temperature=0.8)
            reward = reward_fn.compute_reward(seq)
            all_rewards.append(reward)
            all_log_probs.append(log_prob)
            all_sequences.append(seq)

        rewards_np = np.array(all_rewards)
        # 使用均值作为 baseline
        baseline = rewards_np.mean()
        advantages = rewards_np - baseline

        # REINFORCE 更新
        total_loss = 0.0
        for i in range(len(all_sequences)):
            log_prob = policy.compute_log_prob(all_sequences[i])
            loss = -log_prob * advantages[i]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mean_loss = total_loss / len(all_sequences)
        mean_reward = rewards_np.mean()
        max_reward = rewards_np.max()
        reward_std = rewards_np.std()

        history['mean_reward'].append(mean_reward)
        history['max_reward'].append(max_reward)
        history['reward_std'].append(reward_std)
        history['loss'].append(mean_loss)

        if step % 10 == 0:
            print(f"  Step {step:3d} | "
                  f"Mean Reward: {mean_reward:.4f} | "
                  f"Max Reward: {max_reward:.4f} | "
                  f"Loss: {mean_loss:.4f}")

    return history


# ============================================================
# Part 8: 可视化
# ============================================================
def plot_grpo_training(history):
    """绘制 GRPO 训练过程"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GRPO 训练过程', fontsize=16, fontweight='bold')

    # 平均奖励
    axes[0, 0].plot(history['mean_reward'], color='steelblue', linewidth=1.5)
    axes[0, 0].set_title('平均奖励')
    axes[0, 0].set_xlabel('训练步数')
    axes[0, 0].set_ylabel('奖励')
    axes[0, 0].grid(True, alpha=0.3)

    # 最大奖励
    axes[0, 1].plot(history['max_reward'], color='coral', linewidth=1.5)
    axes[0, 1].set_title('最大奖励')
    axes[0, 1].set_xlabel('训练步数')
    axes[0, 1].set_ylabel('奖励')
    axes[0, 1].grid(True, alpha=0.3)

    # 损失
    axes[1, 0].plot(history['loss'], color='forestgreen', linewidth=1.5)
    axes[1, 0].set_title('策略损失')
    axes[1, 0].set_xlabel('训练步数')
    axes[1, 0].set_ylabel('损失')
    axes[1, 0].grid(True, alpha=0.3)

    # KL 散度
    axes[1, 1].plot(history['kl'], color='purple', linewidth=1.5)
    axes[1, 1].set_title('KL 散度（策略 vs 参考）')
    axes[1, 1].set_xlabel('训练步数')
    axes[1, 1].set_ylabel('KL')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_27/grpo_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  训练曲线已保存到 e:/rl_lessons/lessons_27/grpo_training.png")


def plot_grpo_vs_reinforce(grpo_history, reinforce_history):
    """对比 GRPO 和 REINFORCE"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('GRPO vs REINFORCE 对比', fontsize=16, fontweight='bold')

    # 平均奖励对比
    axes[0].plot(grpo_history['mean_reward'], label='GRPO', color='steelblue', linewidth=1.5)
    axes[0].plot(reinforce_history['mean_reward'], label='REINFORCE', color='coral', linewidth=1.5)
    axes[0].set_title('平均奖励对比')
    axes[0].set_xlabel('训练步数')
    axes[0].set_ylabel('平均奖励')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 奖励标准差对比
    axes[1].plot(grpo_history['reward_std'], label='GRPO', color='steelblue', linewidth=1.5)
    axes[1].plot(reinforce_history['reward_std'], label='REINFORCE', color='coral', linewidth=1.5)
    axes[1].set_title('奖励标准差对比（训练稳定性）')
    axes[1].set_xlabel('训练步数')
    axes[1].set_ylabel('奖励标准差')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_27/grpo_vs_reinforce.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  对比图已保存到 e:/rl_lessons/lessons_27/grpo_vs_reinforce.png")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # 步骤 1: GRPO 训练
    print("=" * 60)
    print("步骤 1: GRPO 训练")
    print("=" * 60)
    grpo_history = train_grpo(num_steps=80, batch_size=8, K=4, lr=1e-3)
    plot_grpo_training(grpo_history)

    # 步骤 2: REINFORCE 训练（对照）
    print("\n" + "=" * 60)
    print("步骤 2: REINFORCE 训练（对照）")
    print("=" * 60)
    reinforce_history = train_reinforce(num_steps=80, batch_size=8, lr=1e-3)

    # 步骤 3: 对比
    print("\n" + "=" * 60)
    print("步骤 3: GRPO vs REINFORCE 对比")
    print("=" * 60)
    plot_grpo_vs_reinforce(grpo_history, reinforce_history)

    print(f"\nGRPO 最终平均奖励: {grpo_history['mean_reward'][-1]:.4f}")
    print(f"REINFORCE 最终平均奖励: {reinforce_history['mean_reward'][-1]:.4f}")
    print("\n关键发现：")
    print("1. GRPO 无需价值模型，用组内相对排序代替")
    print("2. GRPO 训练通常比 REINFORCE 更稳定（方差更低）")
    print("3. 组大小 K 越大，基线越准确，但计算开销越大")
