"""
Lesson 24: RLHF 详解
练习代码 - 参考实现

本练习模拟完整的 RLHF 训练流程，包括：
1. 模拟 SFT 模型（简单 RNN 语言模型）
2. 奖励模型训练（Bradley-Terry）
3. PPO 优化语言模型
4. KL 惩罚与自适应调节
5. Reward Hacking 分析
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
# Part 1: 简化语言模型环境
# ============================================================

class SimpleLMEnv:
    """
    简化的语言模型环境

    用离散序列模拟文本生成：
    - 词汇表大小: vocab_size
    - 序列长度: seq_len
    - 每个 token 是一个整数 [0, vocab_size)
    """

    def __init__(self, vocab_size=20, seq_len=8, num_prompts=100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        # 生成固定的 prompt 集合
        self.prompts = [
            np.random.randint(0, vocab_size, size=3) for _ in range(num_prompts)
        ]

    def sample_prompt(self):
        """随机采样一个 prompt"""
        idx = np.random.randint(len(self.prompts))
        return self.prompts[idx].copy()


# ============================================================
# Part 2: 策略网络（简化语言模型）
# ============================================================

class PolicyNetwork(nn.Module):
    """
    简化的策略网络（模拟语言模型）

    使用 GRU + Linear 生成 token 概率
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
            input_ids: (batch, seq_len) token 序列

        返回:
            logits: (batch, seq_len, vocab_size)
        """
        embeds = self.embedding(input_ids)
        output, _ = self.gru(embeds)
        logits = self.head(output)
        return logits

    def generate(self, prompt, max_len=8, temperature=1.0):
        """
        自回归生成序列

        参数:
            prompt: (prompt_len,) prompt token 序列
            max_len: 生成的最大长度
            temperature: 采样温度

        返回:
            sequence: 完整序列 (prompt + generated)
            log_probs: 每个生成 token 的 log 概率
        """
        self.eval()
        tokens = list(prompt)
        log_probs = []

        with torch.no_grad():
            for _ in range(max_len):
                input_ids = torch.LongTensor([tokens]).unsqueeze(0) if len(tokens) == 1 \
                    else torch.LongTensor([tokens])
                logits = self.forward(input_ids)
                # 取最后一个 token 的 logits
                next_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                token = torch.multinomial(probs, 1).item()
                log_prob = torch.log(probs[token]).item()

                tokens.append(token)
                log_probs.append(log_prob)

        self.train()
        return np.array(tokens), np.array(log_probs)

    def compute_log_probs(self, input_ids, response_ids):
        """
        计算给定 response 的 log 概率

        参数:
            input_ids: (batch, prompt_len) prompt
            response_ids: (batch, response_len) response

        返回:
            log_probs: (batch, response_len) 每个 token 的 log 概率
        """
        # 拼接 prompt + response
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        logits = self.forward(full_ids)

        # 只取 response 部分的 logits
        prompt_len = input_ids.shape[1]
        response_logits = logits[:, prompt_len-1:-1, :]  # 预测 response 的每个 token

        # 计算 log 概率
        log_probs = F.log_softmax(response_logits, dim=-1)
        # 取对应 token 的 log 概率
        log_probs = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)

        return log_probs


# ============================================================
# Part 3: 价值网络
# ============================================================

class ValueNetwork(nn.Module):
    """价值网络：估计状态的价值"""

    def __init__(self, vocab_size=20, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        embeds = self.embedding(input_ids)
        output, _ = self.gru(embeds)
        values = self.head(output).squeeze(-1)
        return values


# ============================================================
# Part 4: 奖励模型
# ============================================================

class RewardModel(nn.Module):
    """奖励模型：给 (prompt, response) 评分"""

    def __init__(self, vocab_size=20, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        """
        参数:
            input_ids: (batch, seq_len) 完整序列 (prompt + response)

        返回:
            reward: (batch,) 标量奖励
        """
        embeds = self.embedding(input_ids)
        output, _ = self.gru(embeds)
        # 取最后一个 token 的隐藏状态
        reward = self.head(output[:, -1, :]).squeeze(-1)
        return reward


def create_preference_data(env, sft_model, num_pairs=500):
    """
    创建偏好对比数据

    参数:
        env: 语言模型环境
        sft_model: SFT 模型
        num_pairs: 偏好对数量

    返回:
        data: [(prompt, chosen, rejected), ...]
    """
    # TODO 1: 创建偏好数据
    # 提示：
    # 1. 定义一个简单的"真实"偏好函数:
    #    prefer(seq) = -重复 token 数 + 多样性奖励
    # 2. 对每个 prompt 生成两个 response
    # 3. 用偏好函数判断哪个更好
    # ---- 在这里实现 ----
    def true_preference(sequence):
        """真实的人类偏好（模拟）：偏好多样、不重复的序列"""
        seq = sequence[3:]  # 去掉 prompt 部分
        if len(seq) == 0:
            return 0.0
        unique_ratio = len(set(seq)) / len(seq)  # 多样性
        # 连续重复惩罚
        repeat_penalty = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i-1])
        return unique_ratio * 2.0 - repeat_penalty * 0.5

    data = []
    for _ in range(num_pairs):
        prompt = env.sample_prompt()

        # 生成两个不同的 response
        seq_a, _ = sft_model.generate(prompt, max_len=env.seq_len, temperature=1.0)
        seq_b, _ = sft_model.generate(prompt, max_len=env.seq_len, temperature=1.2)

        score_a = true_preference(seq_a)
        score_b = true_preference(seq_b)

        if score_a > score_b:
            data.append((prompt, seq_a, seq_b))
        else:
            data.append((prompt, seq_b, seq_a))

    return data
    # ---- 代码结束 ----


def train_reward_model(rm, preference_data, num_epochs=30, lr=1e-3):
    """
    训练奖励模型

    参数:
        rm: 奖励模型
        preference_data: 偏好数据
        num_epochs: 训练轮数
        lr: 学习率

    返回:
        losses: 损失记录
        accuracies: 准确率记录
    """
    optimizer = optim.Adam(rm.parameters(), lr=lr)
    losses = []
    accuracies = []

    # TODO 2: 训练奖励模型
    # 提示：
    # 1. 遍历偏好数据
    # 2. 计算 chosen 和 rejected 的奖励
    # 3. Bradley-Terry 损失: -log(sigmoid(r_chosen - r_rejected))
    # 4. 记录准确率
    # ---- 在这里实现 ----
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_correct = 0

        # 打乱数据
        indices = np.random.permutation(len(preference_data))

        for idx in indices:
            prompt, chosen, rejected = preference_data[idx]

            chosen_ids = torch.LongTensor(chosen).unsqueeze(0)
            rejected_ids = torch.LongTensor(rejected).unsqueeze(0)

            r_chosen = rm(chosen_ids)
            r_rejected = rm(rejected_ids)

            # Bradley-Terry 损失
            loss = -F.logsigmoid(r_chosen - r_rejected).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += (r_chosen > r_rejected).float().sum().item()

        avg_loss = epoch_loss / len(preference_data)
        accuracy = epoch_correct / len(preference_data)
        losses.append(avg_loss)
        accuracies.append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f"RM Epoch {epoch+1}/{num_epochs}, "
                  f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

    return losses, accuracies
    # ---- 代码结束 ----


# ============================================================
# Part 5: PPO-RLHF 训练
# ============================================================

class AdaptiveKLController:
    """自适应 KL 系数控制器"""

    def __init__(self, init_kl_coef=0.1, target_kl=6.0, horizon=10000):
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.horizon = horizon

    def update(self, current_kl):
        """
        更新 KL 系数

        参数:
            current_kl: 当前的 KL 散度
        """
        # TODO 3: 实现自适应 KL 控制
        # 提示：
        # 1. 计算误差: error = current_kl - target_kl
        # 2. 按比例调节: kl_coef *= (1 + error / horizon)
        # 3. 确保 kl_coef > 0
        # ---- 在这里实现 ----
        proportional_error = (current_kl - self.target_kl) / self.target_kl
        self.kl_coef *= (1.0 + 0.1 * proportional_error)
        self.kl_coef = max(self.kl_coef, 0.001)
        # ---- 代码结束 ----


def ppo_rlhf_step(policy, ref_policy, value_net, reward_model,
                   env, kl_controller, policy_optimizer, value_optimizer,
                   batch_size=16, ppo_epochs=4, clip_range=0.2,
                   gamma=1.0, lam=0.95):
    """
    执行一步 PPO-RLHF 更新

    参数:
        policy: 策略网络
        ref_policy: 参考策略（冻结）
        value_net: 价值网络
        reward_model: 奖励模型（冻结）
        env: 环境
        kl_controller: KL 控制器
        policy_optimizer: 策略优化器
        value_optimizer: 价值优化器
        batch_size: 批大小
        ppo_epochs: PPO 内循环次数
        clip_range: PPO clip 范围

    返回:
        stats: 训练统计信息
    """
    # TODO 4: 实现 PPO-RLHF 更新步骤
    # 提示：
    # 1. 采样阶段：
    #    - 采样 prompt
    #    - 用 policy 生成 response
    #    - 用 reward_model 计算奖励
    #    - 计算 policy 和 ref_policy 的 log_probs
    #    - 计算 KL 惩罚: kl_penalty = kl_coef * (log_prob - ref_log_prob)
    #    - 计算总奖励: total_reward = rm_reward - kl_penalty.sum()
    # 2. 优势计算（简化版：单步奖励）
    # 3. PPO 更新：ratio, clipped loss
    # 4. 更新 KL 系数
    # ---- 在这里实现 ----

    # ===== 采样阶段 =====
    all_prompts = []
    all_responses = []
    all_old_log_probs = []
    all_ref_log_probs = []
    all_rm_rewards = []

    policy.eval()
    ref_policy.eval()
    reward_model.eval()

    for _ in range(batch_size):
        prompt = env.sample_prompt()
        sequence, log_probs = policy.generate(prompt, max_len=env.seq_len)

        # 参考模型的 log_probs
        prompt_ids = torch.LongTensor([prompt])
        response = sequence[len(prompt):]
        response_ids = torch.LongTensor([response])

        with torch.no_grad():
            ref_lp = ref_policy.compute_log_probs(prompt_ids, response_ids)

        # 奖励模型评分
        with torch.no_grad():
            rm_reward = reward_model(torch.LongTensor(sequence).unsqueeze(0)).item()

        all_prompts.append(prompt)
        all_responses.append(response)
        all_old_log_probs.append(log_probs)
        all_ref_log_probs.append(ref_lp.squeeze(0).numpy())
        all_rm_rewards.append(rm_reward)

    # 计算 KL 和总奖励
    kl_values = []
    total_rewards = []

    for i in range(batch_size):
        old_lp = all_old_log_probs[i]
        ref_lp = all_ref_log_probs[i]
        min_len = min(len(old_lp), len(ref_lp))
        kl = np.sum(old_lp[:min_len] - ref_lp[:min_len])
        kl_penalty = kl_controller.kl_coef * kl
        total_reward = all_rm_rewards[i] - kl_penalty
        kl_values.append(kl)
        total_rewards.append(total_reward)

    # ===== PPO 更新 =====
    policy.train()
    value_net.train()

    total_policy_loss = 0
    total_value_loss = 0

    # 归一化奖励
    rewards_array = np.array(total_rewards)
    if rewards_array.std() > 0:
        rewards_array = (rewards_array - rewards_array.mean()) / (rewards_array.std() + 1e-8)

    for ppo_epoch in range(ppo_epochs):
        for i in range(batch_size):
            prompt_ids = torch.LongTensor([all_prompts[i]])
            response_ids = torch.LongTensor([all_responses[i]])

            # 新的 log_probs
            new_log_probs = policy.compute_log_probs(prompt_ids, response_ids)
            old_log_probs_t = torch.FloatTensor(all_old_log_probs[i][:new_log_probs.shape[1]])

            # 策略比率
            ratio = (new_log_probs.squeeze(0) - old_log_probs_t).exp()
            advantage = rewards_array[i]

            # PPO-Clip 损失
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()

            policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            policy_optimizer.step()

            total_policy_loss += policy_loss.item()

    # 更新 KL 系数
    mean_kl = np.mean(kl_values)
    kl_controller.update(mean_kl)

    stats = {
        'rm_reward': np.mean(all_rm_rewards),
        'total_reward': np.mean(total_rewards),
        'kl': mean_kl,
        'kl_coef': kl_controller.kl_coef,
        'policy_loss': total_policy_loss / (ppo_epochs * batch_size),
    }

    return stats
    # ---- 代码结束 ----


# ============================================================
# Part 6: 完整训练流程
# ============================================================

def run_rlhf_training(num_rlhf_steps=100, batch_size=16):
    """运行完整的 RLHF 训练流程"""
    vocab_size = 20
    seq_len = 8

    # 创建环境
    env = SimpleLMEnv(vocab_size=vocab_size, seq_len=seq_len)

    # 1. 创建 SFT 模型（随机初始化作为简化的 SFT）
    print("=" * 60)
    print("步骤 1: 初始化 SFT 模型")
    print("=" * 60)
    sft_model = PolicyNetwork(vocab_size=vocab_size)
    print(f"SFT 模型参数量: {sum(p.numel() for p in sft_model.parameters())}")

    # 2. 创建偏好数据并训练奖励模型
    print("\n" + "=" * 60)
    print("步骤 2: 训练奖励模型")
    print("=" * 60)
    pref_data = create_preference_data(env, sft_model, num_pairs=500)
    print(f"偏好数据量: {len(pref_data)} 对")

    rm = RewardModel(vocab_size=vocab_size)
    rm_losses, rm_accs = train_reward_model(rm, pref_data, num_epochs=30, lr=1e-3)
    rm.eval()  # 冻结

    # 3. PPO-RLHF 训练
    print("\n" + "=" * 60)
    print("步骤 3: PPO-RLHF 训练")
    print("=" * 60)

    # 初始化策略（从 SFT 复制）
    policy = PolicyNetwork(vocab_size=vocab_size)
    policy.load_state_dict(sft_model.state_dict())

    # 参考模型（冻结）
    ref_policy = PolicyNetwork(vocab_size=vocab_size)
    ref_policy.load_state_dict(sft_model.state_dict())
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    # 价值网络
    value_net = ValueNetwork(vocab_size=vocab_size)

    # 优化器
    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

    # KL 控制器
    kl_controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=6.0)

    # 训练循环
    history = defaultdict(list)

    for step in range(num_rlhf_steps):
        stats = ppo_rlhf_step(
            policy, ref_policy, value_net, rm, env,
            kl_controller, policy_optimizer, value_optimizer,
            batch_size=batch_size, ppo_epochs=2, clip_range=0.2
        )

        for key, val in stats.items():
            history[key].append(val)

        if (step + 1) % 20 == 0:
            print(f"Step {step+1}/{num_rlhf_steps}, "
                  f"RM Reward: {stats['rm_reward']:.4f}, "
                  f"KL: {stats['kl']:.4f}, "
                  f"KL Coef: {stats['kl_coef']:.4f}")

    return history, rm_losses, rm_accs


# ============================================================
# Part 7: 可视化
# ============================================================

def plot_rlhf_training(history, rm_losses, rm_accs):
    """绘制完整的 RLHF 训练结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # RM 训练
    ax = axes[0, 0]
    ax.plot(rm_losses, color='blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('损失')
    ax.set_title('奖励模型训练损失')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(rm_accs, color='green')
    ax.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('准确率')
    ax.set_title('奖励模型准确率')
    ax.grid(True, alpha=0.3)

    # RLHF 训练
    ax = axes[0, 2]
    ax.plot(history['rm_reward'], color='blue', alpha=0.5)
    # 平滑
    if len(history['rm_reward']) >= 10:
        smoothed = np.convolve(history['rm_reward'],
                               np.ones(10)/10, mode='valid')
        ax.plot(range(9, len(history['rm_reward'])), smoothed,
                color='blue', linewidth=2, label='平滑')
    ax.set_xlabel('Step')
    ax.set_ylabel('RM 奖励')
    ax.set_title('RLHF: 奖励模型分数')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history['kl'], color='orange')
    ax.set_xlabel('Step')
    ax.set_ylabel('KL 散度')
    ax.set_title('RLHF: KL 散度')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(history['kl_coef'], color='red')
    ax.set_xlabel('Step')
    ax.set_ylabel('KL 系数')
    ax.set_title('RLHF: 自适应 KL 系数')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(history['policy_loss'], color='purple', alpha=0.5)
    if len(history['policy_loss']) >= 10:
        smoothed = np.convolve(history['policy_loss'],
                               np.ones(10)/10, mode='valid')
        ax.plot(range(9, len(history['policy_loss'])), smoothed,
                color='purple', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('策略损失')
    ax.set_title('RLHF: PPO 策略损失')
    ax.grid(True, alpha=0.3)

    plt.suptitle('RLHF 完整训练流程', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_24/rlhf_training.png',
                dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    history, rm_losses, rm_accs = run_rlhf_training(
        num_rlhf_steps=100, batch_size=16
    )

    plot_rlhf_training(history, rm_losses, rm_accs)

    print("\n" + "=" * 60)
    print("RLHF 训练完成！")
    print("=" * 60)
    print(f"奖励模型最终准确率: {rm_accs[-1]:.2%}")
    print(f"最终 RM 奖励: {history['rm_reward'][-1]:.4f}")
    print(f"最终 KL 散度: {history['kl'][-1]:.4f}")
    print(f"最终 KL 系数: {history['kl_coef'][-1]:.4f}")
    print("\n关键要点：")
    print("1. 奖励模型从偏好数据中学习评判标准")
    print("2. PPO 优化策略使 RM 奖励上升")
    print("3. 自适应 KL 控制器保持 KL 散度稳定")
    print("4. KL 约束防止策略偏离参考模型太远")
