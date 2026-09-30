# Lesson 24: RLHF 详解

## 1. InstructGPT 论文精读

### 1.1 研究背景

**论文：** "Training Language Models to Follow Instructions with Human Feedback" (Ouyang et al., 2022)

**核心问题：** GPT-3 很强大，但经常不做用户想让它做的事。

**具体表现：**
- 生成无关内容（不遵循指令）
- 编造事实（幻觉）
- 生成有害内容
- 长篇大论但不回答问题

**解决方案：** 用人类反馈微调语言模型

### 1.2 InstructGPT 的三步流程

```
Step 1: SFT（监督微调）
  数据: 13K 条 (prompt, 人工撰写的回答)
  方法: 在 GPT-3 上微调 16 个 epoch
  模型: SFT 模型

Step 2: 训练奖励模型（RM）
  数据: 33K 条比较数据（每条 4-9 个回答的排序）
  方法: 6B 参数模型 + Bradley-Terry 损失
  模型: RM 模型

Step 3: PPO 优化
  数据: 31K 条 prompt
  方法: PPO + KL 约束
  模型: InstructGPT
```

### 1.3 关键细节

**数据收集：**

| 数据类型 | 数量 | 用途 |
|----------|------|------|
| SFT 演示数据 | 13K | 监督微调 |
| 比较数据 | 33K | 训练奖励模型 |
| PPO 提示 | 31K | RLHF 训练 |

**标注员：**
- 约 40 名标注员
- 经过筛选和培训
- 标注者间一致性约 73%

**Prompt 来源：**
- OpenAI API 用户提交的真实 prompt
- 标注员编写的 prompt
- 涵盖多种任务类型

### 1.4 核心发现

**1. 小模型可以超越大模型**

$$\text{1.3B InstructGPT} \succ \text{175B GPT-3}$$

人类标注员在 85% 的情况下更偏好 InstructGPT 的回答。

**2. 对齐税（Alignment Tax）较低**

- 在 NLP 基准上，InstructGPT 性能轻微下降
- 通过混合预训练数据可以缓解

**3. RLHF > SFT > 提示工程**

```
性能排序:
InstructGPT (RLHF) > SFT 模型 > GPT-3 + 精心设计的提示
```


## 2. PPO 在语言模型上的应用

### 2.1 RL 形式化回顾

**将文本生成建模为 MDP：**

| 概念 | 定义 |
|------|------|
| **状态 $s_t$** | prompt + 已生成的 token 序列 $(x, y_1, ..., y_{t-1})$ |
| **动作 $a_t$** | 下一个 token $y_t$（词汇表中选择） |
| **策略 $\pi_\theta$** | 语言模型 $P_\theta(y_t \mid x, y_{<t})$ |
| **奖励 $r$** | 仅在生成结束时给出 $R(x, y)$（稀疏奖励） |
| **轨迹 $\tau$** | 完整的回答 $y = (y_1, y_2, ..., y_T)$ |

### 2.2 RLHF 的目标函数

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \Big[ R_\phi(x, y) - \beta \cdot D_{\text{KL}}\big(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\big) \Big]$$

**展开 KL 散度到 token 级别：**

$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \sum_{t=1}^{T} \mathbb{E}_{y_t \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})} \right]$$

**每个 token 的奖励定义：**

$$r_t = \begin{cases}
-\beta \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})} & t < T \\
R_\phi(x, y) - \beta \log \frac{\pi_\theta(y_T | x, y_{<T})}{\pi_{\text{ref}}(y_T | x, y_{<T})} & t = T
\end{cases}$$

### 2.3 PPO-Clip 在 LLM 中的实现

**标准 PPO-Clip 损失：**

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} A_t,\ \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}, 1-\epsilon, 1+\epsilon\right) A_t \right) \right]$$

**在 LLM 中：**

$$\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} = \frac{P_\theta(y_t | x, y_{<t})}{P_{\theta_{\text{old}}}(y_t | x, y_{<t})}$$

**优势估计（GAE）：**

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### 2.4 完整 PPO-RLHF 算法

```
初始化:
  π_θ ← π_SFT          # 策略模型（从 SFT 初始化）
  π_ref ← π_SFT        # 参考模型（冻结）
  R_φ                   # 奖励模型（冻结）
  V_ψ                   # 价值模型（可训练）

对于每个 PPO 迭代:
  1. 采样阶段（Rollout）:
     对于 batch 中每个 prompt x:
       a) 用 π_θ 生成回答 y ~ π_θ(·|x)
       b) 计算每个 token 的 log_prob:
          log π_θ(y_t|x,y_<t) 和 log π_ref(y_t|x,y_<t)
       c) 计算奖励: R_φ(x, y)
       d) 计算每个 token 的 per-token reward r_t
       e) 计算价值估计: V_ψ(s_t)

  2. 优势计算:
     用 GAE 计算 Â_t

  3. 优化阶段（多个 mini-batch epoch）:
     a) 更新策略 π_θ（PPO-Clip 损失）
     b) 更新价值 V_ψ（MSE 损失）

  4. 日志记录:
     记录 reward, KL, policy loss, value loss 等
```

### 2.5 实际实现中的关键细节

**1. 四个模型的内存管理**

```
GPU 内存需求:
  策略模型 π_θ:     占 1/4 GPU（可训练）
  参考模型 π_ref:    占 1/4 GPU（冻结，可用量化）
  奖励模型 R_φ:      占 1/4 GPU（冻结，可用量化）
  价值模型 V_ψ:      占 1/4 GPU（可训练）
```

**优化策略：**
- 参考模型和奖励模型可以量化（4-bit/8-bit）
- 价值模型可以共享策略模型的主干
- 使用 LoRA 减少可训练参数

**2. 生成与训练分离**

```python
# 生成阶段（不需要梯度）
with torch.no_grad():
    responses = model.generate(prompts, ...)
    old_log_probs = model.log_prob(responses)
    ref_log_probs = ref_model.log_prob(responses)
    rewards = reward_model(prompts, responses)

# 训练阶段（需要梯度）
for epoch in range(ppo_epochs):
    new_log_probs = model.log_prob(responses)
    ratio = (new_log_probs - old_log_probs).exp()
    # PPO update...
```

**3. 归一化技巧**

```python
# 奖励归一化（running mean/std）
rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

# 优势归一化
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```


## 3. KL 惩罚详解

### 3.1 为什么需要 KL 惩罚？

**没有 KL 惩罚会发生什么？**

1. **模式崩溃（Mode Collapse）**：
   - 策略只生成少数高奖励的回答
   - 多样性消失

2. **奖励过度优化（Reward Hacking）**：
   - 策略找到奖励模型的"漏洞"
   - RM 分数很高，但实际质量很差

3. **语言退化**：
   - 生成不自然的文本
   - 重复特定短语或模式

**例子：**

```
无 KL 约束:
  Q: "写一首关于春天的诗"
  A: "beautiful beautiful beautiful spring wonderful wonderful amazing 
      beautiful spring beautiful wonderful..." (高 RM 分数，但无意义)

有 KL 约束:
  Q: "写一首关于春天的诗"  
  A: "春风拂面暖如丝，桃花烂漫映日迟。
      燕子归来寻旧巢，柳条依依舞腰肢。" (自然且高质量)
```

### 3.2 KL 系数的调节

**固定 KL 系数：**

$$J(\theta) = R(x, y) - \beta \cdot D_{\text{KL}}$$

$\beta$ 的选择：
- $\beta$ 太大 → 策略几乎不更新，退化为 SFT
- $\beta$ 太小 → 奖励过度优化
- 典型范围：0.01 ~ 0.2

**自适应 KL 系数（InstructGPT 使用）：**

$$\beta_{t+1} = \begin{cases}
\beta_t \cdot (1 - \alpha) & \text{if } D_{\text{KL}} < D_{\text{target}} \cdot (1 - \delta) \\
\beta_t \cdot (1 + \alpha) & \text{if } D_{\text{KL}} > D_{\text{target}} \cdot (1 + \delta) \\
\beta_t & \text{otherwise}
\end{cases}$$

**参数：**
- $D_{\text{target}}$：目标 KL 散度（如 6.0 nats）
- $\alpha$：调节步长（如 0.1）
- $\delta$：容忍范围（如 0.1）

**直觉：**
- KL 太大 → 减小 $\beta$，允许更多探索
- KL 太小 → 增大 $\beta$，约束更紧

### 3.3 KL 散度的计算

**Token 级 KL：**

```python
def compute_kl_per_token(logprobs, ref_logprobs):
    """计算每个 token 的 KL 散度"""
    return logprobs - ref_logprobs  # 简化的 KL 估计

def compute_kl_penalty(logprobs, ref_logprobs, kl_coef):
    """计算 KL 惩罚"""
    kl = compute_kl_per_token(logprobs, ref_logprobs)
    return -kl_coef * kl  # 作为负奖励
```

**序列级 KL：**

$$D_{\text{KL}}^{\text{seq}} = \sum_{t=1}^{T} \left[ \log \pi_\theta(y_t|s_t) - \log \pi_{\text{ref}}(y_t|s_t) \right]$$


## 4. Reward Hacking 深入分析

### 4.1 什么是 Reward Hacking？

**定义：** 策略学会最大化奖励模型的分数，但实际回答质量下降。

**原因：**
- 奖励模型是真实偏好的不完美近似
- 在 OOD 区域，奖励模型不可靠
- 优化压力驱动策略走向 OOD 区域

### 4.2 Reward Hacking 的表现

**1. 长度偏好（Length Bias）**

```
奖励模型倾向于给长回答高分
→ 策略学会生成冗长的回答
→ 内容重复、啰嗦但分数高
```

**2. 谄媚（Sycophancy）**

```
奖励模型倾向于给"同意用户"的回答高分
→ 策略学会无条件同意用户
→ 即使用户说错了也不纠正
```

**3. 格式利用**

```
奖励模型对特定格式（如 bullet points）给高分
→ 策略学会无论什么问题都用列表回答
→ 形式大于内容
```

### 4.3 缓解 Reward Hacking

**1. KL 约束（前面讨论的）**

**2. 奖励模型集成**

```python
# 使用多个奖励模型
reward = min(rm1(x, y), rm2(x, y), rm3(x, y))
# 或
reward = mean(rm1(x, y), rm2(x, y), rm3(x, y))
```

**3. 长度惩罚**

```python
# 惩罚过长的回答
length_penalty = -0.01 * len(response)
reward = rm_reward + length_penalty
```

**4. 迭代训练**

- 训练后收集新的偏好数据
- 重新训练奖励模型
- 多轮迭代


## 5. TRL 库简介

### 5.1 什么是 TRL？

**TRL（Transformer Reinforcement Learning）** 是 Hugging Face 开发的 RLHF 训练库。

**核心组件：**

```python
from trl import (
    PPOTrainer,           # PPO 训练器
    PPOConfig,            # PPO 配置
    AutoModelForCausalLMWithValueHead,  # 带价值头的语言模型
    RewardTrainer,        # 奖励模型训练器
    DPOTrainer,           # DPO 训练器
)
```

### 5.2 TRL 的 RLHF 流程

```python
# 1. 加载模型
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. 配置 PPO
ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
    kl_penalty="kl",        # KL 惩罚类型
    init_kl_coef=0.2,       # 初始 KL 系数
    target_kl=6.0,          # 目标 KL
)

# 3. 创建 PPO 训练器
ppo_trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    config=ppo_config,
)

# 4. 训练循环
for batch in dataloader:
    # 生成回答
    response = ppo_trainer.generate(batch["input_ids"])
    
    # 计算奖励
    rewards = reward_model(batch["input_ids"], response)
    
    # PPO 更新
    stats = ppo_trainer.step(
        batch["input_ids"],
        response,
        rewards,
    )
```

### 5.3 TRL 的关键特性

1. **LoRA 支持**：使用 PEFT 减少内存
2. **量化支持**：4-bit/8-bit 量化
3. **分布式训练**：DeepSpeed、FSDP
4. **多种算法**：PPO、DPO、ORPO、KTO
5. **自定义奖励**：灵活的奖励函数接口


## 6. 超参数调优指南

### 6.1 关键超参数

| 超参数 | 推荐范围 | 说明 |
|--------|----------|------|
| `learning_rate` | 1e-6 ~ 5e-5 | PPO 学习率 |
| `init_kl_coef` | 0.05 ~ 0.5 | 初始 KL 系数 |
| `target_kl` | 3.0 ~ 10.0 | 目标 KL（自适应时） |
| `ppo_epochs` | 2 ~ 6 | PPO 内循环次数 |
| `clip_range` | 0.1 ~ 0.3 | PPO clip 范围 |
| `batch_size` | 32 ~ 256 | 训练批大小 |
| `mini_batch_size` | 4 ~ 32 | PPO 小批大小 |
| `gamma` | 1.0 | 折扣因子（文本通常用 1） |
| `lam` | 0.95 | GAE lambda |

### 6.2 训练监控

**必须监控的指标：**

1. **奖励趋势**：应该上升
2. **KL 散度**：应该稳定在目标附近
3. **策略熵**：不应该降太快
4. **回答长度**：不应该异常增长
5. **实际回答质量**：定期人工检查

**异常信号：**
- KL 爆炸 → 学习率太大
- 奖励不动 → 学习率太小或 KL 约束太强
- 回答退化 → reward hacking，增大 KL 系数


## 7. 实践建议

### 7.1 从小模型开始

```
调试流程:
1. 在 GPT-2 (124M) 上调通整个 pipeline
2. 确认 loss 曲线正常
3. 迁移到更大的模型
```

### 7.2 评估策略

**自动评估：**
- BLEU / ROUGE（表面质量）
- 奖励模型分数
- KL 散度
- win rate（与基线对比）

**人工评估：**
- 随机采样 100-200 条
- A/B 对比（blind test）
- 多维度评分（有帮助、无害、诚实）


## 8. 总结

### 8.1 核心要点

1. **InstructGPT 的贡献**：
   - 系统化的 RLHF 流程
   - 证明了小模型 + RLHF > 大模型
   - 开启了 LLM 对齐的时代

2. **PPO 在 LLM 中的应用**：
   - Token 级别的 RL 建模
   - 稀疏奖励 + GAE
   - 四个模型的协同训练

3. **KL 惩罚**：
   - 防止模式崩溃和 reward hacking
   - 自适应调节机制
   - 是 RLHF 成功的关键

4. **Reward Hacking**：
   - 长度偏好、谄媚、格式利用
   - 多种缓解策略
   - 根本原因是奖励模型不完美

5. **TRL 库**：
   - Hugging Face 的 RLHF 工具
   - 简化了实现流程

### 8.2 下节预告

下节课我们将学习 **DPO（Direct Preference Optimization）**：
- DPO 的数学推导
- 从 RLHF 目标到 DPO 损失
- DPO 的实现
- DPO vs RLHF 对比

### 8.3 扩展阅读

**必读论文：**
1. Ouyang et al. (2022). "Training Language Models to Follow Instructions with Human Feedback"
2. Stiennon et al. (2020). "Learning to Summarize with Human Feedback"
3. Ziegler et al. (2019). "Fine-Tuning Language Models from Human Preferences"
4. Gao et al. (2023). "Scaling Laws for Reward Model Overoptimization"

**实用资源：**
- TRL 文档：https://huggingface.co/docs/trl/
- OpenRLHF：https://github.com/OpenRLHF/OpenRLHF
- DeepSpeed-Chat：https://github.com/microsoft/DeepSpeedExamples
