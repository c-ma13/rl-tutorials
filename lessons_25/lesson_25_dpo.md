# Lesson 25: DPO (Direct Preference Optimization)

## 1. RLHF 的局限性

### 1.1 RLHF 的复杂性

回顾 RLHF 的完整流程：

```
1. SFT 训练
2. 收集偏好数据
3. 训练奖励模型 R_φ
4. 用 PPO 优化策略 π_θ（需要参考模型 π_ref 和价值模型 V_ψ）
```

**问题：**
- 需要训练**两个模型**（奖励模型 + 策略模型）
- PPO 训练需要**四个模型**同时在内存中
- 训练不稳定（PPO 对超参数敏感）
- 奖励模型可能不准确（reward hacking）

### 1.2 能否简化？

**核心问题：** 能否直接从偏好数据优化策略，跳过奖励模型？

$$\text{偏好数据} \xrightarrow{\text{?}} \text{策略}$$

**DPO 的答案：** 可以！通过重新参数化 RLHF 目标。


## 2. DPO 的数学推导

### 2.1 RLHF 目标回顾

RLHF 的目标是最大化：

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ R_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right]$$

其中：
- $R_\phi(x, y)$：奖励模型
- $\beta$：KL 惩罚系数
- $\pi_{\text{ref}}$：参考策略（SFT 模型）

### 2.2 最优策略的闭式解

**关键洞察（Rafailov et al., 2023）：** 上述目标有闭式解！

**定理：** RLHF 目标的最优策略为：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} R(x, y)\right)$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} R(x, y)\right)$ 是配分函数。

**证明思路：**

这是一个约束优化问题（KL 约束下最大化奖励）。用拉格朗日乘数法：

$$\mathcal{L} = \mathbb{E}_{y \sim \pi}[R(x,y)] - \beta D_{\text{KL}}(\pi \| \pi_{\text{ref}})$$

展开 KL 散度：

$$= \mathbb{E}_{y \sim \pi}\left[R(x,y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right] + \beta \log Z(x)$$

对 $\pi(y|x)$ 求导并令其为零，得到最优解。

### 2.3 从最优策略反解奖励

**重新整理最优策略公式：**

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} R(x, y)\right)$$

两边取对数：

$$\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} R(x, y) - \log Z(x)$$

**解出奖励：**

$$R(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

**关键简化：** $\log Z(x)$ 只依赖于 $x$，在比较同一 prompt 的不同回答时会**抵消**！

### 2.4 Bradley-Terry 模型下的 DPO 损失

**偏好概率（Bradley-Terry）：**

$$P(y_w \succ y_l | x) = \sigma(R(x, y_w) - R(x, y_l))$$

**代入隐式奖励：**

$$R(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

$$R(x, y_w) - R(x, y_l) = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$$

（$\log Z(x)$ 项抵消）

**DPO 损失函数：**

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

**简化记号：**

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right) \right]$$


## 3. DPO 算法

### 3.1 DPO 训练流程

```
输入:
  - 偏好数据集 D = {(x, y_w, y_l)}
  - 参考模型 π_ref（SFT 模型，冻结）
  - 超参数 β

初始化:
  π_θ ← π_ref（从 SFT 模型初始化）

对于每个训练步:
  1. 采样 batch: (x, y_w, y_l) ~ D
  
  2. 计算 log 概率:
     log π_θ(y_w|x), log π_θ(y_l|x)
     log π_ref(y_w|x), log π_ref(y_l|x)
  
  3. 计算隐式奖励差:
     Δr = β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x))
  
  4. 计算 DPO 损失:
     L = -log σ(Δr)
  
  5. 反向传播更新 θ
```

### 3.2 DPO vs RLHF 对比

| 方面 | RLHF | DPO |
|------|------|-----|
| **模型数量** | 4 个（策略、参考、奖励、价值） | 2 个（策略、参考） |
| **训练阶段** | 3 阶段（SFT → RM → PPO） | 2 阶段（SFT → DPO） |
| **稳定性** | PPO 不稳定 | 监督学习，稳定 |
| **内存占用** | 高（4 个模型） | 低（2 个模型） |
| **超参数** | 多（PPO 相关） | 少（主要是 β） |
| **奖励模型** | 需要显式训练 | 隐式（策略比率） |
| **理论保证** | 收敛性弱 | 等价于 RLHF 最优解 |

### 3.3 DPO 的优势

1. **简单**：
   - 不需要训练奖励模型
   - 不需要 PPO（监督学习即可）
   - 不需要价值模型

2. **稳定**：
   - 标准的监督学习
   - 梯度稳定
   - 易于调试

3. **高效**：
   - 内存占用减半
   - 训练速度更快
   - 更容易扩展到大模型

4. **理论优雅**：
   - 直接优化 RLHF 目标
   - 有理论保证

### 3.4 DPO 的局限

1. **依赖 SFT 质量**：
   - 参考模型 $\pi_{\text{ref}}$ 必须是好的 SFT 模型
   - 如果 SFT 很差，DPO 也无法救

2. **无法在线学习**：
   - 只能从固定数据集学习
   - 不能像 RLHF 那样在线采样

3. **对 β 敏感**：
   - β 太大：策略不动
   - β 太小：过度偏离参考模型

4. **长度偏差**：
   - 可能倾向于生成更长的回答
   - 需要额外的长度惩罚


## 4. DPO 实现细节

### 4.1 计算 Log 概率

```python
def compute_log_probs(model, input_ids, labels):
    """
    计算序列的 log 概率
    
    参数:
        model: 语言模型
        input_ids: (batch, seq_len) prompt + response
        labels: (batch, seq_len) 标签（prompt 部分为 -100）
    
    返回:
        log_probs: (batch,) 每个序列的总 log 概率
    """
    outputs = model(input_ids, labels=labels)
    logits = outputs.logits  # (batch, seq_len, vocab_size)
    
    # 计算每个 token 的 log 概率
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 取对应 token 的 log 概率
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    
    # 只对 response 部分求和（labels != -100）
    mask = (labels != -100).float()
    sequence_log_prob = (token_log_probs * mask).sum(dim=1)
    
    return sequence_log_prob
```

### 4.2 DPO 损失计算

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta=0.1):
    """
    计算 DPO 损失
    
    参数:
        policy_chosen_logps: π_θ(y_w|x) 的 log 概率
        policy_rejected_logps: π_θ(y_l|x) 的 log 概率
        reference_chosen_logps: π_ref(y_w|x) 的 log 概率
        reference_rejected_logps: π_ref(y_l|x) 的 log 概率
        beta: 温度参数
    
    返回:
        loss: DPO 损失
        chosen_rewards: chosen 的隐式奖励
        rejected_rewards: rejected 的隐式奖励
    """
    # 计算策略比率的 log
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps
    
    # 隐式奖励
    logits = beta * (policy_logratios - reference_logratios)
    
    # DPO 损失
    loss = -F.logsigmoid(logits).mean()
    
    # 计算隐式奖励（用于监控）
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    
    return loss, chosen_rewards, rejected_rewards
```

### 4.3 完整训练循环

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

# 加载模型
model = AutoModelForCausalLM.from_pretrained("gpt2")
ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
ref_model.eval()  # 冻结参考模型

tokenizer = AutoTokenizer.from_pretrained("gpt2")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

# 训练循环
for batch in dataloader:
    # batch 包含: prompt, chosen, rejected
    
    # 1. Tokenize
    chosen_ids = tokenizer(batch["chosen"], return_tensors="pt")
    rejected_ids = tokenizer(batch["rejected"], return_tensors="pt")
    
    # 2. 计算策略模型的 log 概率
    policy_chosen_logps = compute_log_probs(model, chosen_ids)
    policy_rejected_logps = compute_log_probs(model, rejected_ids)
    
    # 3. 计算参考模型的 log 概率
    with torch.no_grad():
        ref_chosen_logps = compute_log_probs(ref_model, chosen_ids)
        ref_rejected_logps = compute_log_probs(ref_model, rejected_ids)
    
    # 4. 计算 DPO 损失
    loss, _, _ = dpo_loss(
        policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        beta=0.1
    )
    
    # 5. 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```


## 5. 超参数调优

### 5.1 关键超参数

**β（温度参数）：**
- 控制策略偏离参考模型的程度
- 典型范围：0.01 ~ 0.5
- β 越大 → 越保守（接近 SFT）
- β 越小 → 越激进（可能过拟合偏好数据）

**学习率：**
- 比 SFT 小一个数量级
- 典型：1e-7 ~ 1e-5
- 太大会导致训练不稳定

**Batch Size：**
- 越大越稳定
- 典型：32 ~ 128
- 受 GPU 内存限制

### 5.2 β 的选择

**经验法则：**

```python
# 根据数据集大小调整
if dataset_size < 10K:
    beta = 0.1
elif dataset_size < 100K:
    beta = 0.05
else:
    beta = 0.01
```

**自适应 β（实验性）：**

```python
# 根据 KL 散度动态调整
kl = compute_kl(policy, ref_policy)
if kl > target_kl:
    beta *= 1.1  # 增大约束
else:
    beta *= 0.9  # 放松约束
```


## 6. 评估与监控

### 6.1 训练指标

**必须监控：**

1. **DPO 损失**：应该下降
2. **准确率**：$P(\text{chosen reward} > \text{rejected reward})$
3. **隐式奖励差**：chosen - rejected 的平均差值
4. **KL 散度**：$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$

**警告信号：**
- 准确率 < 60%：模型没学到东西
- KL 爆炸：β 太小或学习率太大
- 奖励差不增长：β 太大

### 6.2 评估方法

**自动评估：**
- Win rate（与基线对比）
- GPT-4 评分
- 奖励模型评分（如果有）

**人工评估：**
- A/B 测试
- 多维度评分（有帮助、无害、诚实）


## 7. DPO 的实际应用

### 7.1 成功案例

**Zephyr（HuggingFace, 2023）：**
- 7B 模型
- 用 DPO 微调 Mistral-7B
- 在 MT-Bench 上超越 Llama-2-70B-Chat

**Starling（UC Berkeley, 2023）：**
- 用 DPO + RLAIF 数据
- 性能接近 GPT-4

### 7.2 最佳实践

1. **高质量 SFT 模型**：
   - DPO 的起点很重要
   - 先确保 SFT 模型质量

2. **数据质量 > 数量**：
   - 10K 高质量偏好对 > 100K 低质量
   - 标注一致性很重要

3. **多轮迭代**：
   - DPO → 收集新数据 → 再 DPO
   - 类似 RLHF 的迭代训练

4. **混合训练**：
   - DPO + SFT 混合损失
   - 防止遗忘预训练知识


## 8. 总结

### 8.1 核心要点

1. **DPO 的本质**：
   - 重新参数化 RLHF 目标
   - 隐式奖励 = 策略比率
   - 直接从偏好数据优化策略

2. **DPO 的优势**：
   - 简单：无需奖励模型和 PPO
   - 稳定：监督学习
   - 高效：内存和计算开销小

3. **DPO 的局限**：
   - 依赖 SFT 质量
   - 无法在线学习
   - 对 β 敏感

4. **实践建议**：
   - 从 β=0.1 开始
   - 监控 KL 散度
   - 多轮迭代训练

### 8.2 下节预告

下节课我们将学习 **DPO 的变体**：
- IPO（Identity Preference Optimization）
- KTO（Kahneman-Tversky Optimization）
- ORPO（Odds Ratio Preference Optimization）
- SimPO（Simple Preference Optimization）

### 8.3 扩展阅读

**必读论文：**
1. Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
2. Tunstall et al. (2023). "Zephyr: Direct Distillation of LM Alignment"
3. Ivison et al. (2023). "Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2"

**代码资源：**
- TRL DPO Trainer: https://huggingface.co/docs/trl/dpo_trainer
- DPO 原始实现: https://github.com/eric-mitchell/direct-preference-optimization
