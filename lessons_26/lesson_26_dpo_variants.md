# Lesson 26: DPO 变体与发展

## 1. DPO 的不足与改进动机

### 1.1 DPO 的已知问题

**1. 过拟合偏好数据**
- DPO 是监督学习，容易过拟合
- 特别是数据量小时

**2. 长度偏差（Length Exploitation）**
- 模型可能学到"长回答更好"
- 因为 log 概率与长度相关

**3. β 敏感性**
- 性能对 β 选择非常敏感
- 不同任务需要不同的 β

**4. 无参考模型的需求**
- 参考模型占用大量内存
- 能否去掉参考模型？

### 1.2 改进方向

```
DPO (2023)
  ├── IPO (2023)  — 解决过拟合
  ├── KTO (2024)  — 无需偏好对，只需二元反馈
  ├── ORPO (2024) — 去掉参考模型
  └── SimPO (2024) — 长度归一化 + 去掉参考模型
```


## 2. IPO (Identity Preference Optimization)

### 2.1 动机

**DPO 的问题：** 当 $\beta \to 0$ 时，DPO 退化为确定性地选择 chosen，完全忽视偏好的不确定性。

**IPO 的解决：** 用更鲁棒的损失函数。

### 2.2 IPO 损失

$$\mathcal{L}_{\text{IPO}} = \mathbb{E}\left[\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \frac{1}{2\beta}\right)^2\right]$$

**与 DPO 对比：**
- DPO 用 sigmoid + log → 无界损失
- IPO 用平方损失 → 有界，更稳定
- IPO 有一个"目标间距" $1/(2\beta)$

### 2.3 直觉

- IPO 不追求无限大的奖励差
- 而是让 chosen 比 rejected 好"恰好 $1/(2\beta)$"
- 避免过度自信


## 3. KTO (Kahneman-Tversky Optimization)

### 3.1 动机

**偏好对数据的问题：**
- 需要成对比较数据 $(x, y_w, y_l)$
- 标注成本高
- 现实中更容易获得"好/坏"的二元反馈

**KTO 的创新：** 只需要二元反馈（好/坏），不需要配对！

### 3.2 前景理论启发

**Kahneman & Tversky 前景理论：**
- 人对损失比对收益更敏感
- 损失厌恶系数约 2.0

**KTO 据此设计不对称损失：**

$$\mathcal{L}_{\text{KTO}} = \mathbb{E}_{y \sim \text{desirable}}\left[1 - \sigma(\beta r(x,y) - z_0)\right] + \lambda \cdot \mathbb{E}_{y \sim \text{undesirable}}\left[1 - \sigma(z_0 - \beta r(x,y))\right]$$

其中：
- $r(x,y) = \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$（隐式奖励）
- $z_0 = \mathbb{E}[r(x,y)]$（基线）
- $\lambda$：损失厌恶系数（通常 > 1）

### 3.3 KTO 的优势

1. **数据需求低**：
   - 只需要 $(x, y, \text{good/bad})$
   - 不需要配对比较
   - 可以利用点赞/踩的数据

2. **心理学基础**：
   - 基于前景理论
   - 对"坏"回答更严格惩罚

3. **实际性能**：
   - 在多个基准上接近或超过 DPO
   - 数据量少时优势更明显


## 4. ORPO (Odds Ratio Preference Optimization)

### 4.1 动机

**去掉参考模型：** DPO 需要参考模型计算 $\log \pi_{\text{ref}}$，占用额外内存。

**ORPO 的方案：** 将 SFT 和偏好优化统一到一个损失函数中。

### 4.2 ORPO 损失

$$\mathcal{L}_{\text{ORPO}} = \underbrace{\mathcal{L}_{\text{SFT}}(y_w|x)}_{\text{SFT 项}} + \lambda \cdot \underbrace{\mathcal{L}_{\text{OR}}}_{\text{偏好项}}$$

**SFT 项：** 标准的负对数似然（只对 chosen）

$$\mathcal{L}_{\text{SFT}} = -\log P_\theta(y_w|x)$$

**偏好项（Odds Ratio）：**

$$\mathcal{L}_{\text{OR}} = -\log \sigma\left(\log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\right)$$

其中 odds ratio 定义为：

$$\text{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1 - P_\theta(y|x)}$$

### 4.3 ORPO 的优势

1. **无需参考模型**：节省一半 GPU 内存
2. **统一训练**：SFT + 偏好优化合并
3. **一阶段训练**：不需要先 SFT 再 DPO


## 5. SimPO (Simple Preference Optimization)

### 5.1 动机

**DPO 的两个问题：**
1. 需要参考模型
2. 对序列长度敏感（长序列的 log 概率自然更小）

### 5.2 SimPO 损失

$$\mathcal{L}_{\text{SimPO}} = -\log \sigma\left(\frac{\beta}{|y_w|}\log P_\theta(y_w|x) - \frac{\beta}{|y_l|}\log P_\theta(y_l|x) - \gamma\right)$$

**三个关键设计：**

1. **长度归一化**：除以 $|y|$（response 长度）
   - 消除长度偏差
   - 公平比较不同长度的回答

2. **无需参考模型**：直接用策略的平均 log 概率
   - 节省内存
   - 简化训练

3. **目标间距 $\gamma$**：
   - 要求 chosen 比 rejected 好至少 $\gamma$
   - 防止过拟合
   - 类似 margin loss

### 5.3 SimPO vs DPO

| 方面 | DPO | SimPO |
|------|-----|-------|
| 参考模型 | 需要 | 不需要 |
| 长度归一化 | 无 | 有 |
| 目标间距 | 无 | 有 ($\gamma$) |
| 内存占用 | 2x 模型 | 1x 模型 |
| 性能 | 好 | 更好（多个基准上） |

### 5.4 SimPO 实现

```python
def simpo_loss(policy_chosen_logps, policy_rejected_logps,
               chosen_length, rejected_length,
               beta=2.0, gamma=0.5):
    """
    SimPO 损失函数
    
    参数:
        policy_chosen_logps: chosen 的总 log 概率
        policy_rejected_logps: rejected 的总 log 概率
        chosen_length: chosen 的长度
        rejected_length: rejected 的长度
        beta: 温度参数
        gamma: 目标间距
    """
    # 长度归一化的平均 log 概率
    chosen_avg = policy_chosen_logps / chosen_length
    rejected_avg = policy_rejected_logps / rejected_length
    
    # SimPO 损失
    logits = beta * (chosen_avg - rejected_avg) - gamma
    loss = -F.logsigmoid(logits).mean()
    
    return loss
```


## 6. 方法全面对比

### 6.1 总结表

| 方法 | 参考模型 | 数据格式 | 长度归一化 | 目标间距 | 核心创新 |
|------|----------|----------|------------|----------|----------|
| **DPO** | 需要 | 偏好对 | 无 | 无 | 隐式奖励 |
| **IPO** | 需要 | 偏好对 | 无 | 有 | 平方损失 |
| **KTO** | 需要 | 二元反馈 | 无 | 有 | 前景理论 |
| **ORPO** | 不需要 | 偏好对 | 无 | 无 | Odds Ratio + SFT |
| **SimPO** | 不需要 | 偏好对 | 有 | 有 | 长度归一化 |

### 6.2 如何选择？

```
有偏好对数据？
├── 是 → GPU 内存充足？
│   ├── 是 → DPO（经典可靠）或 IPO（更鲁棒）
│   └── 否 → SimPO（无需参考模型）或 ORPO（统一训练）
└── 否 → 只有好/坏标签？
    ├── 是 → KTO
    └── 否 → 先收集数据
```

### 6.3 性能基准

**在 AlpacaEval 2 上的典型表现（7B 模型）：**

```
SimPO > DPO ≈ IPO > ORPO > KTO > SFT
```

**注意：** 实际排名取决于数据质量、模型大小、任务类型。


## 7. 实践建议

### 7.1 推荐起步方案

1. **首选 DPO**：最成熟，社区支持最好
2. **内存不够 → SimPO**：去掉参考模型
3. **没有配对数据 → KTO**：只需好/坏标签
4. **想简化流程 → ORPO**：一阶段训练

### 7.2 通用调参建议

- **β**：从 0.1 开始（DPO/IPO），从 2.0 开始（SimPO）
- **γ（SimPO）**：通常 0.5 ~ 1.5
- **学习率**：1e-6 ~ 5e-6
- **训练轮数**：1-3 epochs（偏好数据容易过拟合）


## 8. 总结

### 8.1 核心要点

1. **DPO 开创了无 RL 对齐的方向**
2. **IPO** 通过平方损失提高鲁棒性
3. **KTO** 降低了数据门槛（无需配对）
4. **ORPO** 去掉了参考模型，统一 SFT + 偏好
5. **SimPO** 长度归一化 + 无参考模型，当前最佳之一

### 8.2 下节预告

下节课我们将学习 **GRPO（Group Relative Policy Optimization）**：
- DeepSeek 的 GRPO 方法
- 无需价值模型的策略优化
- 组内相对排序
- GRPO 的实现

### 8.3 扩展阅读

**必读论文：**
1. Azar et al. (2023). "A General Theoretical Paradigm to Understand Learning from Human Feedback" (IPO)
2. Ethayarajh et al. (2024). "KTO: Model Alignment as Prospect Theoretic Optimization"
3. Hong et al. (2024). "ORPO: Monolithic Preference Optimization without Reference Model"
4. Meng et al. (2024). "SimPO: Simple Preference Optimization with a Reference-Free Reward"
