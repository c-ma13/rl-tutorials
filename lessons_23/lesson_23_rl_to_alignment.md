# Lesson 23: 从 RL 到 LLM 对齐

## 1. 为什么需要对齐？

### 1.1 LLM 的能力与风险

**大语言模型（LLM）的能力：**
- 语言理解与生成
- 代码编写与调试
- 逻辑推理与分析
- 创意写作与翻译
- 多轮对话与问答

**但预训练 LLM 存在问题：**

1. **有害内容**：可能生成有毒、歧视、暴力的文本
2. **虚假信息**：会"一本正经地胡说八道"（幻觉）
3. **不遵循指令**：可能忽略用户的真实意图
4. **安全风险**：可能泄露训练数据中的隐私信息
5. **偏见**：反映训练数据中的社会偏见

### 1.2 对齐的目标

**对齐（Alignment）= 让 AI 系统的行为符合人类的意图和价值观**

**InstructGPT（Ouyang et al., 2022）提出的三个核心目标：**

1. **有帮助（Helpful）**：
   - 准确回答用户问题
   - 遵循用户指令
   - 提供有用的信息

2. **无害（Harmless）**：
   - 不生成有害内容
   - 拒绝危险请求
   - 不歧视任何群体

3. **诚实（Honest）**：
   - 不编造信息
   - 承认不确定性
   - 承认自身局限

### 1.3 对齐的挑战

**为什么对齐很难？**

1. **价值观难以形式化**：
   - "有帮助"的精确定义是什么？
   - 不同人的价值观可能冲突

2. **奖励信号稀疏**：
   - 监督学习只给出"正确答案"
   - 无法覆盖所有可能的回答

3. **分布偏移**：
   - 训练数据 ≠ 实际使用场景
   - 用户可能提出训练中未见过的问题

4. **可扩展性**：
   - 人工标注成本高
   - 需要自动化的对齐方法


## 2. LLM 训练三阶段

### 2.1 总览

```
阶段 1: 预训练 (Pretraining)
  ↓  大量无标注文本，自回归语言建模
阶段 2: 监督微调 (SFT - Supervised Fine-Tuning)
  ↓  高质量指令-回答对，监督学习
阶段 3: 人类反馈强化学习 (RLHF)
  ↓  奖励模型 + PPO 优化
最终: 对齐的 LLM
```

### 2.2 阶段 1: 预训练（Pretraining）

**目标：** 学习语言的通用表示

**方法：** 自回归语言建模

$$\mathcal{L}_{\text{pretrain}} = -\sum_{t=1}^{T} \log P_\theta(x_t | x_{<t})$$

**数据：**
- 互联网文本（网页、书籍、代码）
- 数万亿 token
- 无需人工标注

**结果：**
- 学会了语法、事实、推理模式
- 但不知道如何遵循指令
- 可能生成有害内容
- "补全文本"而非"回答问题"

**例子：**

```
用户输入: "法国的首都是什么？"
预训练 LLM: "这是一个常见的地理问题。类似的问题还有..."  ← 补全而非回答
对齐的 LLM: "法国的首都是巴黎。"  ← 直接回答
```

### 2.3 阶段 2: 监督微调（SFT）

**目标：** 让模型学会遵循指令

**方法：** 在高质量指令-回答数据上微调

$$\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T} \log P_\theta(y_t | x, y_{<t})$$

其中 $x$ 是指令，$y$ 是期望的回答。

**数据：**
- 人工编写的 (指令, 回答) 对
- 通常 10K ~ 100K 条
- 质量 >> 数量

**数据示例：**

```json
{
    "instruction": "用一句话总结光合作用",
    "response": "光合作用是植物利用阳光将二氧化碳和水转化为葡萄糖和氧气的过程。"
}
```

**SFT 的局限：**
- 只能从"正确回答"中学习
- 一个问题可能有多种好的回答方式
- 无法表达回答的好坏程度
- 覆盖面有限

### 2.4 阶段 3: RLHF

**目标：** 进一步优化，超越监督学习的天花板

**方法：** 
1. 训练奖励模型（从人类偏好中学习）
2. 用 PPO 优化策略（最大化奖励）

**为什么需要 RLHF？**

| 方面 | SFT | RLHF |
|------|-----|------|
| 信号类型 | 二元（对/错） | 连续（好/差的程度） |
| 学习方式 | 模仿 | 优化 |
| 天花板 | 数据质量 | 可超越数据 |
| 多样性 | 照搬训练数据 | 探索更好的回答 |
| 偏好 | 无法表达 | 直接建模 |


## 3. 奖励模型（Reward Model）

### 3.1 核心思想

**奖励模型 = 将人类偏好转化为数值信号**

$$R_\phi: (\text{prompt}, \text{response}) \to \mathbb{R}$$

- 输入：(提示, 回答) 对
- 输出：标量分数（越高越好）

### 3.2 数据收集

**比较数据（Comparison Data）：**

```
给定提示 x，SFT 模型生成多个回答：y_1, y_2, ...
人类标注员比较并排序：y_w ≻ y_l（w 比 l 好）
```

**为什么用比较而非打分？**
- 比较更容易、更一致
- 绝对打分主观差异大
- 排序比评分更可靠

**数据格式：**

```json
{
    "prompt": "如何学好编程？",
    "chosen": "1. 选一门语言入门（推荐 Python）...",
    "rejected": "编程很简单，看几个视频就行了。"
}
```

### 3.3 Bradley-Terry 模型

**偏好建模：**

假设人类偏好遵循 Bradley-Terry 模型：

$$P(y_w \succ y_l | x) = \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))$$

其中 $\sigma$ 是 sigmoid 函数。

**训练目标：**

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(R_\phi(x, y_w) - R_\phi(x, y_l)) \right]$$

**直觉：**
- 让好回答的奖励 > 差回答的奖励
- 差值越大，越有信心
- 类似于对比学习

### 3.4 奖励模型架构

**基于 LLM 的奖励模型：**

```
输入: [prompt + response]
  ↓
预训练 LLM（如 GPT）
  ↓
取最后一个 token 的隐藏状态
  ↓
线性层 → 标量分数 R(x, y)
```

**实现：**

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model  # 预训练 LLM
        self.head = nn.Linear(hidden_size, 1)  # 奖励头
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask)
        # 取最后一个 token 的隐藏状态
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.head(last_hidden)
        return reward.squeeze(-1)
```

### 3.5 奖励模型的质量

**评估指标：**
- 偏好准确率：模型在测试集上的排序正确率
- 通常需要 > 70% 才有用

**影响因素：**
- 标注数据质量和数量
- 基础模型大小
- 标注者一致性
- 提示多样性


## 4. RLHF 训练流程

### 4.1 将 LLM 视为 RL 问题

**RL 视角下的 LLM：**

| RL 概念 | LLM 中的对应 |
|---------|-------------|
| **环境** | 用户的提示（prompt） |
| **状态** | 已生成的 token 序列 |
| **动作** | 下一个要生成的 token |
| **策略** | LLM 本身 $\pi_\theta(a_t \mid s_t)$ |
| **奖励** | 奖励模型的评分 $R_\phi(x, y)$ |
| **轨迹** | 完整的回答 |

### 4.2 RLHF 的目标函数

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ R_\phi(x, y) - \beta \cdot D_{\text{KL}}(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)) \right]$$

**三个部分：**

1. **奖励最大化**：$R_\phi(x, y)$
   - 让回答质量尽可能高

2. **KL 散度约束**：$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$
   - 不要偏离 SFT 模型太远
   - 防止"过度优化"奖励模型

3. **权衡系数**：$\beta$
   - 控制探索与保守的平衡
   - 太小：可能"hack"奖励模型
   - 太大：和 SFT 没区别

### 4.3 KL 散度的重要性

**为什么需要 KL 约束？**

**问题：奖励模型不完美**

- 奖励模型是人类偏好的近似
- 存在 OOD（分布外）区域
- 策略可能找到"骗过"奖励模型的方式

**奖励过度优化（Reward Hacking）：**

```
优化过程中：
  - 奖励分数不断提高 ✓
  - 但实际质量下降 ✗
  - 模型学会了"讨好"奖励模型
```

**例子：**
- 模型学会使用冗长但空洞的回答（因为长回答得分高）
- 模型重复某些"讨好"的短语
- 回答变得不自然

**KL 约束的作用：**
- 限制策略不要偏离太远
- 保持语言质量和流畅性
- 防止模式崩溃

### 4.4 PPO 在 RLHF 中的应用

**算法流程：**

```
初始化：
  - 策略模型 π_θ（从 SFT 模型初始化）
  - 参考模型 π_ref（SFT 模型，冻结）
  - 奖励模型 R_φ（冻结）
  - 价值模型 V_ψ（评估状态价值）

对于每个 batch：
  1. 采样提示：x ~ D_prompts
  2. 生成回答：y ~ π_θ(·|x)
  3. 计算奖励：r = R_φ(x, y) - β · KL(π_θ(y|x) || π_ref(y|x))
  4. 计算优势：A = GAE(r, V_ψ)
  5. PPO 更新：
     - 更新策略 π_θ（PPO-Clip）
     - 更新价值 V_ψ
```

### 4.5 实际工程挑战

**1. 计算开销**

RLHF 训练需要同时加载：
- 策略模型（可训练）
- 参考模型（冻结）
- 奖励模型（冻结）
- 价值模型（可训练）

→ 4 个大模型同时在 GPU 上

**2. 训练稳定性**

- PPO 对超参数敏感
- 需要仔细调节 KL 系数 $\beta$
- 价值函数估计可能不准确

**3. 评估困难**

- 自动评估不可靠
- 需要人工评估
- 人工评估成本高且慢


## 5. 经典案例

### 5.1 InstructGPT（OpenAI, 2022）

**里程碑：** 第一个系统化的 RLHF 实践

**流程：**
1. SFT：在 13K 条标注数据上微调 GPT-3
2. RM：在 33K 条比较数据上训练奖励模型
3. RLHF：用 PPO 优化 SFT 模型

**结果：**
- 1.3B InstructGPT > 175B GPT-3
- 人类标注员偏好 InstructGPT 85% 的时间
- 有害内容生成减少 25%

### 5.2 ChatGPT（OpenAI, 2022）

**基于 InstructGPT 的方法，针对对话场景优化：**
- 多轮对话数据
- 更大规模的标注
- 更精细的安全策略

### 5.3 Claude（Anthropic）

**Constitutional AI（CAI）：**
- 减少对人工标注的依赖
- 用 AI 自我批评（AI feedback）
- 定义明确的"宪法"原则

### 5.4 Llama 2（Meta, 2023）

**开源 RLHF 实践：**
- 详细公开了 RLHF 流程
- 拒绝采样（Rejection Sampling）
- 迭代式 RLHF


## 6. 奖励模型的数据格式

### 6.1 标准格式

**偏好对（Preference Pairs）：**

```json
{
    "prompt": "请解释什么是量子计算",
    "chosen": "量子计算是一种利用量子力学原理进行计算的新型计算范式...",
    "rejected": "量子计算就是用量子来计算。"
}
```

### 6.2 排序格式

**多个回答的排序：**

```json
{
    "prompt": "如何保持健康？",
    "responses": [
        {"text": "均衡饮食、规律运动、充足睡眠是健康三要素...", "rank": 1},
        {"text": "多运动对身体好。", "rank": 2},
        {"text": "吃点保健品就行了。", "rank": 3}
    ]
}
```

**从排序生成比较对：**

$K$ 个排序可生成 $\binom{K}{2}$ 个比较对。

### 6.3 常用数据集

| 数据集 | 规模 | 来源 | 特点 |
|--------|------|------|------|
| **Anthropic HH-RLHF** | 170K | Anthropic | 有帮助 + 无害 |
| **OpenAssistant** | 160K | 社区 | 多语言、开源 |
| **Stanford SHP** | 385K | Reddit | 自然偏好 |
| **UltraFeedback** | 64K | 多模型 | GPT-4 标注 |


## 7. 超越 RLHF

### 7.1 RLHF 的局限

1. **成本高**：需要大量人工标注
2. **复杂**：需要训练多个模型
3. **不稳定**：PPO 训练困难
4. **标注偏差**：标注者可能不代表所有用户

### 7.2 替代方法预览

| 方法 | 核心思想 | 优势 |
|------|----------|------|
| **DPO** | 直接从偏好学习策略 | 无需奖励模型 |
| **GRPO** | 组相对策略优化 | 无需价值模型 |
| **RLAIF** | 用 AI 代替人类反馈 | 降低标注成本 |
| **CAI** | 宪法 AI | 原则驱动 |
| **KTO** | 从二元信号学习 | 数据要求低 |

**后续课程会深入讲解 DPO 和 GRPO。**


## 8. 总结

### 8.1 核心要点

1. **对齐的必要性**：
   - 预训练 LLM 能力强但不可控
   - 对齐 = 有帮助 + 无害 + 诚实

2. **三阶段训练**：
   - 预训练：学习语言能力
   - SFT：学习遵循指令
   - RLHF：学习人类偏好

3. **奖励模型**：
   - 从人类比较数据学习
   - Bradley-Terry 模型
   - 质量决定 RLHF 上限

4. **RLHF 训练**：
   - PPO + KL 约束
   - 防止奖励过度优化
   - 工程挑战大

### 8.2 下节预告

下节课我们将学习 **DPO（Direct Preference Optimization）**：
- DPO 的数学推导
- 为什么 DPO 可以省去奖励模型
- DPO 的实现与训练
- DPO vs RLHF 对比

### 8.3 扩展阅读

**必读论文：**
1. Ouyang et al. (2022). "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT)
2. Christiano et al. (2017). "Deep Reinforcement Learning from Human Preferences"
3. Bai et al. (2022). "Training a Helpful and Harmless Assistant with RLHF" (Anthropic)
4. Touvron et al. (2023). "Llama 2: Open Foundation and Fine-Tuned Chat Models"

**推荐资源：**
- Hugging Face TRL 库文档
- OpenAI 对齐研究博客
- Anthropic 研究论文
