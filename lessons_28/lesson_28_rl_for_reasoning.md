# Lesson 28: RL 用于推理 (RL for Reasoning)

## 1. 为什么推理需要 RL？

### 1.1 LLM 推理的现状

**LLM 的推理能力：**
- 数学推理（GSM8K, MATH）
- 代码生成（HumanEval, SWE-bench）
- 逻辑推理（ARC, BBH）
- 常识推理（HellaSwag, WinoGrande）

**问题：**
- 一次生成（System 1）：快但容易出错
- 缺乏"深度思考"（System 2）
- 无法自我验证和修正
- 复杂问题准确率低

### 1.2 从 System 1 到 System 2

**Daniel Kahneman 的双系统理论：**

| | System 1 | System 2 |
|--|----------|----------|
| 特点 | 快速、直觉 | 慢速、深思 |
| 类比 | 标准 LLM 推理 | 思维链 + 搜索 |
| 准确性 | 中等 | 高 |
| 计算量 | 低 | 高 |

**RL 的作用：** 训练 LLM 学会 System 2 推理

- 学会何时"慢下来思考"
- 学会分步推理
- 学会自我验证


## 2. OpenAI o1/o3 的思路

### 2.1 核心理念

**Test-time Compute Scaling：**

$$\text{性能} = f(\text{模型大小}, \textbf{推理时计算量})$$

传统 Scaling Law 只增加模型大小；o1 同时增加**推理时的计算量**。

**关键技术：**
1. 长思维链（Chain-of-Thought）
2. 搜索与回溯
3. 自我验证
4. RL 训练推理策略

### 2.2 推测的技术方案

```
训练阶段:
  1. SFT: 在高质量推理数据上微调
  2. RL: 用 RL 优化推理过程
     - 奖励: 最终答案正确性
     - 过程奖励: 中间步骤的质量
     - 搜索: 探索不同推理路径

推理阶段:
  1. 生成多个推理路径
  2. 每条路径包含多个推理步骤
  3. 用搜索策略（如 MCTS）选择最优路径
  4. 输出最终答案
```

### 2.3 关键观察

**更多推理时间 = 更高准确率：**

```
GSM8K (数学):
  标准推理:     ~85%
  o1 (更多思考): ~95%

MATH (竞赛数学):
  标准推理:     ~50%
  o1 (更多思考): ~83%
```


## 3. 结果奖励 vs 过程奖励

### 3.1 结果奖励模型 (ORM)

**Outcome Reward Model：只看最终结果**

$$R_{\text{ORM}}(x, y) = \begin{cases} 1 & \text{最终答案正确} \\ 0 & \text{最终答案错误} \end{cases}$$

**问题：**
- 稀疏奖励：只在最后一步给信号
- 信用分配困难：不知道哪步出错
- 无法引导中间推理过程

### 3.2 过程奖励模型 (PRM)

**Process Reward Model：评估每一步**

$$R_{\text{PRM}}(x, y_1, y_2, ..., y_T) = \prod_{t=1}^{T} P(\text{step } t \text{ correct} | x, y_1, ..., y_t)$$

**优势：**
- 密集奖励：每步都有反馈
- 精确定位：知道哪步出错
- 更好的引导：鼓励正确的中间步骤

### 3.3 ORM vs PRM 对比

| 方面 | ORM | PRM |
|------|-----|-----|
| 奖励粒度 | 序列级 | 步骤级 |
| 标注成本 | 低（只标注最终答案） | 高（标注每步） |
| 信用分配 | 差 | 好 |
| 训练信号 | 稀疏 | 密集 |
| 性能 | 基线 | 显著提升 |

### 3.4 PRM 训练

**数据格式：**

```json
{
    "problem": "求解 2x + 3 = 7",
    "steps": [
        {"text": "两边减 3: 2x = 4", "label": "positive"},
        {"text": "两边除以 2: x = 2", "label": "positive"}
    ]
}
```

**训练目标：**

$$\mathcal{L}_{\text{PRM}} = -\sum_{t=1}^T \left[ c_t \log P(+|s_t) + (1-c_t) \log P(-|s_t) \right]$$

其中 $c_t \in \{0, 1\}$ 是第 $t$ 步的正确性标签。

### 3.5 PRM 数据收集

**方法 1：人工标注**
- OpenAI 的 PRM800K 数据集
- 标注员逐步判断每步是否正确
- 成本高但质量好

**方法 2：自动标注（Math-Shepherd 方法）**
1. 在每步之后，尝试多次完成剩余推理
2. 如果多次完成都能得到正确答案 → 该步正确
3. 如果都得不到正确答案 → 该步错误

```
问题: 2x + 3 = 7

Step 1: "2x = 4"
  → 从这步继续，10 次中 9 次得到 x=2（正确）
  → 标签: positive (0.9)

Step 2 (错误版): "x = 3"
  → 从这步继续，10 次中 0 次得到正确答案
  → 标签: negative (0.0)
```


## 4. MCTS + LLM

### 4.1 蒙特卡洛树搜索回顾

**MCTS 的四个阶段：**

```
1. 选择 (Selection):
   从根节点出发，用 UCB 选择子节点
   UCB = Q(s,a) + c * sqrt(ln N(s) / N(s,a))

2. 扩展 (Expansion):
   到达叶节点，添加新的子节点

3. 模拟 (Simulation):
   从新节点开始，随机模拟到终局

4. 回传 (Backpropagation):
   用模拟结果更新路径上所有节点的统计
```

### 4.2 LLM 中的 MCTS

**将推理建模为树搜索：**

```
根节点: 问题
  ├── 推理步骤 1a
  │   ├── 推理步骤 2a → 答案 A (正确)
  │   └── 推理步骤 2b → 答案 B (错误)
  └── 推理步骤 1b
      ├── 推理步骤 2c → 答案 C (错误)
      └── 推理步骤 2d → 答案 A (正确)
```

**LLM 的作用：**
- **策略网络**：生成候选推理步骤
- **价值网络/PRM**：评估当前状态的价值

### 4.3 MCTS + LLM 实现

```python
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state          # 当前推理状态
        self.parent = parent
        self.children = {}          # {action: MCTSNode}
        self.visit_count = 0
        self.value_sum = 0
    
    def ucb_score(self, c=1.414):
        if self.visit_count == 0:
            return float('inf')
        exploitation = self.value_sum / self.visit_count
        exploration = c * sqrt(log(self.parent.visit_count) / self.visit_count)
        return exploitation + exploration

def mcts_search(problem, llm, prm, num_simulations=100):
    root = MCTSNode(state=problem)
    
    for _ in range(num_simulations):
        # 1. Selection
        node = select(root)
        
        # 2. Expansion: 用 LLM 生成下一步
        next_steps = llm.generate_next_steps(node.state, K=3)
        for step in next_steps:
            child = MCTSNode(node.state + step, parent=node)
            node.children[step] = child
        
        # 3. Simulation: 用 PRM 评估
        leaf = random.choice(list(node.children.values()))
        value = prm.evaluate(leaf.state)
        
        # 4. Backpropagation
        backpropagate(leaf, value)
    
    # 返回最佳路径
    return best_path(root)
```


## 5. STaR 与 Quiet-STaR

### 5.1 STaR (Self-Taught Reasoner)

**Zelikman et al. (2022)：** 让模型从自己的推理中学习。

**核心思想：**

```
循环:
  1. 对每个问题，让模型生成推理 + 答案
  2. 如果答案正确 → 保留推理作为训练数据
  3. 如果答案错误 → 提供正确答案，让模型"合理化"推理
  4. 在收集的数据上微调模型
  重复
```

**"合理化"（Rationalization）：**
- 给定正确答案，让模型生成导致该答案的推理
- 类似于事后合理化
- 扩充正确推理数据

### 5.2 Quiet-STaR

**Zelikman et al. (2024)：** 在每个 token 处"安静地思考"。

**核心思想：**

```
对于每个 token 位置:
  1. 生成内部"思考" (thought)
  2. 用思考来改善下一个 token 的预测
  3. 训练：让有思考的预测优于没思考的预测
```

**训练目标：**

$$\mathcal{L} = -\log P_{\text{with\_thought}}(x_{t+1} | x_{\leq t}, \text{thought}_t)$$

**直觉：** 模型学会在需要时"停下来想一想"。


## 6. 实践：过程奖励训练

### 6.1 简化设置

**任务：** 多步数学推理（简化版）

```
问题: 计算 (a + b) * c - d
步骤 1: 计算 a + b = e
步骤 2: 计算 e * c = f
步骤 3: 计算 f - d = 答案
```

**PRM 训练：**
- 对每步判断是否正确
- 正确步骤：正奖励
- 错误步骤：负奖励

### 6.2 使用 PRM 引导搜索

**Best-of-N 采样：**

```python
def best_of_n(problem, model, prm, N=8):
    """
    生成 N 个推理路径，用 PRM 选最好的
    """
    candidates = []
    
    for _ in range(N):
        reasoning = model.generate(problem)
        # PRM 评分：所有步骤得分的乘积
        score = prm.score_all_steps(problem, reasoning)
        candidates.append((reasoning, score))
    
    # 选择 PRM 分数最高的
    best = max(candidates, key=lambda x: x[1])
    return best[0]
```

### 6.3 用 RL 训练推理策略

```python
# 奖励 = PRM 评分（密集）+ 最终正确性（稀疏）
reward = lambda_prm * prm_score + lambda_orm * correctness

# 用 GRPO 优化
# 组内比较不同推理路径
```


## 7. 前沿进展

### 7.1 Scaling Test-time Compute

**关键论文：**
- Snell et al. (2024): "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters"

**核心发现：**
- 推理时增加计算量的收益可以超过增加模型大小
- 最优策略取决于问题难度
- 简单问题：少量搜索即可
- 困难问题：需要大量搜索

### 7.2 DeepSeek-R1 的成功

**训练流程：**
1. 冷启动 SFT
2. GRPO 强化学习（奖励 = 正确性）
3. 拒绝采样 + 再 SFT

**关键发现：**
- RL 训练自然涌现了思维链
- 不需要显式的 CoT 数据
- 自我验证和反思行为自发出现

### 7.3 未来方向

1. **更好的 PRM**：自动标注、少样本学习
2. **更高效的搜索**：学习何时搜索、搜索多深
3. **多模态推理**：结合视觉和语言推理
4. **推理蒸馏**：将 o1 级别的推理能力蒸馏到小模型


## 8. 总结

### 8.1 核心要点

1. **推理需要 RL**：
   - System 2 思维需要搜索和验证
   - RL 训练模型学会深度推理

2. **PRM vs ORM**：
   - PRM 提供密集的步骤级奖励
   - 显著优于只看结果的 ORM
   - 标注成本可通过自动方法降低

3. **MCTS + LLM**：
   - 树搜索探索推理空间
   - LLM 作为策略和价值网络

4. **STaR 系列**：
   - 自我改进推理能力
   - 无需外部标注

5. **Test-time Compute**：
   - 推理时增加计算量的新范式
   - 与模型大小互补

### 8.2 下节预告

下节课我们将学习 **在线 RL 与 LLM Agent**：
- LLM 作为策略网络
- 工具调用 RL
- WebAgent、SWE-agent

### 8.3 扩展阅读

**必读论文：**
1. Lightman et al. (2023). "Let's Verify Step by Step" (PRM)
2. Zelikman et al. (2022). "STaR: Bootstrapping Reasoning With Reasoning"
3. Wang et al. (2024). "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations"
4. DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
5. Snell et al. (2024). "Scaling LLM Test-Time Compute"
