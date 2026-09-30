# Lesson 21: 离线 RL (Offline RL)

## 1. 为什么需要离线 RL？

### 1.1 在线 RL 的局限性

到目前为止，我们学习的所有算法都是**在线 RL（Online RL）**：

- 智能体与环境**实时交互**
- 边探索边学习
- 可以尝试任意动作

**在线 RL 的问题：**

1. **交互成本高**：
   - 自动驾驶：真车上路测试危险且昂贵
   - 医疗：不能随意尝试治疗方案
   - 推荐系统：每次推荐都影响用户体验

2. **探索风险大**：
   - 机器人可能损坏设备
   - 金融交易可能造成巨额损失
   - 工业控制可能导致事故

3. **效率低**：
   - 需要大量交互才能学到好策略
   - AlphaGo 训练需要数百万局自博弈

### 1.2 离线 RL 的动机

**核心思想：** 从**已有的数据集**中学习策略，无需与环境交互。

$$\text{在线 RL: } \quad \pi \xrightarrow{\text{交互}} \mathcal{D} \xrightarrow{\text{学习}} \pi'$$

$$\text{离线 RL: } \quad \mathcal{D}_{\text{fixed}} \xrightarrow{\text{学习}} \pi$$

**优势：**
- 利用历史数据（日志、专家演示、次优策略）
- 无需在线交互 → 安全、低成本
- 可以从多个来源收集数据

**应用场景：**
- **医疗**：从电子病历学习治疗策略
- **推荐系统**：从用户历史行为学习
- **自动驾驶**：从人类驾驶数据学习
- **机器人**：从遥操作演示学习


## 2. 离线 RL 的形式化

### 2.1 问题设定

**给定：**
- 固定数据集 $\mathcal{D} = \{(s, a, r, s')\}$
- 数据由**行为策略** $\pi_\beta$ 收集

**目标：**
- 学习策略 $\pi$ 使得 $J(\pi) = \mathbb{E}_\pi[\sum_t \gamma^t r_t]$ 最大

**关键约束：**
- 不能与环境交互
- 只能从 $\mathcal{D}$ 中学习

### 2.2 与其他范式的关系

| 范式 | 数据来源 | 能否交互 | 目标 |
|------|----------|----------|------|
| **监督学习** | 固定数据集 | 否 | 模仿行为策略 |
| **模仿学习** | 专家演示 | 否 | 模仿专家 |
| **在线 RL** | 自己收集 | 是 | 最大化回报 |
| **离线 RL** | 固定数据集 | 否 | **超越行为策略** |

**离线 RL 的独特之处：**
- 数据可能来自次优策略
- 目标是学到**比行为策略更好**的策略
- 不能简单地模仿数据


## 3. 离线 RL 的核心挑战

### 3.1 分布偏移（Distribution Shift）

**问题：** 学习的策略 $\pi$ 可能访问数据集中**没有**的状态-动作对。

$$\pi(a|s) \neq \pi_\beta(a|s) \Rightarrow d^\pi(s, a) \neq d^{\pi_\beta}(s, a)$$

**后果：**
- Q 函数在 OOD（Out-of-Distribution）区域不准确
- 策略可能选择看起来好但实际很差的动作

**例子：**
- 数据集：专家总是在红灯停车
- 学习策略：发现"闯红灯"在数据中没有负奖励
- 实际执行：灾难性后果

### 3.2 外推误差（Extrapolation Error）

**Q-learning 的更新：**

$$Q(s, a) \leftarrow r + \gamma \max_{a'} Q(s', a')$$

**问题：** $\max_{a'} Q(s', a')$ 可能选择数据集中**没有**的动作。

- 如果 $a'$ 在数据集中很少出现，$Q(s', a')$ 不可靠
- 但 $\max$ 操作会选择**高估**的 Q 值
- 导致**累积误差**：误差通过 Bellman 更新传播

**数学分析：**

假设 Q 函数估计有误差 $\epsilon$：

$$\hat{Q}(s, a) = Q^*(s, a) + \epsilon(s, a)$$

则 $\max$ 操作会放大误差：

$$\mathbb{E}[\max_a \hat{Q}(s, a)] \geq \max_a \mathbb{E}[\hat{Q}(s, a)] = \max_a Q^*(s, a)$$

这就是**最大化偏差（Maximization Bias）**。

### 3.3 为什么在线 RL 不受影响？

**在线 RL 的自我修正机制：**

1. 如果 Q 函数高估了某个动作
2. 策略会尝试这个动作
3. 发现实际奖励低
4. 更新 Q 函数，降低估计

**离线 RL 没有这个机制：**
- 不能尝试新动作
- 高估的 Q 值无法被纠正
- 误差累积


## 4. 朴素方法的失败

### 4.1 行为克隆（Behavior Cloning）

**方法：** 监督学习，直接模仿数据集中的动作。

$$\pi = \arg\min_\pi \mathbb{E}_{(s,a) \sim \mathcal{D}}[\ell(\pi(s), a)]$$

**问题：**
- 只能达到行为策略的水平
- 无法超越数据集
- 对分布偏移敏感（复合误差）

### 4.2 离线 Q-learning

**方法：** 直接在固定数据集上运行 Q-learning。

$$Q(s, a) \leftarrow r + \gamma \max_{a'} Q(s', a')$$

**问题：**
- 外推误差导致 Q 值严重高估
- 策略选择数据集外的动作
- 性能崩溃

**实验观察（Fu et al., 2020）：**
- 在 D4RL 基准上，离线 Q-learning 性能远低于行为策略
- 甚至出现负回报


## 5. CQL：保守 Q 学习

### 5.1 核心思想

**Conservative Q-Learning (Kumar et al., 2020)：**

**目标：** 学习一个**保守**的 Q 函数，在数据集外的动作上给出**低估**。

$$\min_Q \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp(Q(s, a)) - \mathbb{E}_{a \sim \pi_\beta(a|s)}[Q(s, a)] \right] + \text{Bellman Error}$$

**直觉：**
- 第一项：惩罚所有动作的 Q 值（使其降低）
- 第二项：提升数据集中动作的 Q 值
- 结果：数据集外的动作 Q 值被压低

### 5.2 CQL 算法

**完整目标函数：**

$$\mathcal{L}_{\text{CQL}}(Q) = \alpha \underbrace{\mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp(Q(s, a)) - \mathbb{E}_{a \sim \mathcal{D}}[Q(s, a)] \right]}_{\text{保守正则项}} + \underbrace{\mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ (Q(s,a) - (r + \gamma \max_{a'} Q(s', a')))^2 \right]}_{\text{Bellman 误差}}$$

**参数：**
- $\alpha$：保守程度（通常 1.0 - 10.0）
- 越大越保守，但可能过于悲观

**算法流程：**

```
初始化 Q 网络
对于每个训练步：
  1. 从数据集采样 batch: (s, a, r, s')
  2. 计算 Bellman 目标: y = r + γ max_a' Q(s', a')
  3. 计算 Bellman 损失: L_TD = (Q(s,a) - y)²
  4. 计算保守损失: L_CQL = log Σ_a exp(Q(s,a)) - Q(s,a)
  5. 总损失: L = L_TD + α * L_CQL
  6. 更新 Q
```

### 5.3 为什么 CQL 有效？

**理论保证（Kumar et al., 2020）：**

CQL 学到的 Q 函数满足：

$$Q_{\text{CQL}}(s, a) \leq Q^\pi(s, a), \quad \forall (s, a) \notin \mathcal{D}$$

即：对于数据集外的动作，CQL 给出**下界估计**。

**实践效果：**
- 避免选择数据集外的高估动作
- 策略倾向于选择数据集内的动作
- 在 D4RL 基准上显著优于离线 Q-learning


## 6. IQL：隐式 Q 学习

### 6.1 核心思想

**Implicit Q-Learning (Kostrikov et al., 2021)：**

**问题：** CQL 仍然使用 $\max$ 操作，可能高估。

**解决方案：** 用**期望**代替 $\max$。

$$V(s) = \mathbb{E}_{a \sim \pi_\beta}[Q(s, a)]$$

**关键创新：** 用**分位数回归**学习 V 函数。

### 6.2 IQL 算法

**三个网络：**

1. **Q 函数**：$Q_\theta(s, a)$
2. **V 函数**：$V_\psi(s)$
3. **策略**：$\pi_\phi(a|s)$

**更新规则：**

**1. 更新 V（分位数回归）：**

$$\mathcal{L}_V(\psi) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ L_\tau(Q_\theta(s, a) - V_\psi(s)) \right]$$

其中 $L_\tau$ 是分位数损失：

$$L_\tau(u) = |\tau - \mathbb{1}(u < 0)| \cdot u^2$$

- $\tau \in (0, 1)$：分位数参数（如 0.7）
- $\tau > 0.5$：V 学习 Q 的上分位数（乐观）
- $\tau < 0.5$：V 学习 Q 的下分位数（保守）

**2. 更新 Q（用 V 作为目标）：**

$$\mathcal{L}_Q(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ (Q_\theta(s, a) - (r + \gamma V_\psi(s')))^2 \right]$$

**3. 更新策略（AWR 风格）：**

$$\mathcal{L}_\pi(\phi) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \exp(\beta (Q_\theta(s, a) - V_\psi(s))) \log \pi_\phi(a|s) \right]$$

### 6.3 IQL 的优势

1. **简单**：没有复杂的正则项
2. **稳定**：避免 $\max$ 操作
3. **高效**：在 D4RL 上与 CQL 相当或更好
4. **理论优雅**：分位数回归有理论保证


## 7. Decision Transformer

### 7.1 序列建模视角

**Decision Transformer (Chen et al., 2021)：**

**核心思想：** 把 RL 当作**序列建模**问题。

**输入序列：**

$$(\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, ..., \hat{R}_T, s_T, a_T)$$

其中 $\hat{R}_t = \sum_{t'=t}^T r_{t'}$ 是**未来回报**（Return-to-Go）。

**目标：** 给定 $(\hat{R}_t, s_t)$，预测 $a_t$。

$$\pi(a_t | \hat{R}_t, s_t, \hat{R}_{t-1}, s_{t-1}, a_{t-1}, ...)$$

### 7.2 Transformer 架构

**模型：** GPT 风格的 Transformer

```
输入: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
      ↓
   Embedding
      ↓
   Transformer Blocks (Self-Attention + FFN)
      ↓
   预测 a_t
```

**训练：** 监督学习（交叉熵或 MSE）

$$\mathcal{L} = \mathbb{E}_{\tau \sim \mathcal{D}} \left[ \sum_t \ell(\pi(s_t, \hat{R}_t), a_t) \right]$$

### 7.3 推理：条件生成

**关键技巧：** 在推理时，设置**期望回报** $\hat{R}$。

```
初始化: R_desired = 100（期望总回报）
对于每个时间步 t：
  1. 输入: (R_desired, s_t, 历史)
  2. 模型输出: a_t
  3. 执行 a_t，获得 r_t, s_{t+1}
  4. 更新: R_desired ← R_desired - r_t
```

**直觉：** 模型学会"如果我想要回报 R，在状态 s 应该做什么"。

### 7.4 Decision Transformer 的特点

**优势：**
- 简单：标准监督学习
- 灵活：可以条件化任意目标回报
- 可扩展：利用 Transformer 的表达能力
- 多任务：一个模型处理不同回报水平

**局限：**
- 需要高质量数据（回报标签）
- 不能超越数据集中的最大回报
- 本质上是条件化的行为克隆


## 8. 方法对比

| 方法 | 核心思想 | 优势 | 劣势 |
|------|----------|------|------|
| **行为克隆** | 监督学习 | 简单 | 无法超越数据 |
| **离线 Q-learning** | 标准 Q-learning | 简单 | 外推误差严重 |
| **CQL** | 保守 Q 估计 | 理论保证，性能好 | 超参数敏感 |
| **IQL** | 期望 + 分位数 | 简单稳定 | 需要调 τ |
| **Decision Transformer** | 序列建模 | 灵活，可扩展 | 受限于数据质量 |

**实践建议：**
- **简单任务**：IQL（易调试）
- **需要理论保证**：CQL
- **大规模/多任务**：Decision Transformer
- **基线**：行为克隆


## 9. 数据质量的重要性

### 9.1 数据集类型

**D4RL 基准（Fu et al., 2020）分类：**

1. **Random**：随机策略收集
2. **Medium**：部分训练的策略
3. **Expert**：专家策略
4. **Medium-Replay**：训练过程中的所有数据
5. **Medium-Expert**：混合数据

### 9.2 性能与数据质量

**实验发现：**

| 数据集 | 行为克隆 | CQL | IQL |
|--------|----------|-----|-----|
| Random | 差 | 差 | 差 |
| Medium | 中 | 好 | 好 |
| Expert | 好 | 好 | 好 |
| Medium-Expert | 中 | **很好** | **很好** |

**结论：**
- 离线 RL 算法对数据质量敏感
- 混合数据集（多样性 + 质量）效果最好
- CQL/IQL 能从次优数据中提取价值


## 10. 实用建议

### 10.1 数据收集

1. **多样性**：覆盖状态-动作空间
2. **质量**：包含一些高回报轨迹
3. **混合策略**：不同水平的策略
4. **标注**：记录回报、策略信息

### 10.2 算法选择

**决策树：**

```
数据质量高（专家演示）？
  ├─ 是 → 行为克隆
  └─ 否 → 数据多样性好？
           ├─ 是 → CQL / IQL
           └─ 否 → 收集更多数据
```

### 10.3 调试技巧

1. **基线对比**：
   - 行为策略性能
   - 行为克隆性能

2. **Q 值分析**：
   - 检查 Q 值是否合理
   - 数据集内 vs 数据集外

3. **策略分析**：
   - 策略是否接近数据分布
   - 是否选择 OOD 动作


## 11. 前沿方向

### 11.1 在线微调（Online Fine-tuning）

**思路：** 离线预训练 + 在线微调

- 离线阶段：从数据集学习初始策略
- 在线阶段：少量交互改进策略

**代表算法：**
- AWAC（Nair et al., 2020）
- Cal-QL（Nakamoto et al., 2023）

### 11.2 离线到在线（Offline-to-Online）

**挑战：** 如何平滑过渡？

- 离线策略可能过于保守
- 直接在线可能不稳定

**方法：**
- 逐步减少保守性
- 混合离线和在线数据

### 11.3 模型基础的离线 RL

**思路：** 学习世界模型，在模型中规划。

- 避免直接的分布偏移
- 可以生成合成数据

**代表：**
- MOPO（Yu et al., 2020）
- MOReL（Kidambi et al., 2020）


## 12. 总结

### 12.1 核心要点

1. **离线 RL 动机**：
   - 利用历史数据，无需在线交互
   - 适用于高成本、高风险场景

2. **核心挑战**：
   - 分布偏移
   - 外推误差
   - 最大化偏差

3. **关键方法**：
   - **CQL**：保守 Q 估计，惩罚 OOD 动作
   - **IQL**：期望 + 分位数，避免 max
   - **Decision Transformer**：序列建模，条件生成

4. **实践建议**：
   - 数据质量至关重要
   - 从行为克隆基线开始
   - CQL/IQL 适合大多数场景

### 12.2 下节预告

下节课我们将学习 **RL 工程实践**：
- Stable-Baselines3 库
- Weights & Biases 实验跟踪
- 超参数调优
- 分布式训练

### 12.3 扩展阅读

**必读论文：**
1. Levine et al. (2020). "Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems"
2. Kumar et al. (2020). "Conservative Q-Learning for Offline Reinforcement Learning" (CQL)
3. Kostrikov et al. (2021). "Offline Reinforcement Learning with Implicit Q-Learning" (IQL)
4. Chen et al. (2021). "Decision Transformer: Reinforcement Learning via Sequence Modeling"

**推荐资源：**
- D4RL 基准：https://github.com/Farama-Foundation/D4RL
- Sergey Levine 的 CS285 Offline RL 讲座
- RL Unplugged 数据集
