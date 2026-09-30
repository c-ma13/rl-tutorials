# Lesson 12: 探索策略专题

## 本课目标

1. 系统理解**探索-利用困境** (Exploration-Exploitation Dilemma)
2. 掌握 **Multi-Armed Bandit** 问题框架
3. 实现并对比多种探索策略：ε-greedy、UCB、Thompson Sampling
4. 理解**好奇心驱动**和 **RND** 等深度探索方法的思想
5. 总结探索策略在 RL 各算法中的应用

---

## 1. 探索-利用困境

### 1.1 核心问题

```
利用 (Exploitation)：选择当前已知最好的动作 → 短期最优
探索 (Exploration)：尝试未知的动作 → 可能发现更好的

困境：过度利用 → 错过更好的选择
      过度探索 → 浪费资源在差的动作上
```

**例子**：
- 餐厅选择：总去最爱的店（利用）vs 尝试新店（探索）
- 广告投放：投已知高点击率的（利用）vs 测试新创意（探索）
- RL 智能体：选 Q 值最高的动作 vs 尝试其他动作

### 1.2 不同层面的探索

| 层面 | 已学方法 | 本课新内容 |
|------|----------|-----------|
| 动作选择 | ε-greedy (L05-08) | UCB, Thompson Sampling |
| 策略随机性 | softmax 策略 (L10) | 熵正则化, 噪声网络 |
| 内在动机 | — | 好奇心, RND |
| 系统探索 | — | Count-based, 信息增益 |

---

## 2. Multi-Armed Bandit 问题

### 2.1 问题定义

**K-Armed Bandit**（K 臂老虎机）：

```
- K 个动作（拉杆），每个有未知的奖励分布
- 每次选择一个动作，获得随机奖励
- 目标：最大化累积奖励（或最小化遗憾）
```

**与 RL 的关系**：
- Bandit = 只有 1 个状态的 MDP
- 没有状态转移，只有动作选择
- 探索策略的纯粹测试场

### 2.2 数学形式

```
动作 a ∈ {1, 2, ..., K}
真实价值：q*(a) = E[R_t | A_t = a]
估计价值：Q_t(a) ≈ q*(a)
最优动作：a* = argmax_a q*(a)
```

### 2.3 遗憾 (Regret)

```
即时遗憾：r_t = q*(a*) - q*(A_t)
累积遗憾：R_T = Σ_{t=1}^T r_t
```

**目标**：最小化 R_T。好的探索策略使 R_T 增长尽可能慢。

**理论下界** (Lai & Robbins, 1985)：
```
R_T ≥ Ω(K log T)  （任何算法都无法避免）
```

---

## 3. 探索策略

### 3.1 ε-Greedy

```
以概率 ε: 随机选动作
以概率 1-ε: 选 Q 值最大的动作
```

**优点**：简单，保证所有动作被探索
**缺点**：不区分"几乎确定差"和"不确定"的动作

**遗憾**：R_T = O(ε T + K/ε) → 选 ε = O(√(K/T)) 可得 O(√(KT))

### 3.2 ε 递减策略

```
ε_t = min(1, C/(d² t))
其中 C > 0, d = min gap between optimal and suboptimal
```

**实践中**常用：
```python
epsilon = max(epsilon_end, epsilon_start * decay**t)
```

### 3.3 UCB (Upper Confidence Bound)

**思想**：选择"乐观估计"最高的动作

```
A_t = argmax_a [Q_t(a) + c √(ln t / N_t(a))]
                ‾‾‾‾‾‾   ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                利用项         探索项（不确定性奖励）
```

- `Q_t(a)`：当前估计值（利用）
- `c √(ln t / N_t(a))`：置信上界（探索）
  - `N_t(a)` 小（很少尝试）→ 探索项大 → 鼓励尝试
  - `N_t(a)` 大（多次尝试）→ 探索项小 → 转向利用

**优点**：
- 理论最优：R_T = O(√(KT log T))
- 自动平衡探索和利用
- 不需要调 ε

**缺点**：
- 需要计数 N_t(a)，表格型方法才方便
- 不适合非平稳环境

### 3.4 Thompson Sampling（汤普森采样）

**思想**：维护每个动作奖励分布的后验概率，从后验采样决策

```
算法（Bernoulli Bandit）：
    对每个动作 a：
        维护 Beta 分布参数 (α_a, β_a)
    
    每步：
        对每个动作 a：采样 θ_a ~ Beta(α_a, β_a)
        选择 A_t = argmax_a θ_a
        观察奖励 R_t
        更新后验：
            如果 R_t = 1: α_{A_t} += 1
            如果 R_t = 0: β_{A_t} += 1
```

**对于高斯奖励**：
```
维护正态分布 N(μ_a, σ_a²)
每步：采样 θ_a ~ N(μ_a, σ_a²)
选择 A_t = argmax_a θ_a
```

**优点**：
- 理论最优（渐近）
- 自动适应不确定性
- 实现简单

**缺点**：
- 需要选择先验分布
- 计算可能较贵（复杂后验）

### 3.5 Softmax/Boltzmann 探索

```
π(a|s) = exp(Q(s,a) / τ) / Σ_a' exp(Q(s,a') / τ)
```

- τ → ∞：均匀随机
- τ → 0：贪心
- τ 控制探索程度

**优点**：Q 值差异大的动作被探索概率低（合理）
**缺点**：对 Q 值尺度敏感

### 3.6 乐观初始化

```
初始化 Q(s, a) = 大值（如 5.0）
```

初期所有动作都被高估 → 自然探索所有动作 → 逐渐修正。

**优点**：极简，不需要修改算法
**缺点**：只在初期有效，非平稳环境不适用

---

## 4. 三种策略的直觉对比

```
ε-greedy:
    "随机抽签决定是否探索"
    → 不看 Q 值差异，盲目探索

UCB:
    "给不确定的选项加分"
    → 确定性的，基于计数

Thompson Sampling:
    "相信概率，让运气决定"
    → 随机性的，基于后验分布
```

---

## 5. 深度探索方法（简介）

### 5.1 Count-Based 探索

```
奖励 = 外部奖励 + β / √(N(s))
```

访问次数少的状态获得额外"探索奖励"。

**问题**：连续/高维状态空间无法精确计数。

**解决**：用密度模型近似计数（伪计数, Pseudo-Count）。

### 5.2 好奇心驱动 (Curiosity-Driven)

```
ICM (Intrinsic Curiosity Module)：
    奖励 = 外部奖励 + η × 预测误差
```

如果状态转移"出乎意料"（预测误差大），给予高奖励 → 鼓励去未知区域。

### 5.3 RND (Random Network Distillation)

```
固定随机网络：f(s) → z_target
可学习网络：  f̂(s) → z_predict

内在奖励 = ||z_target - z_predict||²
```

- 新状态：预测误差大 → 高内在奖励
- 旧状态：预测误差小 → 低内在奖励

**优点**：简单、高效、不需要动力学模型。

### 5.4 与 LLM 的联系

RLHF 中的探索问题：
- **Reward Hacking**：策略找到奖励模型的漏洞 → KL 惩罚限制探索范围
- **GRPO**：组内相对排序 → 隐式探索
- **Constitutional AI**：自我改进 → 结构化探索

---

## 6. 本课练习内容

你将实现：

1. **K-Armed Bandit 环境**
2. **ε-Greedy 策略**（固定 ε + 递减 ε）
3. **UCB 策略**
4. **Thompson Sampling 策略**
5. **Softmax/Boltzmann 策略**
6. **乐观初始化**
7. **对比实验**：累积遗憾曲线、最优动作选择比例
8. **非平稳环境**测试

---

## 7. 关键概念总结

| 策略 | 核心思想 | 遗憾 | 适用场景 |
|------|----------|------|---------|
| ε-greedy | 固定概率随机 | O(εT) | 简单、通用 |
| UCB | 乐观面对不确定 | O(√(KT log T)) | 平稳、表格型 |
| Thompson | 后验采样 | O(√(KT)) | 平稳、贝叶斯 |
| Softmax | 按 Q 值比例 | — | Q 值可靠时 |
| 乐观初始化 | 高初始值 | — | 初期探索 |

---

## 8. 与前后课程的联系

| Lesson 11 | **Lesson 12** | Lesson 13 |
|-----------|---------------|-----------|
| Actor-Critic | **探索策略** | DQN |
| 熵正则化 | 系统化探索 | 经验回放 |
| 策略梯度 | Bandit 理论 | 深度 RL |

**阶段二总结**：
- L07-08：SARSA/Q-Learning → 核心 TD 控制
- L09：函数逼近 → 处理大规模问题
- L10-11：策略梯度/Actor-Critic → 连续动作、随机策略
- L12：探索策略 → 贯穿所有算法的核心问题

**阶段三预告**：进入 Deep RL（DQN → PPO → SAC），所有基础已就绪。

---

## 参考资料

- Sutton & Barto 第 2 章：Multi-armed Bandits
- Lattimore & Szepesvári (2020) "Bandit Algorithms"
- Auer et al. (2002) "Finite-time Analysis of the Multiarmed Bandit Problem" (UCB 原论文)
- Thompson (1933) "On the Likelihood that One Unknown Probability Exceeds Another"
- Burda et al. (2019) "Exploration by Random Network Distillation" (RND)
- Pathak et al. (2017) "Curiosity-driven Exploration by Self-Supervised Prediction" (ICM)
