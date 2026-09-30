# Lesson 01: 强化学习概览与马尔可夫决策过程 (MDP)

---

## 1. 什么是强化学习？

### 1.1 学习范式对比

你已经非常熟悉监督学习（语音增强本质上就是一种映射学习）。让我们从对比出发：

| | 监督学习 | 无监督学习 | 强化学习 |
|---|---|---|---|
| **数据** | (x, y) 对 | x（无标签） | (s, a, r, s') 交互序列 |
| **反馈** | 即时、精确的标签 | 无反馈 | 延迟、稀疏的奖励信号 |
| **目标** | 最小化预测误差 | 发现数据结构 | 最大化累积奖励 |
| **数据分布** | i.i.d. 假设 | i.i.d. 假设 | 非 i.i.d.，动作影响后续数据 |

**核心区别**：在 RL 中，智能体的动作会改变它未来看到的数据分布。这是和你做语音增强最大的不同——你的模型不会因为输出了一个降噪结果而改变下一帧输入，但 RL 智能体的每一步决策都在塑造未来。

### 1.2 RL 的核心要素

一个 RL 问题包含：

- **Agent（智能体）**：做决策的主体
- **Environment（环境）**：智能体交互的外部世界
- **State（状态）s**：环境的描述
- **Action（动作）a**：智能体可以执行的操作
- **Reward（奖励）r**：环境给智能体的标量反馈信号

交互循环：

```
t=0:  Agent 观测 s_0 → 选择 a_0 → 环境返回 r_1, s_1
t=1:  Agent 观测 s_1 → 选择 a_1 → 环境返回 r_2, s_2
...
```

这个循环生成一条**轨迹（trajectory / episode）**：

$$\tau = (s_0, a_0, r_1, s_1, a_1, r_2, s_2, \ldots)$$

### 1.3 类比：语音增强 vs RL

把你熟悉的语音增强想象成 RL 问题：
- **State**：当前帧的含噪语音特征
- **Action**：选择一个增强mask或滤波器参数
- **Reward**：增强后的语音质量（PESQ 分数？SI-SNR 提升？）
- **关键差异**：在你的实际工作中，每帧独立处理（或有限依赖），而真正的 RL 问题中，当前的 action 会从根本上改变未来的 state

---

## 2. 马尔可夫决策过程 (MDP)

MDP 是 RL 的数学基石。你可以把它理解为 RL 问题的"数据结构定义"。

### 2.1 马尔可夫性质 (Markov Property)

**定义**：未来状态只依赖于当前状态，与历史无关。

$$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0, a_0) = P(s_{t+1} | s_t, a_t)$$

用你的话说：当前状态已经"压缩"了所有历史信息，是一个**充分统计量**。

这和信号处理中的思想相通——如果一个系统的状态向量足够丰富，我们就不需要回溯历史。在语音处理中，LSTM/Transformer 的隐状态其实就在起这个作用。

### 2.2 MDP 五元组

一个 MDP 由五元组 $(S, A, P, R, \gamma)$ 定义：

**S — 状态空间 (State Space)**
- 所有可能状态的集合
- 可以是离散的（棋盘格子）或连续的（机器人关节角度）

**A — 动作空间 (Action Space)**
- 所有可能动作的集合
- 同样可以离散或连续

**P — 状态转移函数 (Transition Function)**

$$P(s' | s, a) = \Pr(s_{t+1} = s' | s_t = s, a_t = a)$$

这是一个条件概率分布，给定当前状态和动作，描述下一个状态的概率。注意，转移可以是**随机的**。

**R — 奖励函数 (Reward Function)**

$$R(s, a, s') = \mathbb{E}[r_{t+1} | s_t = s, a_t = a, s_{t+1} = s']$$

有时简写为 $R(s, a)$ 或 $R(s)$。返回一个标量值。

**γ — 折扣因子 (Discount Factor)**
- $\gamma \in [0, 1]$
- 控制对未来奖励的重视程度
- $\gamma = 0$：只关心即时奖励（近视）
- $\gamma = 1$：同等重视所有未来奖励（远视）
- 实践中常用 $\gamma = 0.99$ 或 $\gamma = 0.999$

### 2.3 累积回报 (Return)

从时刻 t 开始的**折扣累积回报**：

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \ldots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

注意递推关系（后面推导 Bellman 方程会用到）：

$$G_t = r_{t+1} + \gamma G_{t+1}$$

**为什么要折扣？** 三个原因：
1. 数学上保证 $G_t$ 有界（当 $\gamma < 1$，$|G_t| \leq \frac{R_{max}}{1 - \gamma}$）
2. 建模"不确定性"——未来越远越不可靠
3. 经济学中的"时间价值"——今天的一块钱比明天的值钱

---

## 3. 策略与价值函数

### 3.1 策略 (Policy)

策略 $\pi$ 定义了智能体的行为方式：

**确定性策略**：$a = \pi(s)$

**随机策略**：$\pi(a|s) = \Pr(a_t = a | s_t = s)$

用深度学习的类比：策略就是一个以状态为输入、以动作（或动作分布）为输出的函数。你可以把它理解为一个"决策网络"。

### 3.2 状态价值函数 (State-Value Function)

在策略 $\pi$ 下，状态 $s$ 的价值：

$$V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \middle| s_t = s\right]$$

直觉：从状态 $s$ 出发，按照策略 $\pi$ 行动，期望能获得多少总回报？

### 3.3 动作价值函数 (Action-Value Function)

在策略 $\pi$ 下，状态 $s$ 执行动作 $a$ 的价值：

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]$$

直觉：在状态 $s$ 先执行动作 $a$，之后按 $\pi$ 行动，期望总回报是多少？

**V 和 Q 的关系**：

$$V^\pi(s) = \sum_{a \in A} \pi(a|s) Q^\pi(s, a) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)]$$

---

## 4. Bellman 方程 — RL 的核心递推

### 4.1 Bellman 期望方程 (Bellman Expectation Equation)

利用 $G_t = r_{t+1} + \gamma G_{t+1}$ 的递推性质：

$$V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]$$
$$= \mathbb{E}_\pi[r_{t+1} + \gamma G_{t+1} | s_t = s]$$
$$= \mathbb{E}_\pi[r_{t+1} + \gamma V^\pi(s_{t+1}) | s_t = s]$$

展开期望（对动作和下一状态求和）：

$$\boxed{V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]}$$

这就是 **Bellman 期望方程**。它表达了一个核心思想：**一个状态的价值等于即时奖励加上后继状态的折扣价值的期望**。

类似地，Q 函数的 Bellman 期望方程：

$$\boxed{Q^\pi(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a') \right]}$$

### 4.2 Bellman 最优方程 (Bellman Optimality Equation)

定义最优价值函数：

$$V^*(s) = \max_\pi V^\pi(s), \quad Q^*(s,a) = \max_\pi Q^\pi(s,a)$$

最优策略下，智能体总是选择让价值最大的动作：

$$\boxed{V^*(s) = \max_{a} \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]}$$

$$\boxed{Q^*(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a'} Q^*(s', a') \right]}$$

**注意和期望方程的关键区别**：$\sum_a \pi(a|s)$ 变成了 $\max_a$。最优策略是贪心的——在每个状态都选最好的动作。

### 4.3 矩阵形式（你会喜欢这个）

对于有限状态 MDP，Bellman 期望方程可以写成矩阵形式。设 $|\mathcal{S}| = n$：

$$\mathbf{v}^\pi = \mathbf{r}^\pi + \gamma \mathbf{P}^\pi \mathbf{v}^\pi$$

其中：
- $\mathbf{v}^\pi \in \mathbb{R}^n$ 是价值向量
- $\mathbf{r}^\pi \in \mathbb{R}^n$ 是每个状态的期望即时奖励
- $\mathbf{P}^\pi \in \mathbb{R}^{n \times n}$ 是策略 $\pi$ 下的状态转移矩阵

直接求解：

$$\mathbf{v}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{r}^\pi$$

复杂度 $O(n^3)$，只适用于小规模问题。但这个精确解非常适合验证你的迭代算法是否正确！

---

## 5. 一个完整的例子：4x4 GridWorld

让我们用一个具体例子串联所有概念。

```
+----+----+----+----+
| S  |    |    |    |
+----+----+----+----+
|    |  X |    |    |
+----+----+----+----+
|    |    |    |    |
+----+----+----+----+
|    |    |    |  G |
+----+----+----+----+

S = 起点 (0,0)
G = 终点 (3,3)，到达奖励 +1
X = 陷阱 (1,1)，到达奖励 -1
其他移动：奖励 -0.04（鼓励快速到达终点）
撞墙：留在原地
γ = 0.9
```

**MDP 定义**：
- $S$: 16 个格子 {(0,0), (0,1), ..., (3,3)}
- $A$: {上, 下, 左, 右}
- $P$: 确定性转移（动作一定成功，除非撞墙）
- $R$: 到达 G 得 +1，到达 X 得 -1，其他 -0.04
- $\gamma$: 0.9

**随机策略**下（四个方向等概率），你可以用 Bellman 方程组建立一个 16 元线性方程组来求解 $V^\pi$。

---

## 6. 本课要点总结

1. RL 与监督学习的核心区别：**交互式、延迟奖励、非 i.i.d.**
2. MDP 五元组 $(S, A, P, R, \gamma)$ 是 RL 的数学框架
3. **策略** $\pi$ 定义行为，**价值函数** $V^\pi, Q^\pi$ 评估策略好坏
4. **Bellman 方程**建立了价值函数的递推关系，是所有 RL 算法的基石
5. Bellman **期望**方程用于评估给定策略；Bellman **最优**方程用于寻找最优策略

---

## 参考资料

- Sutton & Barto, Chapter 3: Finite Markov Decision Processes
- David Silver RL Course, Lecture 2: Markov Decision Processes
- [MDP Wikipedia](https://en.wikipedia.org/wiki/Markov_decision_process)
