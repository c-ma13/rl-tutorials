# Lesson 02: Bellman 方程深入 — 算子、收缩映射与精确求解

---

## 0. 回顾与本课目标

Lesson 01 中你已经掌握了：
- MDP 五元组 $(S, A, P, R, \gamma)$
- Bellman 期望方程：$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$
- 矩阵法精确求解 $\mathbf{v}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{r}^\pi$

本课的核心问题：

1. 矩阵求逆是 $O(n^3)$，大规模 MDP 怎么办？ → **迭代法**
2. 迭代法凭什么收敛？ → **收缩映射定理**
3. 如何从 $V^\pi$ 计算 $Q^\pi$？如何找到最优策略？ → **V-Q 关系与 Bellman 最优算子**

---

## 1. V 与 Q 的计算关系

### 1.1 从 V 到 Q

在 Lesson 01 中我们分别定义了 $V^\pi(s)$ 和 $Q^\pi(s,a)$，但没有详细讨论它们之间的**计算转换**。

**已知 $V^\pi$，计算 $Q^\pi$**：

$$Q^\pi(s, a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

直觉：$Q^\pi(s,a)$ 就是"先固定动作 $a$，然后看后续按 $\pi$ 走能获得多少回报"。

对于**确定性转移**（如我们的 GridWorld），这简化为：

$$Q^\pi(s, a) = R(s, a, s'_a) + \gamma V^\pi(s'_a)$$

其中 $s'_a$ 是在状态 $s$ 执行动作 $a$ 后确定到达的下一个状态。

### 1.2 从 Q 到 V

$$V^\pi(s) = \sum_a \pi(a|s) \, Q^\pi(s, a) = \mathbb{E}_{a \sim \pi(\cdot|s)}[Q^\pi(s,a)]$$

$V^\pi$ 是 $Q^\pi$ 在策略分布下的期望 —— 这就是为什么 $Q$ 比 $V$ 信息量更大：$Q$ 记录了每个动作的具体价值，$V$ 只是策略加权平均。

### 1.3 最优策略的提取

如果你知道 $Q^*(s,a)$，最优策略就是：

$$\pi^*(a|s) = \begin{cases} 1 & \text{if } a = \arg\max_{a'} Q^*(s, a') \\ 0 & \text{otherwise} \end{cases}$$

如果你只知道 $V^*(s)$，提取策略还需要环境模型 $P(s'|s,a)$：

$$\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$$

这就是为什么很多算法偏好学 $Q$ —— **$Q$ 函数本身就包含了选择最优动作所需的全部信息**，不需要环境模型。

> **类比**：这和语音增强中直接预测 mask vs 预测中间特征再推 mask 的区别类似。直接输出 mask 更端到端，如同直接学 Q 更方便做决策。

---

## 2. Bellman 算子 (Bellman Operator)

### 2.1 什么是算子？

在函数分析中，**算子**是"函数的函数"——接受一个函数作为输入，输出另一个函数。

你在信号处理中已经很熟悉这个概念：
- 卷积算子 $\mathcal{F}$：输入信号 $x(t)$，输出 $(h * x)(t)$
- 频域变换算子：输入时域信号，输出频谱

Bellman 算子的思想完全一样——它接受一个价值函数 $v$，输出一个"更新后"的价值函数。

### 2.2 Bellman 期望算子 $T^\pi$

给定策略 $\pi$，定义 **Bellman 期望算子** $T^\pi$：

$$(T^\pi v)(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma v(s') \right]$$

它的作用：给定任意价值函数 $v$（不一定是 $V^\pi$），计算"按 $\pi$ 走一步后的期望回报 + 后续价值"。

**关键性质**：$V^\pi$ 是 $T^\pi$ 的**不动点**（fixed point）：

$$T^\pi V^\pi = V^\pi$$

这正是 Bellman 期望方程的算子表达。

用矩阵记号：$T^\pi \mathbf{v} = \mathbf{r}^\pi + \gamma \mathbf{P}^\pi \mathbf{v}$

### 2.3 Bellman 最优算子 $T^*$

定义 **Bellman 最优算子** $T^*$：

$$(T^* v)(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma v(s') \right]$$

与 $T^\pi$ 的区别：$\sum_a \pi(a|s)$ 变成了 $\max_a$。

**关键性质**：$V^*$ 是 $T^*$ 的**不动点**：

$$T^* V^* = V^*$$

---

## 3. 收缩映射定理 (Contraction Mapping Theorem)

这是理解所有迭代 RL 算法收敛性的数学基石。

### 3.1 直觉

想象你有一张橡皮地图，每次操作都把地图上所有点之间的距离缩小到原来的 $\gamma$ 倍。不管初始状态如何，反复操作后所有点最终都会"坍缩"到同一个点 —— 这就是不动点。

信号处理类比：一个增益 $< 1$ 的反馈系统，无论初始状态如何，输出最终都会衰减到稳态。$\gamma$ 就是这个反馈增益。

### 3.2 形式化定义

**定义（收缩映射）**：在度量空间 $(X, d)$ 中，映射 $T: X \to X$ 是 $\gamma$-收缩的，如果：

$$d(Tv, Tw) \leq \gamma \cdot d(v, w), \quad \forall v, w \in X, \quad \gamma \in [0, 1)$$

**Banach 不动点定理**：如果 $T$ 是完备度量空间上的 $\gamma$-收缩映射，则：
1. $T$ 有且仅有一个不动点 $v^*$
2. 从任意初值 $v_0$ 出发，$v_{k+1} = T v_k$ 收敛到 $v^*$
3. 收敛速度：$\|v_k - v^*\|_\infty \leq \gamma^k \|v_0 - v^*\|_\infty$（几何/指数收敛）

### 3.3 Bellman 算子是收缩映射

**定理**：$T^\pi$ 和 $T^*$ 都是在 $\|\cdot\|_\infty$ 范数下的 $\gamma$-收缩映射。

**证明（对 $T^\pi$）**：

$$\|T^\pi v - T^\pi w\|_\infty = \max_s |(T^\pi v)(s) - (T^\pi w)(s)|$$

$$= \max_s \left| \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \gamma [v(s') - w(s')] \right|$$

$$\leq \max_s \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \, \gamma |v(s') - w(s')|$$

$$\leq \gamma \max_{s'} |v(s') - w(s')| = \gamma \|v - w\|_\infty$$

关键步骤：$\sum_a \pi(a|s) = 1$，$\sum_{s'} P(s'|s,a) = 1$（概率归一化），所以加权平均不超过最大值。

$T^*$ 的证明类似，利用 $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$。

### 3.4 实践意义

这个定理告诉我们：

| 性质 | 含义 |
|------|------|
| 不动点唯一 | $V^\pi$ 和 $V^*$ 各只有一个解 |
| 任意初始化 | 初始 $v_0$ 可以全设为 0，不影响最终结果 |
| 几何收敛 | 每次迭代误差缩小 $\gamma$ 倍，$\gamma = 0.9$ 时 44 次迭代误差降到 $10^{-2}$ |
| $\gamma$ 是收敛率 | $\gamma$ 越小收敛越快，但价值函数更"近视" |

> **信号处理视角**：$T^\pi \mathbf{v} = \mathbf{r}^\pi + \gamma \mathbf{P}^\pi \mathbf{v}$ 是一个仿射变换。其线性部分 $\gamma \mathbf{P}^\pi$ 的谱半径 $\leq \gamma < 1$（因为 $\mathbf{P}^\pi$ 是随机矩阵，谱半径 $\leq 1$）。所以矩阵 $(\mathbf{I} - \gamma \mathbf{P}^\pi)$ 可逆，系统有唯一稳态——这和 Lesson 01 的矩阵法完全对应。

---

## 4. 迭代求解方法

### 4.1 迭代策略评估 (Iterative Policy Evaluation)

反复应用 $T^\pi$ 来逼近 $V^\pi$：

```
v_0 = 0  (或任意初始化)
v_{k+1} = T^π v_k
重复直到 ||v_{k+1} - v_k||_∞ < ε
```

每一步的计算：
$$v_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma v_k(s')]$$

**复杂度**：每次迭代 $O(|S|^2 |A|)$，比矩阵求逆 $O(|S|^3)$ 更适合大状态空间。

### 4.2 价值迭代 (Value Iteration) — 预告

反复应用 $T^*$ 来逼近 $V^*$：

```
v_0 = 0
v_{k+1} = T* v_k
重复直到 ||v_{k+1} - v_k||_∞ < ε
```

每一步的计算：
$$v_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma v_k(s')]$$

价值迭代将在 Lesson 04 详细讨论。本课重点是理解这两个算子的理论基础。

---

## 5. 策略枚举与最优策略（小规模 MDP）

### 5.1 暴力枚举的可行性

对于状态数和动作数都很小的 MDP，一种直观的方法是：枚举所有可能的确定性策略，逐一求解 $V^\pi$，比较后选最优。

确定性策略数量 = $|A|^{|S_{non-terminal}|}$

对于 $|A| = 2, |S_{non-terminal}| = 3$ 的问题：$2^3 = 8$ 种策略，完全可行。
对于 $|A| = 4, |S_{non-terminal}| = 14$ 的 4×4 GridWorld：$4^{14} \approx 2.7 \times 10^8$，不可行。

### 5.2 2×2 TinyGridWorld 示例

为了演示策略枚举，考虑一个极简的 2×2 网格：

```
+-----+-----+
| S=0 |  1  |
+-----+-----+
|  2  | G=3 |
+-----+-----+

S = 起点 (0,0), 状态索引 0
G = 终点 (1,1), 状态索引 3, reward = +1
其他移动: reward = -0.1
动作: 0=右, 1=下
撞墙: 留在原地
γ = 0.9
```

非终止状态 3 个 {0, 1, 2}，每个有 2 种动作选择，共 $2^3 = 8$ 种确定性策略。

**手算示例 — 策略 (右, 下, 右)**：

即 $\pi(0) = \text{右}, \pi(1) = \text{下}, \pi(2) = \text{右}$

转移过程：
- 状态 0 → 右 → 状态 1，奖励 -0.1
- 状态 1 → 下 → 状态 3(Goal)，奖励 +1
- 状态 2 → 右 → 状态 3(Goal)，奖励 +1

建立方程组：
$$V(0) = -0.1 + 0.9 \cdot V(1)$$
$$V(1) = 1.0 + 0.9 \cdot V(3) = 1.0 + 0 = 1.0$$
$$V(2) = 1.0 + 0.9 \cdot V(3) = 1.0$$
$$V(3) = 0 \text{ (终止态)}$$

解得 $V(0) = -0.1 + 0.9 \times 1.0 = 0.8$

在练习中你将枚举全部 8 种策略，找到最优策略并验证 Bellman 最优方程。

---

## 6. 本课要点总结

1. **V-Q 转换**：已知 $V^\pi$ 可以计算 $Q^\pi$；$Q$ 包含的信息比 $V$ 更多，可以直接用于提取策略
2. **Bellman 算子**：将 Bellman 方程重写为 $T^\pi v = v$ 的不动点问题，$T^\pi$ 和 $T^*$ 是两个核心算子
3. **收缩映射定理**：$T^\pi$ 和 $T^*$ 都是 $\gamma$-收缩映射，保证不动点唯一且迭代收敛
4. **迭代求解**：反复应用 Bellman 算子可以逼近精确解，是 DP 和 RL 算法的基础
5. **策略枚举**：小规模 MDP 可以穷举所有策略找最优，但规模增长后必须依赖迭代算法

---

## 参考资料

- Sutton & Barto, Chapter 4.1-4.3: Dynamic Programming
- Bertsekas "Dynamic Programming and Optimal Control", Chapter 1.3: Contraction Mappings
- David Silver RL Course, Lecture 3: Planning by Dynamic Programming
