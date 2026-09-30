# Lesson 08: Q-Learning — Off-Policy TD 控制

## 本课目标

1. 理解 **Off-Policy** 的核心思想
2. 掌握 **Q-Learning** 算法
3. 理解 SARSA vs Q-Learning 的本质区别
4. 在 CliffWalking 上对比两种算法的行为差异
5. 理解 **最大化偏差** (Maximization Bias) 和 **Double Q-Learning**

---

## 1. Off-Policy 的动机

### 1.1 On-Policy 的局限

回顾 SARSA (Lesson 07)：

```
行为策略（收集数据）= 目标策略（正在学习）= ε-greedy(Q)
```

**问题**：
- 学到的 Q 值是 ε-greedy 策略的，不是纯贪心策略的
- 无法利用旧策略的数据（数据效率低）
- 无法从人类演示或其他智能体的数据中学习

### 1.2 Off-Policy 的核心思想

**Off-Policy**：行为策略 ≠ 目标策略

```
行为策略 μ（收集数据）：ε-greedy(Q)  ← 探索性强
目标策略 π（正在学习）：greedy(Q)    ← 纯贪心
```

**优势**：
1. 学到的是最优策略（贪心），而不是探索性策略
2. 可以重用旧数据（经验回放）
3. 可以从其他策略的数据中学习（模仿学习）
4. 探索和利用完全解耦

---

## 2. Q-Learning 算法

### 2.1 更新规则

```
Q(S_t, A_t) ← Q(S_t, A_t) + α [R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
                                          ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                                          目标策略：贪心
```

**与 SARSA 的对比**：

| | SARSA | Q-Learning |
|---|---|---|
| 更新公式 | Q(S,A) + α[R + γQ(S',A') - Q(S,A)] | Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)] |
| 下一动作 | A' 由 ε-greedy 选出 | max_a（贪心） |
| 策略类型 | On-Policy | Off-Policy |
| 学到的策略 | ε-greedy | 纯贪心 |

### 2.2 关键区别

**SARSA**：
```
S_t, A_t → R_{t+1}, S_{t+1}
                    ↓
                  选 A_{t+1}（用 ε-greedy）
                    ↓
          用 Q(S_{t+1}, A_{t+1}) 更新
```

**Q-Learning**：
```
S_t, A_t → R_{t+1}, S_{t+1}
                    ↓
          用 max_a Q(S_{t+1}, a) 更新
          （不需要实际选 A_{t+1}）
```

Q-Learning 不需要知道下一步实际会采取什么动作，只需要知道**最优动作的 Q 值**。

### 2.3 完整算法

```
初始化：Q(s, a) = 0  ∀s, a

对每个 episode：
    S ← 起始状态

    重复（每一步）：
        A ← ε-greedy(Q, S)        # 行为策略：探索
        执行 A，观察 R, S'

        如果 S' 是终止状态：
            Q(S, A) ← Q(S, A) + α [R - Q(S, A)]
        否则：
            Q(S, A) ← Q(S, A) + α [R + γ max_a Q(S', a) - Q(S, A)]
                                        ‾‾‾‾‾‾‾‾‾‾‾‾‾
                                        目标策略：贪心

        S ← S'

        直到 S 是终止状态
```

### 2.4 为什么叫 Q-Learning？

因为它直接学习**最优动作价值函数 Q***，而不是某个特定策略的 Q^π。

---

## 3. Off-Policy 的理论基础

### 3.1 Bellman 最优方程

Q-Learning 的更新目标来自 Bellman 最优方程：

```
Q*(s, a) = E[R + γ max_{a'} Q*(s', a') | s, a]
```

Q-Learning 用采样近似这个期望：

```
Q(s, a) ← Q(s, a) + α [R + γ max_{a'} Q(s', a') - Q(s, a)]
                        ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                        采样的 Bellman 最优目标
```

### 3.2 收敛性

**定理** (Watkins & Dayan, 1992)：在满足以下条件时，Q-Learning 收敛到 Q*：
1. 每个 (s, a) 被无限次访问
2. Robbins-Monro 条件：Σ α_t = ∞，Σ α_t² < ∞
3. 奖励有界

**关键**：即使行为策略是 ε-greedy（探索性的），学到的仍然是 Q*（最优的）。

### 3.3 为什么 Off-Policy 有效？

直觉：Q-Learning 的更新只依赖于 (s, a, r, s')，不依赖于下一步实际采取的动作。只要 (s, a) 被访问到，就能正确更新 Q(s, a)。

---

## 4. SARSA vs Q-Learning：CliffWalking 对比

### 4.1 行为差异

在 CliffWalking 环境上，两种算法学到的策略截然不同：

```
SARSA（On-Policy）：
→ → → → → → → → → → → ↓
↑                       ↓
↑                       ↓
S · · · · · · · · · · G

安全路径：远离悬崖，因为考虑了 ε 探索可能掉下去

Q-Learning（Off-Policy）：
S → → → → → → → → → → G
· · · · · · · · · · · ·
· · · · · · · · · · · ·
· · · · · · · · · · · ·

最短路径：贴着悬崖走，因为学的是贪心策略（不考虑探索）
```

### 4.2 训练曲线对比

```
训练期间的平均奖励：
  SARSA:      -17 左右（安全但绕远）
  Q-Learning: -25 左右（训练时常掉悬崖，但学到最短路径）

评估时的平均奖励（贪心策略）：
  SARSA:      -17 左右（仍然绕远）
  Q-Learning: -13 左右（最短路径）
```

### 4.3 哪个更好？

**取决于应用场景**：

| 场景 | 推荐 |
|------|------|
| 训练时不能失败（如机器人） | SARSA（安全） |
| 训练时可以失败，追求最优 | Q-Learning（最优） |
| 需要利用旧数据 | Q-Learning（Off-Policy） |
| 在线学习，数据珍贵 | SARSA（On-Policy） |

---

## 5. 最大化偏差 (Maximization Bias)

### 5.1 问题

Q-Learning 的 max 操作会导致**正偏差**：

```
Q(s', a) 的估计有噪声（方差）
max_a Q(s', a) 倾向于选中被高估的动作
→ 系统性地高估 Q 值
```

**例子**：
```
真实 Q*(s, a) = [0, 0, 0, 0]（所有动作价值相同）
估计 Q(s, a) = [-0.1, 0.2, -0.3, 0.1]（有噪声）
max_a Q(s, a) = 0.2 > 0（高估了）
```

### 5.2 Double Q-Learning

**思路**：用两个独立的 Q 函数，一个选动作，另一个评估：

```
Q_1 和 Q_2 独立维护

更新 Q_1 时：
    A* = argmax_a Q_1(s', a)    # 用 Q_1 选动作
    Q_1(s, a) ← ... + γ Q_2(s', A*)  # 用 Q_2 评估
                        ‾‾‾‾‾‾‾‾
                        解耦选择和评估

更新 Q_2 时：
    A* = argmax_a Q_2(s', a)
    Q_2(s, a) ← ... + γ Q_1(s', A*)
```

每次更新随机选择更新 Q_1 或 Q_2。

**效果**：减少最大化偏差，Q 值估计更准确。

### 5.3 算法

```
初始化：Q_1(s, a) = 0, Q_2(s, a) = 0  ∀s, a

对每个 episode：
    S ← 起始状态

    重复：
        A ← ε-greedy(Q_1 + Q_2, S)  # 用两者的和选动作
        执行 A，观察 R, S'

        以 0.5 概率：
            A* = argmax_a Q_1(S', a)
            Q_1(S, A) ← Q_1(S, A) + α [R + γ Q_2(S', A*) - Q_1(S, A)]
        否则：
            A* = argmax_a Q_2(S', a)
            Q_2(S, A) ← Q_2(S, A) + α [R + γ Q_1(S', A*) - Q_2(S, A)]

        S ← S'
```

---

## 6. Q-Learning 的变体

### 6.1 Expected SARSA

介于 SARSA 和 Q-Learning 之间：

```
Q(S, A) ← Q(S, A) + α [R + γ Σ_a π(a|S') Q(S', a) - Q(S, A)]
                              ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                              期望（而非采样或 max）
```

- 比 SARSA 方差更低（用期望而非采样）
- 比 Q-Learning 偏差更小（不用 max）
- 可以是 On-Policy 或 Off-Policy

### 6.2 Q-Learning 的实践技巧

1. **学习率衰减**：α 从 0.5 逐渐降到 0.01
2. **ε 衰减**：从 1.0 降到 0.01
3. **初始化**：乐观初始化（Q = 大值）鼓励探索
4. **经验回放**（Deep Q-Learning 中）：存储 (s, a, r, s')，随机采样更新

---

## 7. 本课练习内容

你将实现：

1. **Q-Learning 算法**
2. **Double Q-Learning 算法**
3. **SARSA vs Q-Learning 对比**：在 CliffWalking 上
4. **策略可视化**：对比两种算法学到的路径
5. **最大化偏差实验**：验证 Double Q-Learning 的效果
6. **Expected SARSA**（可选）

测试环境：CliffWalking-v0

---

## 8. 关键概念总结

| 概念 | 定义 | 意义 |
|------|------|------|
| Off-Policy | 行为策略 ≠ 目标策略 | 可学最优策略，可重用数据 |
| Q-Learning | Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)] | Off-Policy TD 控制 |
| max_a | 贪心选择最优动作 | 学习最优策略 Q* |
| 最大化偏差 | max 操作导致高估 | 系统性偏差 |
| Double Q-Learning | 解耦动作选择和评估 | 减少偏差 |

---

## 9. 与前后课程的联系

| Lesson 07 | **Lesson 08** | Lesson 09 |
|-----------|---------------|-----------|
| SARSA | **Q-Learning** | 函数逼近 |
| On-Policy | **Off-Policy** | 线性/非线性 |
| 安全路径 | 最优路径 | 处理大状态空间 |
| ε-greedy 策略 | 贪心策略 | 神经网络 Q |

**过渡**：Lesson 01-08 都是**表格型方法**（Tabular Methods），状态和动作都是离散且数量有限的。Lesson 09 将引入**函数逼近**，用线性函数或神经网络表示 Q，从而处理大规模甚至连续的状态空间。这是通向 Deep RL 的关键一步。

---

## 10. 历史意义

Q-Learning 是 RL 历史上最重要的算法之一：

- **1989**：Watkins 在博士论文中提出
- **1992**：Watkins & Dayan 证明收敛性
- **2013**：DeepMind 的 DQN（Deep Q-Network）将 Q-Learning 与深度学习结合，在 Atari 游戏上达到人类水平
- **2015**：DQN 登上 Nature 封面，标志着 Deep RL 时代的开启

Q-Learning 的核心思想（Off-Policy + Bellman 最优方程）至今仍是现代 Deep RL 的基石。

---

## 参考资料

- Sutton & Barto 第 6.5 节：Q-learning: Off-policy TD Control
- Watkins (1989) "Learning from Delayed Rewards" (博士论文)
- Watkins & Dayan (1992) "Q-learning" (Machine Learning)
- van Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-learning"
- David Silver Lecture 5: Model-Free Control
