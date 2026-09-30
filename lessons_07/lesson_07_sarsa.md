# Lesson 07: SARSA — On-Policy TD 控制

## 本课目标

1. 理解从 **TD 预测**到 **TD 控制**的跨越
2. 掌握 **SARSA** 算法（On-Policy TD 控制）
3. 理解 **On-Policy** 的含义和局限
4. 实现 **SARSA(λ)**（资格迹版本）
5. 在 CliffWalking 环境上训练智能体

---

## 1. 从预测到控制

### 1.1 回顾：预测 vs 控制

| | 预测 (Prediction) | 控制 (Control) |
|---|---|---|
| 目标 | 给定策略 π，求 V^π 或 Q^π | 找到最优策略 π* |
| 已学 | MC 评估 (L05), TD(0) 评估 (L06) | DP 策略/价值迭代 (L03-04) |
| 本课 | — | **SARSA (TD 控制)** |

### 1.2 GPI 框架下的 TD 控制

回忆广义策略迭代 (GPI)：

```
评估：Q^π ← 评估当前策略
改进：π ← ε-greedy(Q^π)
```

- Lesson 05 的 MC 控制：用 MC 做评估
- **本课的 SARSA**：用 TD(0) 做评估

### 1.3 为什么用 Q 不用 V？

和 MC 控制一样，model-free 控制**必须**学 Q 函数：

```
用 V 做改进需要模型：  π'(s) = argmax_a Σ_{s'} P(s'|s,a) [R + γV(s')]
                                              ^^^^^^^^
                                              需要 P！

用 Q 做改进不需要模型：π'(s) = argmax_a Q(s, a)
                                 ✓ 直接可算
```

---

## 2. SARSA 算法

### 2.1 名字的由来

SARSA 的名字来自更新所需的五元组：

```
(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})
  S     A      R       S        A
```

每一步都需要：当前状态、当前动作、奖励、下一状态、**下一动作**。

### 2.2 更新规则

```
Q(S_t, A_t) ← Q(S_t, A_t) + α [R_{t+1} + γ Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
                                  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                                            TD 目标
```

**与 TD(0) 的对比**：

| TD(0) 预测 | SARSA 控制 |
|---|---|
| V(S) ← V(S) + α[R + γV(S') - V(S)] | Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)] |
| 学 V 函数 | 学 Q 函数 |
| 不需要选动作 | 需要选下一个动作 A' |

### 2.3 On-Policy 的含义

**On-Policy**：用于生成数据的策略 = 正在优化的策略。

```
SARSA 中：
  用 ε-greedy(Q) 选动作（行为策略）
  更新 Q 时用的是 ε-greedy 选出的 A'（目标策略）
  → 行为策略 = 目标策略 = ε-greedy(Q)
  → On-Policy
```

**后果**：SARSA 学到的是 ε-greedy 策略下的 Q 值，而不是纯贪心策略的 Q 值。这意味着它学到的策略会考虑探索带来的风险。

### 2.4 完整算法

```
初始化：Q(s, a) = 0  ∀s, a（终止状态 Q = 0）

对每个 episode：
    S ← 起始状态
    A ← ε-greedy(Q, S)

    重复（每一步）：
        执行 A，观察 R, S'

        如果 S' 是终止状态：
            Q(S, A) ← Q(S, A) + α [R - Q(S, A)]
            跳出循环
        否则：
            A' ← ε-greedy(Q, S')
            Q(S, A) ← Q(S, A) + α [R + γQ(S', A') - Q(S, A)]
            S ← S'
            A ← A'
```

### 2.5 ε-greedy 动作选择

```python
def epsilon_greedy(Q, state, n_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)  # 探索
    else:
        return np.argmax(Q[state])            # 利用
```

---

## 3. SARSA 的性质

### 3.1 收敛性

**定理**：在满足以下条件时，SARSA 收敛到 Q*：
1. GLIE 条件：ε → 0（最终变成贪心），且每个 (s,a) 被无限次访问
2. Robbins-Monro 条件：Σ α_t = ∞，Σ α_t² < ∞

**实践中**：通常用固定 ε 和固定 α，不严格满足收敛条件，但效果足够好。

### 3.2 On-Policy 的特点

**优点**：
- 学到的策略考虑了探索的风险
- 更安全：在有危险的环境中，SARSA 会学会避开悬崖边缘

**缺点**：
- 学到的不是真正的最优策略（因为 ε > 0 时总有随机性）
- 数据效率较低（不能用旧策略的数据）

### 3.3 SARSA vs MC 控制

| | MC 控制 | SARSA |
|---|---|---|
| 更新时机 | episode 结束后 | 每一步 |
| 偏差 | 无偏 | 有偏（自举） |
| 方差 | 高 | 低 |
| 连续任务 | 不适用 | 适用 |
| 实现复杂度 | 简单 | 简单 |

---

## 4. SARSA(λ)：加入资格迹

### 4.1 动机

和 TD(0) → TD(λ) 的推广一样，SARSA 也可以加入资格迹：

- **SARSA(0)**：标准 SARSA，只看一步
- **SARSA(λ)**：综合多步信息，λ 控制 MC 和 TD 的混合程度

### 4.2 算法

```
初始化：Q(s, a) = 0  ∀s, a

对每个 episode：
    e(s, a) = 0  ∀s, a        # 重置资格迹
    S ← 起始状态
    A ← ε-greedy(Q, S)

    重复（每一步）：
        执行 A，观察 R, S'
        A' ← ε-greedy(Q, S')

        δ ← R + γQ(S', A') - Q(S, A)     # TD 误差
        e(S, A) ← e(S, A) + 1            # 累积迹

        对所有 (s, a)：
            Q(s, a) ← Q(s, a) + α δ e(s, a)
            e(s, a) ← γ λ e(s, a)

        S ← S', A ← A'

        如果 S 是终止状态：跳出
```

### 4.3 资格迹的效果

```
Episode: S0,A0 → S1,A1 → S2,A2 → S3(终止), R=+1

SARSA(0)：只更新 Q(S2, A2)           ← 最近的一步
SARSA(λ)：更新 Q(S2,A2), Q(S1,A1), Q(S0,A0)  ← 按 λ 衰减
SARSA(1)：等价于 MC，所有步都更新       ← 全部
```

信用分配更快：奖励信号能"回传"给之前的 (s, a) 对。

---

## 5. CliffWalking 环境

### 5.1 环境描述

CliffWalking 是展示 On-Policy vs Off-Policy 差异的经典环境：

```
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│    │    │    │    │    │    │    │    │    │    │    │    │  行0
├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │    │    │    │    │    │    │  行1
├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │    │    │    │    │    │    │  行2
├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
│ S  │CCCC│CCCC│CCCC│CCCC│CCCC│CCCC│CCCC│CCCC│CCCC│CCCC│ G  │  行3
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
 列0   列1  列2  ...                                列10 列11
```

- **S** = 起点 (3, 0)
- **G** = 终点 (3, 11)
- **C** = 悬崖 (3, 1) 到 (3, 10)
- **动作**：0=上, 1=右, 2=下, 3=左
- **奖励**：
  - 每步：-1
  - 掉入悬崖：-100，并回到起点
  - 到达终点：episode 结束
- **4×12 = 48 个状态**

### 5.2 为什么 CliffWalking 经典？

这个环境完美展示了 **SARSA（On-Policy）vs Q-Learning（Off-Policy）** 的差异：

- **SARSA** 考虑到 ε 探索可能掉下悬崖 → 学会走**安全路径**（远离悬崖边）
- **Q-Learning** 学的是纯贪心策略 → 学会走**最短路径**（贴着悬崖边）

```
SARSA 的路径（安全）：        Q-Learning 的路径（最优但危险）：
→ → → → → → → → → → → ↓     S · · · · · · · · · · ↓
↑                       ↓     ↑                       ↓
↑                       ↓     ↑ → → → → → → → → → → G
S · · · · · · · · · · G
```

### 5.3 Gymnasium API

```python
import gymnasium as gym
env = gym.make('CliffWalking-v0')

# 状态：0-47 的整数
# 动作：0=上, 1=右, 2=下, 3=左
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

---

## 6. 实现要点

### 6.1 Q 表的表示

对于表格型方法，Q 函数用 2D 数组表示：

```python
Q = np.zeros((n_states, n_actions))  # shape: (48, 4)
```

### 6.2 训练循环的结构

```python
for episode in range(num_episodes):
    state, _ = env.reset()
    action = epsilon_greedy(Q, state, n_actions, epsilon)

    done = False
    total_reward = 0

    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        next_action = epsilon_greedy(Q, next_state, n_actions, epsilon)

        # SARSA 更新
        td_target = reward + gamma * Q[next_state, next_action] * (1 - terminated)
        Q[state, action] += alpha * (td_target - Q[state, action])

        state = next_state
        action = next_action
```

### 6.3 终止状态的处理

当 `terminated = True` 时，下一状态是终止的：

```python
# 方法 1：显式检查
if terminated:
    td_target = reward
else:
    td_target = reward + gamma * Q[next_state, next_action]

# 方法 2：乘以 (1 - terminated)
td_target = reward + gamma * Q[next_state, next_action] * (1 - int(terminated))
```

---

## 7. 本课练习内容

你将实现：

1. **ε-greedy 动作选择**
2. **SARSA(0) 算法**
3. **SARSA(λ) 算法**（资格迹）
4. **训练曲线分析**：奖励随 episode 的变化
5. **策略可视化**：在 4×12 网格上显示学到的策略
6. **ε 衰减实验**：对比固定 ε 和衰减 ε 的效果

测试环境：CliffWalking-v0

---

## 8. 关键概念总结

| 概念 | 定义 | 意义 |
|------|------|------|
| SARSA | Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)] | On-Policy TD 控制 |
| On-Policy | 行为策略 = 目标策略 | 学到的策略考虑探索风险 |
| ε-greedy | 概率 ε 随机, 1-ε 贪心 | 平衡探索与利用 |
| SARSA(λ) | SARSA + 资格迹 | 加速信用分配 |
| 安全路径 | SARSA 偏好远离危险的路径 | On-Policy 的特点 |

---

## 9. 与前后课程的联系

| Lesson 06 | **Lesson 07** | Lesson 08 |
|-----------|---------------|-----------|
| TD 预测 | **SARSA (TD 控制)** | Q-Learning |
| 学 V(s) | 学 Q(s,a) | 学 Q(s,a) |
| 给定策略评估 | On-Policy 控制 | **Off-Policy 控制** |
| — | 安全路径 | 最优路径 |

**过渡**：Lesson 08 将学习 **Q-Learning**，它是 Off-Policy 的——用 ε-greedy 收集数据，但学习的是纯贪心策略的 Q 值。在 CliffWalking 上，Q-Learning 会学到贴着悬崖走的最短路径。两种算法的对比是 RL 中最经典的实验之一。

---

## 参考资料

- Sutton & Barto 第 6.4 节：SARSA: On-policy TD Control
- Sutton & Barto 第 12.7 节：Sarsa(λ)
- David Silver Lecture 5: Model-Free Control
- [CliffWalking 环境文档](https://gymnasium.farama.org/environments/toy_text/cliff_walking/)
