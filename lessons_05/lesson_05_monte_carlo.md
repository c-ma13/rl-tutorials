# Lesson 05: Monte Carlo 方法

## 本课目标

1. 理解 **Model-Free** 的核心思想：不再需要 P(s'|s,a)
2. 掌握 **MC 策略评估**（首次访问 vs 每次访问）
3. 理解**探索-利用困境** (Exploration-Exploitation Dilemma)
4. 实现 **MC 控制**（On-Policy + ε-greedy）
5. 在 Blackjack 环境上训练智能体

---

## 1. 从 Model-Based 到 Model-Free

### 1.1 动态规划的局限

Lesson 01-04 中，我们所有算法都依赖于 **完整的环境模型**：

```
需要已知：P(s'|s,a)  和  R(s,a,s')
```

**现实问题**：
- 围棋：状态数 ≈ 10^170，无法枚举 P
- 机器人控制：物理动力学太复杂，无法精确建模
- 推荐系统：用户行为随时间变化，P 不是固定的
- LLM 对齐：人类偏好不可建模为已知函数

**解决方案**：从**经验**（experience）中学习，不需要 P 和 R。

### 1.2 Model-Free 的核心思想

```
DP（Model-Based）：                MC（Model-Free）：
  已知 P(s'|s,a) 和 R               只需要 episode 样本
  精确计算期望                        用采样平均近似期望
  更新所有状态                        只更新访问过的状态
```

**数学上的等价性**：

DP 计算的是**期望**：
```
V^π(s) = E_π[R_{t+1} + γR_{t+2} + ... | S_t = s]
```

MC 用**样本均值**来近似这个期望：
```
V^π(s) ≈ (1/N) Σ_{i=1}^N G_i(s)
```

其中 G_i(s) 是第 i 次从状态 s 出发得到的**回报** (return)。

**大数定律保证**：当 N → ∞ 时，样本均值收敛到真实期望。

---

## 2. Episode 与回报

### 2.1 Episode（轨迹）

一个 episode 是从开始到终止的完整轨迹：

```
S_0, A_0, R_1, S_1, A_1, R_2, ..., S_{T-1}, A_{T-1}, R_T, S_T(终止)
```

**重要**：MC 方法**只适用于有限长度的 episode**（必须能终止）。

### 2.2 回报 (Return)

从时刻 t 开始的折扣回报：

```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... + γ^{T-t-1}R_T
```

**递推计算**（从后往前）：

```
G_T = 0
G_{T-1} = R_T
G_{T-2} = R_{T-1} + γG_{T-1}
...
G_t = R_{t+1} + γG_{t+1}
```

**代码实现**：
```python
def compute_returns(rewards, gamma):
    """从后往前计算每个时刻的回报"""
    T = len(rewards)
    G = np.zeros(T)
    G[T-1] = rewards[T-1]
    for t in range(T-2, -1, -1):
        G[t] = rewards[t] + gamma * G[t+1]
    return G
```

---

## 3. MC 策略评估

### 3.1 首次访问 MC (First-Visit MC)

对每个 episode，只在**第一次**访问到状态 s 时记录回报：

```
初始化：
    V(s) = 0  ∀s
    returns(s) = []  （空列表）

对每个 episode：
    生成轨迹 S_0, A_0, R_1, ..., S_T
    计算回报 G_0, G_1, ..., G_{T-1}
    
    对 t = 0, 1, ..., T-1：
        如果 S_t 是该 episode 中首次出现：
            returns(S_t).append(G_t)
            V(S_t) = mean(returns(S_t))
```

### 3.2 每次访问 MC (Every-Visit MC)

每次访问到状态 s 时都记录回报：

```
对 t = 0, 1, ..., T-1：
    returns(S_t).append(G_t)       # 无论是否首次出现
    V(S_t) = mean(returns(S_t))
```

### 3.3 两种方法的对比

| 特性 | 首次访问 MC | 每次访问 MC |
|------|-------------|-------------|
| 偏差 | 无偏估计 | 有偏（但渐近无偏） |
| 方差 | 较高 | 较低（更多样本） |
| 理论基础 | 大数定律直接适用 | 需要更复杂的分析 |
| 实现难度 | 需要检查首次访问 | 更简单 |
| 收敛速度 | 通常差不多 | 通常差不多 |

**实践中两者差别不大**，首次访问 MC 更常用（理论性质更好）。

### 3.4 增量更新

不需要存储所有回报再求均值，可以**增量更新**：

```
V(s) ← V(s) + α [G_t - V(s)]
```

其中 α = 1/N(s) 等价于求均值。

**固定步长**（α 为常数）的优势：
- 更重视最近的样本
- 适应非平稳环境
- 相当于指数加权移动平均

---

## 4. MC 策略评估的实际意义

### 4.1 不需要模型

**DP 策略评估**：
```
V(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R + γV(s')]
                          ^^^^^^
                          需要知道 P
```

**MC 策略评估**：
```
V(s) ≈ mean(采样得到的 G_t)
        只需要能和环境交互
```

### 4.2 可以评估部分状态

DP 必须对**所有**状态进行更新。MC 只需要更新**访问过的**状态。

如果我们只关心起始状态的价值，MC 可以更高效。

---

## 5. 从评估到控制：MC 控制

### 5.1 广义策略迭代 (Generalized Policy Iteration, GPI)

Lesson 03-04 中的策略迭代和价值迭代都遵循 GPI 框架：

```
策略评估：V^π ← 评估当前策略
策略改进：π ← greedy(V^π)
```

MC 控制也遵循 GPI，但用 MC 采样替代 DP 计算。

### 5.2 为什么需要 Q 而不是 V？

**Model-Based 策略改进**（有 P）：
```
π'(s) = argmax_a Σ_{s'} P(s'|s,a) [R + γV(s')]
```

**Model-Free 策略改进**（没有 P）：
```
π'(s) = argmax_a Q(s, a)      ← 不需要 P！
```

**关键洞察**：在 model-free 设定下，必须用 **Q 函数**而不是 V 函数。因为没有 P，我们无法从 V 计算出 "选哪个动作更好"。

### 5.3 探索-利用困境

**问题**：如果策略 π 是确定性的（总是选同一个动作），那么很多 (s, a) 对永远不会被访问到，Q(s, a) 就无法更新。

```
例：状态 s 下有两个动作 a=0 和 a=1
如果 π(s) = 0，则 a=1 永远不被选择
→ Q(s, 1) 永远不被更新
→ 即使 a=1 实际上更好，也永远不会发现
```

**这就是探索-利用困境 (Exploration-Exploitation Dilemma)**：
- **利用 (Exploitation)**：选择当前已知最好的动作
- **探索 (Exploration)**：尝试其他动作以发现更好的选择

### 5.4 ε-Greedy 策略

最简单的解决方案：以概率 ε 随机探索，以概率 1-ε 利用当前最优：

```
           ┌ 随机动作（均匀选择）  概率 ε
π(a|s) = ┤
           └ argmax_a Q(s,a)      概率 1-ε
```

更精确地写：

```
π(a|s) = ε/|A| + (1-ε) · I(a = argmax_{a'} Q(s, a'))
```

**ε 的选择**：
- ε = 1.0：完全随机（纯探索）
- ε = 0.0：完全贪心（纯利用）
- ε = 0.1：常用默认值
- 递减 ε：初期多探索，后期多利用

### 5.5 探索性起始 (Exploring Starts) vs ε-Greedy

**方法一：探索性起始**
- 保证每个 (s, a) 对都有非零概率被选为起始状态
- 理论上完美，但实践中常常不可行

**方法二：ε-Greedy（On-Policy）**
- 在执行策略中加入随机性
- 简单实用，但永远不会完全收敛到确定性最优策略

**方法三：Off-Policy（Lesson 08 会详细学）**
- 用一个策略收集数据，用另一个策略学习
- 重要性采样 (Importance Sampling)

---

## 6. On-Policy MC 控制算法

### 6.1 完整算法

```
初始化：
    Q(s, a) = 任意值  ∀s, a
    N(s, a) = 0       ∀s, a
    π = ε-greedy(Q)

重复（对每个 episode）：
    1. 用 π 生成 episode: S_0, A_0, R_1, ..., S_T
    
    2. 从后往前计算回报: G ← 0
       对 t = T-1, T-2, ..., 0：
           G ← γG + R_{t+1}
           
           如果是首次访问 (S_t, A_t)：
               N(S_t, A_t) += 1
               Q(S_t, A_t) += (1/N(S_t, A_t)) [G - Q(S_t, A_t)]
    
    3. 更新 π 为 ε-greedy(Q)
```

### 6.2 GLIE (Greedy in the Limit with Infinite Exploration)

为了保证收敛到最优策略，需要：
1. 每个 (s, a) 对被无限次访问
2. 策略最终收敛到贪心策略

实现方式：令 ε 随时间递减：
```
ε_k = 1/k     或     ε_k = max(ε_min, ε_0 × decay^k)
```

---

## 7. Blackjack 环境

### 7.1 游戏规则

Blackjack（21 点）是测试 MC 方法的经典环境。

**基本规则**：
- 目标：手牌点数尽可能接近 21 但不超过
- 2-10 按面值，J/Q/K 值 10，A 可以值 1 或 11
- **玩家动作**：叫牌 (Hit, 要一张) 或停牌 (Stick, 不再要)
- **庄家策略**：≤16 必须叫牌，≥17 必须停牌
- 超过 21 点**爆牌** (Bust)，直接输

### 7.2 状态空间

状态是一个三元组：
```
(player_sum, dealer_showing, usable_ace)
```

- `player_sum`：玩家当前点数 (12-21)
- `dealer_showing`：庄家明牌 (1-10, 其中 1=A)
- `usable_ace`：玩家是否有可用的 A (True/False)

**状态总数**：10 × 10 × 2 = 200

### 7.3 动作空间

```
0 = Stick（停牌）
1 = Hit（叫牌）
```

### 7.4 奖励

```
+1：玩家赢
 0：平局
-1：玩家输
```

### 7.5 为什么 Blackjack 适合 MC？

1. **Episode 很短**：通常 2-5 步就结束
2. **不需要折扣**：γ = 1.0（自然终止）
3. **Model-Free 动机**：转移概率取决于整副牌，不好精确建模
4. **经典教科书例子**：Sutton & Barto 第 5 章

### 7.6 Gymnasium API

```python
import gymnasium as gym

env = gym.make('Blackjack-v1', sab=True)  # sab=True 使用 Sutton & Barto 版本

obs, info = env.reset()      # obs = (player_sum, dealer_showing, usable_ace)
obs, reward, terminated, truncated, info = env.step(action)
```

---

## 8. 本课练习内容

你将实现：

1. **MC 策略评估**（首次访问）：评估 Blackjack 的固定策略
2. **回报计算**：从后往前递推
3. **On-Policy MC 控制**（ε-greedy）：训练智能体玩 Blackjack
4. **Q 值可视化**：用 3D 曲面图展示学到的 Q 函数
5. **ε 衰减实验**：对比不同 ε 衰减策略的效果

---

## 9. MC 方法的优缺点总结

### 优点
1. **不需要模型**：只需要能与环境交互
2. **无偏估计**：（首次访问 MC）
3. **概念简单**：直接用采样均值估计期望
4. **适合 episode 任务**：自然地处理有终止条件的问题
5. **可以聚焦感兴趣的状态**：不需要全局更新

### 缺点
1. **高方差**：单次回报的方差可能很大
2. **必须等 episode 结束**：不能在线更新（和 TD 对比）
3. **收敛慢**：需要大量 episode
4. **不适用于连续任务**：没有终止状态怎么办？
5. **探索问题**：需要额外机制保证充分探索

---

## 10. 关键概念总结

| 概念 | 定义 | 意义 |
|------|------|------|
| Model-Free | 不需要 P(s'\|s,a) 和 R | 适用于真实世界问题 |
| Episode | 从开始到终止的完整轨迹 | MC 方法的基本单位 |
| Return G_t | 从时刻 t 的折扣累积奖励 | MC 用 G_t 估计 V(S_t) |
| 首次访问 MC | 每个 episode 中只在首次访问时更新 | 无偏估计 |
| ε-Greedy | 概率 ε 随机 + 概率 1-ε 贪心 | 平衡探索与利用 |
| GPI | 评估-改进交替迭代 | MC 控制的理论框架 |
| GLIE | ε→0 同时保证无限探索 | 收敛到最优的充分条件 |

---

## 11. 与前后课程的联系

| Lesson 04 | **Lesson 05** | Lesson 06 |
|-----------|---------------|-----------|
| 价值迭代（DP） | Monte Carlo | TD 学习 |
| 需要模型 P, R | **不需要模型** | 不需要模型 |
| 精确计算期望 | 采样 + 等 episode 结束 | 采样 + 单步更新 |
| 更新所有状态 | 只更新访问过的状态 | 只更新当前状态 |
| O(\|S\|²\|A\|) | O(episode 长度) | O(1) 每步 |

**过渡**：MC 是第一个 model-free 方法，但它必须等 episode 结束才能更新。Lesson 06 的 TD 学习将解决这个限制——每走一步就可以更新。

---

## 参考资料

- Sutton & Barto 第 5 章：Monte Carlo Methods
- David Silver Lecture 4: Model-Free Prediction
- David Silver Lecture 5: Model-Free Control
- [Blackjack 环境文档](https://gymnasium.farama.org/environments/toy_text/blackjack/)
