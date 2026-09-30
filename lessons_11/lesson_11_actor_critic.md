# Lesson 11: Actor-Critic 方法

## 本课目标

1. 理解 **Actor-Critic** 架构
2. 掌握 **优势函数** (Advantage Function)
3. 实现 **One-Step Actor-Critic**
4. 理解 **A2C** (Advantage Actor-Critic)
5. 实现 **A3C** 的单线程版本
6. 为现代策略梯度方法（PPO、SAC）打下基础

---

## 1. 从 REINFORCE 到 Actor-Critic

### 1.1 REINFORCE 的问题

回顾 REINFORCE (Lesson 10)：

```
θ ← θ + α G_t ∇_θ log π(A_t|S_t; θ)
```

**问题**：
1. **高方差**：G_t 是完整回报，方差很大
2. **必须等 episode 结束**：不能在线更新
3. **样本效率低**：每个样本只用一次

### 1.2 Actor-Critic 的思想

**核心**：用 TD 误差替代回报 G_t

```
REINFORCE:      θ ← θ + α G_t ∇_θ log π(A_t|S_t)
                         ‾‾‾
                         高方差

Actor-Critic:   θ ← θ + α δ_t ∇_θ log π(A_t|S_t)
                         ‾‾‾
                         低方差
```

其中 TD 误差：
```
δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)
```

### 1.3 为什么叫 Actor-Critic？

```
Actor（演员）：策略 π(a|s; θ)
    - 选择动作
    - 根据 Critic 的反馈更新

Critic（评论家）：价值函数 V(s; w)
    - 评估状态的好坏
    - 提供 TD 误差给 Actor
```

**类比**：
- Actor = 演员在舞台上表演
- Critic = 评委打分并给出改进建议

---

## 2. 优势函数 (Advantage Function)

### 2.1 定义

优势函数衡量"选择动作 a 比平均水平好多少"：

```
A^π(s, a) = Q^π(s, a) - V^π(s)
```

**直觉**：
- `Q^π(s, a)`：选择动作 a 的价值
- `V^π(s)`：在状态 s 的平均价值
- `A^π(s, a)`：选择 a 的相对优势

### 2.2 为什么用优势？

**对比**：

| 信号 | 含义 | 问题 |
|------|------|------|
| G_t | 完整回报 | 高方差 |
| Q(s, a) | 动作价值 | 需要学习 Q |
| V(s) | 状态价值 | 无法区分动作 |
| **A(s, a)** | **相对优势** | **低方差 + 区分动作** |

**例子**：
```
状态 s: 所有动作都很好
    Q(s, a1) = 10, Q(s, a2) = 9, V(s) = 9.5
    A(s, a1) = 0.5, A(s, a2) = -0.5

用 Q 更新：两个动作都增加概率（都是正的）
用 A 更新：a1 增加，a2 减少（相对比较）
```

### 2.3 TD 误差作为优势的估计

**关键洞察**：TD 误差是优势的无偏估计

```
δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)
    ≈ Q(S_t, A_t) - V(S_t)
    = A(S_t, A_t)
```

**推导**：
```
Q(s, a) = E[R + γ V(S') | s, a]
δ = R + γ V(S') - V(s)
E[δ | s, a] = E[R + γ V(S') | s, a] - V(s) = Q(s, a) - V(s) = A(s, a)
```

因此可以用 δ_t 替代 A(s, a)！

---

## 3. One-Step Actor-Critic

### 3.1 算法

```
初始化：策略参数 θ, 价值参数 w

对每个 episode：
    S ← 起始状态
    
    重复（每一步）：
        A ← π(·|S; θ)                    # Actor 选动作
        执行 A，观察 R, S'
        
        δ ← R + γ V(S'; w) - V(S; w)     # Critic 计算 TD 误差
        
        w ← w + α_w δ ∇_w V(S; w)        # 更新 Critic
        θ ← θ + α_θ δ ∇_θ log π(A|S; θ)  # 更新 Actor
        
        S ← S'
```

### 3.2 与 REINFORCE 的对比

| | REINFORCE | Actor-Critic |
|---|---|---|
| 更新信号 | G_t（完整回报） | δ_t（TD 误差） |
| 方差 | 高 | 低 |
| 偏差 | 无偏 | 有偏（自举） |
| 更新时机 | episode 结束 | 每一步 |
| 需要 Critic | 可选（基线） | 必须 |

### 3.3 收敛性

**定理**（Konda & Tsitsiklis, 2000）：
在适当条件下（两个时间尺度、Robbins-Monro 条件），Actor-Critic 收敛到局部最优。

**两个时间尺度**：
- Critic 更新快（α_w 大）：快速学习 V
- Actor 更新慢（α_θ 小）：稳定策略改进

---

## 4. A2C (Advantage Actor-Critic)

### 4.1 多步优势

One-Step Actor-Critic 用 1-step TD 误差：
```
δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)
```

可以推广到 n-step：
```
A_t^(n) = Σ_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n V(S_{t+n}) - V(S_t)
```

**特殊情况**：
- n=1：One-Step Actor-Critic
- n=∞：REINFORCE（G_t - V(S_t)）

### 4.2 A2C 算法

A2C 是 A3C 的同步版本（单线程）：

```
初始化：θ, w

对每个 episode：
    收集 n 步轨迹：(S_0, A_0, R_1, ..., S_n)
    
    对每个时刻 t = 0, ..., n-1：
        计算 n-step 回报：
            G_t = Σ_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n V(S_{t+n}; w)
        
        优势：A_t = G_t - V(S_t; w)
        
        更新 Critic：w ← w + α_w A_t ∇_w V(S_t; w)
        更新 Actor：θ ← θ + α_θ A_t ∇_θ log π(A_t|S_t; θ)
```

### 4.3 熵正则化

为了鼓励探索，A2C 加入熵奖励：

```
L_actor = -E[A_t log π(A_t|S_t)] - β H(π(·|S_t))
```

其中熵：
```
H(π) = -Σ_a π(a|s) log π(a|s)
```

**效果**：
- β > 0：鼓励随机策略（探索）
- β = 0：标准 Actor-Critic

---

## 5. A3C (Asynchronous Advantage Actor-Critic)

### 5.1 核心思想

A3C 用多个并行 worker 异步更新全局参数：

```
全局参数：θ_global, w_global

每个 worker：
    1. 复制全局参数：θ ← θ_global, w ← w_global
    2. 收集 n 步经验
    3. 计算梯度
    4. 异步更新全局参数：
        θ_global ← θ_global + ∇θ
        w_global ← w_global + ∇w
    5. 重复
```

### 5.2 优势

1. **样本效率**：多个 worker 并行收集数据
2. **探索多样性**：不同 worker 探索不同策略
3. **稳定性**：异步更新减少相关性
4. **不需要经验回放**：多样性来自并行

### 5.3 单线程版本

本课实现单线程版本（等价于 A2C）：

```python
# 伪代码
for episode in range(num_episodes):
    trajectory = collect_trajectory(n_steps)
    advantages = compute_advantages(trajectory)
    update_actor(advantages)
    update_critic(advantages)
```

---

## 6. 实现细节

### 6.1 优势估计

**GAE (Generalized Advantage Estimation)**：

```
A_t^GAE(λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
```

其中 δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)

**递推计算**（从后往前）：
```python
advantages = []
gae = 0
for t in reversed(range(T)):
    delta = rewards[t] + gamma * V[t+1] - V[t]
    gae = delta + gamma * lam * gae
    advantages.insert(0, gae)
```

### 6.2 价值目标

Critic 的目标是 n-step 回报：
```
V_target = Σ_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n V(S_{t+n})
```

或者用 GAE：
```
V_target = A_t + V(S_t)
```

### 6.3 梯度裁剪

防止梯度爆炸：
```python
grad_norm = np.linalg.norm(grad)
if grad_norm > max_grad_norm:
    grad = grad * (max_grad_norm / grad_norm)
```

---

## 7. 本课练习内容

你将实现：

1. **One-Step Actor-Critic**：
   - Actor 网络（策略）
   - Critic 网络（价值）
   - TD 误差计算

2. **A2C 算法**：
   - n-step 回报
   - 优势估计
   - 熵正则化

3. **GAE (Generalized Advantage Estimation)**

4. **训练和评估**：
   - 在 CartPole 上训练
   - 学习曲线可视化
   - 对比 REINFORCE、Actor-Critic、A2C

5. **超参数实验**：
   - n-step 数量
   - GAE λ
   - 熵系数 β

测试环境：CartPole-v1

---

## 8. 关键概念总结

| 概念 | 定义 | 意义 |
|------|------|------|
| Actor-Critic | Actor 选动作，Critic 评估 | 结合策略梯度和价值方法 |
| 优势函数 | A(s,a) = Q(s,a) - V(s) | 相对优势，低方差 |
| TD 误差 | δ = R + γV(S') - V(S) | 优势的无偏估计 |
| A2C | n-step + 优势 + 熵 | 同步版 A3C |
| A3C | 多线程异步更新 | 样本效率高 |
| GAE | 指数加权 TD 误差 | 平衡偏差-方差 |

---

## 9. 与前后课程的联系

| Lesson 10 | **Lesson 11** | Lesson 12 |
|-----------|---------------|-----------|
| REINFORCE | **Actor-Critic** | 探索策略 |
| G_t（高方差） | δ_t（低方差） | ε-greedy, UCB |
| 必须等 episode 结束 | 每步更新 | 平衡探索-利用 |
| 无 Critic | 有 Critic | 应用到各种方法 |

**过渡**：
- Actor-Critic 是现代 RL 的基础
- PPO (2017)：Actor-Critic + 信任域
- SAC (2018)：Actor-Critic + 最大熵
- TD3 (2018)：Actor-Critic + Double Q

---

## 10. 现代方法预览

### 10.1 PPO (Proximal Policy Optimization)

```
限制策略更新幅度：
    L = min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)
其中 r_t = π_new(a|s) / π_old(a|s)
```

### 10.2 SAC (Soft Actor-Critic)

```
最大熵目标：
    J = E[Σ R_t + α H(π(·|S_t))]
自动调节温度 α
```

### 10.3 TD3 (Twin Delayed DDPG)

```
Actor-Critic + 连续动作：
    - Double Q-Learning（两个 Critic）
    - 延迟策略更新
    - 目标策略平滑
```

这些方法都基于 Actor-Critic 架构。

---

## 参考资料

- Sutton & Barto 第 13.5 节：Actor-Critic Methods
- Konda & Tsitsiklis (2000) "Actor-Critic Algorithms"
- Mnih et al. (2016) "Asynchronous Methods for Deep Reinforcement Learning" (A3C 原论文)
- Schulman et al. (2016) "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (GAE 原论文)
- David Silver Lecture 7: Policy Gradient Methods
