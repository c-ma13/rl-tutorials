# Lesson 10: 策略梯度与 REINFORCE

## 本课目标

1. 理解**策略梯度**的核心思想
2. 掌握 **REINFORCE** 算法
3. 理解**基线**（Baseline）的作用
4. 实现 REINFORCE 解决 CartPole
5. 理解策略梯度的**高方差**问题
6. 为 Actor-Critic 打下基础

---

## 1. 从价值方法到策略方法

### 1.1 两种范式

到目前为止（Lesson 01-09），我们学的都是**价值方法**（Value-Based）：

```
价值方法：
    学习 V(s) 或 Q(s, a)
    策略隐式：π(s) = argmax_a Q(s, a)

策略方法：
    直接学习策略 π(a|s; θ)
    价值函数可选（Actor-Critic 中会用）
```

### 1.2 为什么需要策略方法？

**价值方法的局限**：

1. **离散动作空间**：argmax 只适用于有限动作
   ```
   机器人控制：动作 = 关节角度（连续）
   价值方法：无法 argmax over 连续空间
   策略方法：π(a|s) = N(μ(s), σ²) ← 直接输出连续动作
   ```

2. **确定性策略**：argmax 总是选同一个动作
   ```
   石头剪刀布：最优策略是随机 1/3
   价值方法：Q(s, 石头) = Q(s, 剪刀) = Q(s, 布) → 无法表达
   策略方法：π(石头|s) = π(剪刀|s) = π(布|s) = 1/3 ← 直接表达
   ```

3. **策略平滑性**：小的参数变化 → 小的策略变化
   ```
   价值方法：Q 的微小变化可能导致 argmax 突变
   策略方法：θ 的微小变化 → π 平滑变化
   ```

### 1.3 策略参数化

用参数 θ 表示策略：

```
离散动作：π(a|s; θ) = softmax(f(s, a; θ))
连续动作：π(a|s; θ) = N(μ(s; θ), σ²(s; θ))
```

**例子**（线性策略）：
```python
# 离散动作
logits = θ^T φ(s)  # shape: (n_actions,)
π(a|s; θ) = exp(logits[a]) / Σ_a' exp(logits[a'])

# 连续动作
μ = θ_μ^T φ(s)
σ = exp(θ_σ^T φ(s))
π(a|s; θ) = (1/√(2πσ²)) exp(-(a-μ)²/(2σ²))
```

---

## 2. 策略梯度定理

### 2.1 目标函数

目标：最大化期望回报

```
J(θ) = E_τ~π_θ [G_0]
     = E_τ~π_θ [Σ_{t=0}^T γ^t R_{t+1}]
```

其中 τ = (S_0, A_0, R_1, S_1, ...) 是轨迹。

### 2.2 策略梯度定理

**定理**（Sutton et al., 2000）：

```
∇_θ J(θ) = E_τ~π_θ [Σ_{t=0}^T ∇_θ log π(A_t|S_t; θ) G_t]
```

**直觉**：
- `∇_θ log π(A_t|S_t; θ)`：增加选择 A_t 的概率的方向
- `G_t`：如果回报高，就增加这个动作的概率；如果回报低，就减少

**推导思路**（简化版）：
```
J(θ) = E_τ [G_0]
∇_θ J(θ) = ∇_θ E_τ [G_0]
         = E_τ [G_0 ∇_θ log p(τ|θ)]    ← log 导数技巧
         = E_τ [G_0 Σ_t ∇_θ log π(A_t|S_t; θ)]  ← 轨迹概率分解
```

### 2.3 为什么不需要环境模型？

关键：`∇_θ log π(A_t|S_t; θ)` 只依赖于策略，不依赖于环境的转移概率 P(s'|s,a)。

```
价值方法：需要 P 来计算 Bellman 方程（或用 TD 近似）
策略方法：只需要能采样轨迹，不需要知道 P
```

---

## 3. REINFORCE 算法

### 3.1 基本算法

```
初始化：策略参数 θ

对每个 episode：
    1. 用 π(·|·; θ) 生成轨迹 τ = (S_0, A_0, R_1, ..., S_T)
    
    2. 对每个时刻 t = 0, 1, ..., T-1：
        计算回报 G_t = Σ_{k=t}^{T-1} γ^{k-t} R_{k+1}
    
    3. 梯度更新：
        θ ← θ + α Σ_{t=0}^{T-1} ∇_θ log π(A_t|S_t; θ) G_t
```

### 3.2 实现细节

**对数概率梯度**（离散动作）：
```python
# Softmax 策略
logits = model(state)  # shape: (n_actions,)
probs = softmax(logits)
log_prob = log(probs[action])

# 梯度（PyTorch 自动计算）
loss = -log_prob * G_t
loss.backward()
```

**对数概率梯度**（连续动作）：
```python
# 高斯策略
mu, sigma = model(state)
log_prob = -0.5 * ((action - mu) / sigma)**2 - log(sigma) - 0.5*log(2π)

loss = -log_prob * G_t
loss.backward()
```

### 3.3 为什么叫 REINFORCE？

**REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility**

简单理解：用回报 G_t 作为"强化信号"，增加好动作的概率，减少坏动作的概率。

---

## 4. 基线 (Baseline)

### 4.1 高方差问题

REINFORCE 的主要问题：**高方差**

```
例：两个 episode
    Episode 1: G_0 = 100
    Episode 2: G_0 = 90

梯度估计：
    ∇J ≈ ∇log π(A_0|S_0) × 100  （第一次）
    ∇J ≈ ∇log π(A_0|S_0) × 90   （第二次）

问题：即使都是好的 episode，梯度方向也不稳定
```

### 4.2 引入基线

减去一个基线 b(s)，不改变期望但降低方差：

```
∇_θ J(θ) = E [∇_θ log π(A_t|S_t; θ) (G_t - b(S_t))]
```

**为什么不改变期望？**
```
E [∇_θ log π(A_t|S_t; θ) b(S_t)]
= Σ_a π(a|s) ∇_θ log π(a|s) b(s)
= Σ_a ∇_θ π(a|s) b(s)
= ∇_θ (Σ_a π(a|s)) b(s)
= ∇_θ (1) b(s)
= 0
```

### 4.3 常见基线

**1. 常数基线**
```
b = 平均回报
```
简单但不够好。

**2. 状态价值基线**
```
b(s) = V(s)
```
最优基线（理论上）。需要额外学习 V(s)。

**3. 移动平均基线**
```
b ← β b + (1-β) G_t
```
简单实用。

### 4.4 REINFORCE with Baseline

```
初始化：策略参数 θ, 基线参数 w

对每个 episode：
    1. 生成轨迹 τ
    
    2. 对每个时刻 t：
        G_t = Σ_{k=t}^{T-1} γ^{k-t} R_{k+1}
        δ_t = G_t - V(S_t; w)           # 优势
        
        θ ← θ + α_θ δ_t ∇_θ log π(A_t|S_t; θ)
        w ← w + α_w δ_t ∇_w V(S_t; w)   # 更新基线
```

---

## 5. 策略梯度的性质

### 5.1 优点

1. **收敛性好**：保证收敛到局部最优（至少）
2. **连续动作**：自然处理连续动作空间
3. **随机策略**：可以学习随机策略
4. **策略平滑**：参数小变化 → 策略小变化

### 5.2 缺点

1. **高方差**：需要大量样本
2. **样本效率低**：On-Policy，不能重用旧数据
3. **收敛慢**：相比价值方法
4. **局部最优**：可能陷入局部最优

### 5.3 与价值方法的对比

| | 价值方法 | 策略方法 |
|---|---|---|
| 学习对象 | Q(s, a) | π(a\|s; θ) |
| 动作空间 | 离散 | 离散 + 连续 |
| 策略类型 | 确定性 | 随机 |
| 样本效率 | 高（Off-Policy） | 低（On-Policy） |
| 方差 | 低 | 高 |
| 收敛性 | 可能不收敛 | 保证收敛 |

---

## 6. CartPole 环境

### 6.1 环境描述

CartPole（倒立摆）是测试策略梯度的经典环境：

```
状态：(x, ẋ, θ, θ̇)
    x: 小车位置
    ẋ: 小车速度
    θ: 杆子角度
    θ̇: 杆子角速度

动作：0=左推, 1=右推

目标：保持杆子直立

奖励：每步 +1（杆子未倒）

终止条件：
    - 杆子倾斜 > 12°
    - 小车超出边界
    - 步数 > 500
```

### 6.2 为什么用 CartPole？

1. **连续状态**：4 维连续状态
2. **离散动作**：2 个动作（简单）
3. **快速训练**：episode 短，易于调试
4. **经典 benchmark**：RL 入门标准环境

### 6.3 成功标准

```
连续 100 个 episode 平均奖励 ≥ 195
```

---

## 7. 本课练习内容

你将实现：

1. **策略网络**：
   - 线性策略（Softmax）
   - 神经网络策略（可选）

2. **REINFORCE 算法**：
   - 基本版本（无基线）
   - 带基线版本

3. **基线网络**：学习 V(s)

4. **训练和评估**：
   - 在 CartPole 上训练
   - 学习曲线可视化
   - 对比有无基线的效果

5. **超参数实验**：
   - 学习率
   - 折扣因子
   - 基线系数

测试环境：CartPole-v1

---

## 8. 关键概念总结

| 概念 | 定义 | 意义 |
|------|------|------|
| 策略梯度 | ∇_θ J(θ) = E[∇_θ log π G_t] | 直接优化策略 |
| REINFORCE | 用完整回报 G_t 更新 | Monte Carlo 策略梯度 |
| 基线 | G_t - b(S_t) | 降低方差 |
| log 导数技巧 | ∇_θ p = p ∇_θ log p | 推导策略梯度 |
| 高方差 | 回报的随机性大 | 需要大量样本 |
| On-Policy | 用当前策略采样 | 样本效率低 |

---

## 9. 与前后课程的联系

| Lesson 09 | **Lesson 10** | Lesson 11 |
|-----------|---------------|-----------|
| 函数逼近（价值） | **策略梯度** | Actor-Critic |
| 学习 Q(s,a) | 学习 π(a\|s) | 学习 π 和 V |
| 离散动作 | 离散 + 连续 | 离散 + 连续 |
| 低方差 | 高方差 | 中方差 |

**过渡**：
- REINFORCE 的高方差问题 → Lesson 11 的 Actor-Critic 用 TD 误差替代 G_t
- 基线 V(s) → Actor-Critic 的 Critic
- 策略 π(a|s) → Actor-Critic 的 Actor

---

## 10. 现代策略梯度方法预告

REINFORCE 是最基础的策略梯度方法，现代方法在此基础上改进：

**PPO (Proximal Policy Optimization, 2017)**：
```
限制策略更新幅度，避免破坏性更新
→ 最流行的策略梯度方法
```

**TRPO (Trust Region Policy Optimization, 2015)**：
```
用二阶优化保证单调改进
→ 理论保证强，但计算复杂
```

**A3C (Asynchronous Advantage Actor-Critic, 2016)**：
```
多线程并行训练
→ 提高样本效率
```

这些方法都基于本课的策略梯度定理。

---

## 参考资料

- Sutton & Barto 第 13 章：Policy Gradient Methods
- Williams (1992) "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (REINFORCE 原论文)
- Sutton et al. (2000) "Policy Gradient Methods for Reinforcement Learning with Function Approximation"
- David Silver Lecture 7: Policy Gradient Methods
- [CartPole 环境文档](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
