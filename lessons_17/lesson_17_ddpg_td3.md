# Lesson 17: DDPG & TD3

## 1. 从随机策略到确定性策略

### 1.1 回顾：策略梯度方法的分类

到目前为止，我们学习了两类策略梯度方法：

**随机策略（Stochastic Policy）：**
- REINFORCE、A2C、PPO
- 输出动作的概率分布：$\pi_\theta(a|s)$
- 采样动作：$a \sim \pi_\theta(\cdot|s)$

**确定性策略（Deterministic Policy）：**
- 输出确定的动作：$\mu_\theta(s)$
- 不需要采样，直接输出动作

### 1.2 为什么需要确定性策略？

**问题：高维连续动作空间中，随机策略的方差很大**

随机策略梯度（Stochastic Policy Gradient）：

$$\nabla_\theta J = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s, a)]$$

- 需要对**状态和动作**进行采样
- 动作空间维度越高，方差越大
- 需要大量样本

确定性策略梯度（Deterministic Policy Gradient, DPG）：

$$\nabla_\theta J = \mathbb{E}_{s \sim \rho^\mu} [\nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s, a)|_{a=\mu_\theta(s)}]$$

- 只需要对**状态**进行采样
- 不需要对动作采样
- 方差更低，样本效率更高

### 1.3 确定性策略梯度定理

**Silver et al., 2014 证明了确定性策略梯度定理：**

$$\nabla_\theta J(\mu_\theta) = \mathbb{E}_{s \sim \rho^\mu} [\nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s, a)|_{a=\mu_\theta(s)}]$$

**直观理解：**
1. $\nabla_a Q^\mu(s, a)$：Q 值关于动作的梯度（"动作应该怎么调整？"）
2. $\nabla_\theta \mu_\theta(s)$：策略参数关于输出的梯度（"参数应该怎么调整？"）
3. 链式法则：通过 Q 函数的梯度来更新策略

**与随机策略梯度的关系：**

确定性策略梯度是随机策略梯度的极限情况（当策略方差趋近于 0）。


## 2. DDPG (Deep Deterministic Policy Gradient)

### 2.1 核心思想

**DDPG = DPG + Deep Neural Networks + DQN 技巧**

DDPG 将 DQN 的成功技巧（经验回放 + 目标网络）与确定性策略梯度结合：

- **Actor（策略网络）**：$\mu_\theta(s)$ → 输出确定性动作
- **Critic（Q 网络）**：$Q_\phi(s, a)$ → 评估动作价值
- **经验回放**：打破样本相关性
- **目标网络**：稳定训练

### 2.2 DDPG 算法

```
初始化：
  - Actor 网络 μ_θ(s)
  - Critic 网络 Q_φ(s, a)
  - 目标 Actor 网络 μ_θ'(s) ← θ
  - 目标 Critic 网络 Q_φ'(s, a) ← φ
  - 回放缓冲区 D

对于每个 episode：
  对于每个时间步 t：
    1. 选择动作：
       a = μ_θ(s) + noise   (添加探索噪声)
    
    2. 执行动作 a，观察 r, s'
    
    3. 存储 (s, a, r, s', done) 到 D
    
    4. 从 D 采样 mini-batch
    
    5. 更新 Critic：
       y = r + γ Q_φ'(s', μ_θ'(s'))
       L_critic = E[(y - Q_φ(s, a))²]
       φ ← φ - α_φ ∇_φ L_critic
    
    6. 更新 Actor：
       L_actor = -E[Q_φ(s, μ_θ(s))]
       θ ← θ - α_θ ∇_θ L_actor
    
    7. 软更新目标网络：
       θ' ← τθ + (1-τ)θ'
       φ' ← τφ + (1-τ)φ'
```

### 2.3 DDPG 的关键技术

**1. 探索噪声**

确定性策略本身不具备探索能力，需要添加噪声：

$$a = \mu_\theta(s) + \mathcal{N}$$

常用噪声：
- **Ornstein-Uhlenbeck (OU) 噪声**：时间相关噪声，适合物理控制
- **高斯噪声**：简单，$\mathcal{N} \sim N(0, \sigma^2)$

OU 噪声过程：

$$dx_t = \theta_{OU} (\mu_{OU} - x_t) dt + \sigma_{OU} dW_t$$

**2. 软更新（Soft Update / Polyak Averaging）**

与 DQN 的硬更新不同，DDPG 使用软更新：

$$\theta' \leftarrow \tau \theta + (1 - \tau) \theta'$$

其中 $\tau$ 通常很小（如 0.005）。

**优势：**
- 目标网络缓慢变化，更稳定
- 避免硬更新带来的突然变化

**3. Batch Normalization（可选）**

- 归一化网络各层的输入
- 减少不同环境之间奖励尺度的影响
- 原论文使用，但后续研究发现不是必需的

### 2.4 DDPG 的问题

**1. 高估偏差（Overestimation Bias）**

与 DQN 类似，DDPG 也存在 Q 值高估问题：

$$y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))$$

Critic 的估计误差通过目标网络传播，导致系统性高估。

**2. 训练不稳定**

- 对超参数敏感
- Critic 和 Actor 的更新互相影响
- 可能出现性能突然崩溃

**3. 探索不足**

- 高斯噪声可能不够
- 在复杂环境中探索效率低


## 3. TD3 (Twin Delayed DDPG)

### 3.1 TD3 的三个改进

**Fujimoto et al., 2018 提出了三个关键改进：**

1. **Twin Critics（双 Critic）**：减少高估偏差
2. **Delayed Policy Update（延迟策略更新）**：更稳定的训练
3. **Target Policy Smoothing（目标策略平滑）**：减少方差

### 3.2 Twin Critics

**核心思想：使用两个 Critic 网络，取较小的 Q 值**

$$y = r + \gamma \min_{i=1,2} Q_{\phi_i'}(s', \mu_{\theta'}(s'))$$

**为什么有效？**

- 两个独立的 Critic 分别估计 Q 值
- 取最小值减少了高估偏差
- 类似于 Double DQN 的思想，但更激进

**实现：**

```python
# 两个 Critic 网络
critic1 = CriticNetwork(state_dim, action_dim)
critic2 = CriticNetwork(state_dim, action_dim)

# 目标 Q 值
with torch.no_grad():
    target_q1 = target_critic1(next_states, target_actions)
    target_q2 = target_critic2(next_states, target_actions)
    target_q = torch.min(target_q1, target_q2)
    target = rewards + gamma * target_q * (1 - dones)
```

### 3.3 Delayed Policy Update

**核心思想：Critic 更新频率高于 Actor**

- Critic 每步都更新
- Actor 每 $d$ 步更新一次（通常 $d = 2$）
- 目标网络也每 $d$ 步更新

**为什么有效？**

- Critic 需要先学好，才能给 Actor 提供有意义的梯度
- 避免 Actor 在 Critic 不准确时更新
- 类似于 Actor-Critic 的"两个时间尺度"思想

### 3.4 Target Policy Smoothing

**核心思想：给目标动作添加噪声**

$$a'_{target} = \mu_{\theta'}(s') + \text{clip}(\epsilon, -c, c), \quad \epsilon \sim N(0, \sigma)$$

**为什么有效？**

- 相当于对 Q 值进行正则化
- 使 Q 函数在动作空间更平滑
- 减少 Critic 的过拟合

**参数：**
- $\sigma = 0.2$：噪声标准差
- $c = 0.5$：裁剪范围

### 3.5 TD3 完整算法

```
初始化：
  - Actor 网络 μ_θ(s)
  - 两个 Critic 网络 Q_φ₁(s,a), Q_φ₂(s,a)
  - 目标网络 μ_θ'(s), Q_φ₁'(s,a), Q_φ₂'(s,a)
  - 回放缓冲区 D

对于每个时间步 t：
  1. 选择动作：a = μ_θ(s) + ε, ε ~ N(0, σ_explore)
  
  2. 执行动作 a，存储 (s, a, r, s', done) 到 D
  
  3. 从 D 采样 mini-batch
  
  4. 计算目标动作（带噪声）：
     ã = μ_θ'(s') + clip(ε, -c, c), ε ~ N(0, σ_target)
     ã = clip(ã, a_low, a_high)
  
  5. 计算目标 Q 值：
     y = r + γ min(Q_φ₁'(s', ã), Q_φ₂'(s', ã))
  
  6. 更新两个 Critic：
     L_i = E[(y - Q_φᵢ(s, a))²], i = 1, 2
  
  7. 每隔 d 步：
     a) 更新 Actor：
        L_actor = -E[Q_φ₁(s, μ_θ(s))]
     
     b) 软更新目标网络：
        θ' ← τθ + (1-τ)θ'
        φ₁' ← τφ₁ + (1-τ)φ₁'
        φ₂' ← τφ₂ + (1-τ)φ₂'
```


## 4. DDPG vs TD3 对比

| 特性 | DDPG | TD3 |
|------|------|-----|
| Critic 数量 | 1 | 2（取最小值） |
| Actor 更新频率 | 每步 | 每 d 步 |
| 目标策略噪声 | 无 | 有（平滑化） |
| 高估偏差 | 严重 | 显著减少 |
| 训练稳定性 | 中 | 高 |
| 性能 | 中 | 高 |


## 5. 网络结构设计

### 5.1 Actor 网络

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # tanh 将输出限制到 [-1, 1]，再缩放到动作范围
        return self.max_action * torch.tanh(self.fc3(x))
```

**关键点：**
- 输出层使用 `tanh` 激活（限制范围）
- 乘以 `max_action` 缩放到实际范围
- 隐藏层使用 `ReLU`

### 5.2 Critic 网络

```python
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 状态和动作一起输入
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

**关键点：**
- 输入是**状态和动作的拼接**
- 输出单个 Q 值
- 不使用输出激活函数


## 6. 关键超参数

| 超参数 | DDPG 典型值 | TD3 典型值 | 说明 |
|--------|-------------|------------|------|
| Actor LR | 1e-4 | 3e-4 | Actor 学习率 |
| Critic LR | 1e-3 | 3e-4 | Critic 学习率 |
| γ | 0.99 | 0.99 | 折扣因子 |
| τ | 0.005 | 0.005 | 软更新系数 |
| Buffer Size | 1e6 | 1e6 | 回放缓冲区大小 |
| Batch Size | 64-256 | 256 | 采样批次大小 |
| 探索噪声 σ | 0.1 | 0.1 | 探索噪声标准差 |
| 目标噪声 σ | — | 0.2 | 目标策略平滑噪声 |
| 噪声裁剪 c | — | 0.5 | 目标噪声裁剪范围 |
| 延迟更新 d | — | 2 | Actor 更新频率 |
| 预热步数 | 10000 | 25000 | 随机探索步数 |


## 7. 总结

### 7.1 核心要点

1. **确定性策略梯度**：
   - 直接输出确定性动作
   - 通过 Q 函数的梯度更新策略
   - 方差低于随机策略

2. **DDPG**：
   - DPG + DQN 技巧
   - 经验回放 + 目标网络 + 软更新
   - 需要额外的探索噪声

3. **TD3 的三个改进**：
   - Twin Critics：减少高估偏差
   - Delayed Update：更稳定的训练
   - Target Smoothing：更平滑的 Q 函数

4. **实践建议**：
   - 优先使用 TD3（比 DDPG 更稳定）
   - 先实现 DDPG，再添加 TD3 的改进
   - 调试时监控 Q 值、Actor 损失、Critic 损失

### 7.2 与其他算法的对比

| 算法 | 类型 | 动作空间 | 样本效率 | 稳定性 |
|------|------|----------|----------|--------|
| PPO | On-policy | 离散+连续 | 中 | 高 |
| DDPG | Off-policy | 连续 | 高 | 中 |
| TD3 | Off-policy | 连续 | 高 | 高 |
| SAC | Off-policy | 连续 | 高 | 高 |

### 7.3 下节预告

下节课我们将学习 **SAC (Soft Actor-Critic)**：
- 最大熵 RL 框架
- 温度参数自动调节
- SAC 与 TD3 的对比

### 7.4 扩展阅读

**必读论文：**
1. Silver et al. (2014). "Deterministic Policy Gradient Algorithms"
2. Lillicrap et al. (2016). "Continuous control with deep reinforcement learning" (DDPG)
3. Fujimoto et al. (2018). "Addressing Function Approximation Error in Actor-Critic Methods" (TD3)

**推荐资源：**
- Spinning Up in Deep RL: DDPG, TD3
- CleanRL 实现：https://github.com/vwxyzjn/cleanrl
