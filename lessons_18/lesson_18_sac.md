# Lesson 18: SAC (Soft Actor-Critic)

## 1. 最大熵强化学习

### 1.1 传统 RL 的目标

传统强化学习的目标是最大化累积奖励：

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^T \gamma^t r(s_t, a_t) \right]$$

**问题：**
- 策略可能过早收敛到次优解
- 探索不足
- 对奖励函数敏感

### 1.2 最大熵 RL 的目标

**核心思想：在最大化奖励的同时，最大化策略的熵**

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^T \gamma^t \left( r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right) \right]$$

其中：
- $\mathcal{H}(\pi(\cdot|s)) = -\mathbb{E}_{a \sim \pi} [\log \pi(a|s)]$：策略熵
- $\alpha$：温度参数，控制熵的权重

**熵的直观理解：**
- 熵越大，策略越随机（不确定性越高）
- 熵越小，策略越确定

**最大熵 RL 的优势：**

1. **更好的探索**：
   - 鼓励策略保持随机性
   - 自动探索多种可能的行为

2. **更鲁棒**：
   - 对奖励函数的扰动更鲁棒
   - 学到的策略更多样化

3. **更容易微调**：
   - 学到的策略可以作为其他任务的初始化
   - 保留了多种可能的行为模式

4. **避免过早收敛**：
   - 不会过早地陷入确定性策略
   - 持续探索更好的解

### 1.3 最大熵 Bellman 方程

**软 Q 函数（Soft Q-function）：**

$$Q^{\text{soft}}(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim p} [V^{\text{soft}}(s')]$$

**软状态价值函数（Soft Value function）：**

$$V^{\text{soft}}(s) = \mathbb{E}_{a \sim \pi} [Q^{\text{soft}}(s, a) - \alpha \log \pi(a|s)]$$

**最优策略：**

$$\pi^*(a|s) = \exp\left(\frac{1}{\alpha}(Q^{\text{soft}}(s, a) - V^{\text{soft}}(s))\right)$$


## 2. SAC 算法

### 2.1 SAC 的核心思想

**SAC (Soft Actor-Critic, Haarnoja et al., 2018) 是最大熵 RL 的实现：**

- **Actor**：学习随机策略 $\pi_\theta(a|s)$（高斯策略）
- **Critic**：学习软 Q 函数 $Q_\phi(s, a)$
- **自动调节温度参数** $\alpha$

**与 TD3 的对比：**

| 特性 | TD3 | SAC |
|------|-----|-----|
| 策略类型 | 确定性 | 随机（高斯） |
| 探索方式 | 添加噪声 | 策略本身随机 |
| 熵正则化 | 无 | 有（最大熵） |
| 温度参数 | 无 | 自动调节 |
| 样本效率 | 高 | 高 |
| 训练稳定性 | 高 | 高 |

### 2.2 SAC 的网络结构

**1. Actor 网络（高斯策略）**

输出动作的均值和标准差：

```python
class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action
    
    def forward(self, state):
        x = self.fc(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制范围
        return mean, log_std
```

**2. Critic 网络（Twin Q-networks）**

与 TD3 相同，使用两个 Q 网络：

```python
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.fc(x)
```

### 2.3 重参数化技巧（Reparameterization Trick）

**问题：如何从策略中采样动作并计算梯度？**

朴素方法：$a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$

- 采样操作不可微
- 无法反向传播

**重参数化技巧：**

$$a = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

- 随机性来自 $\epsilon$（与 $\theta$ 无关）
- 可以对 $\mu_\theta$ 和 $\sigma_\theta$ 求梯度

**Squashed Gaussian（压缩高斯）：**

为了将动作限制到有界范围，使用 tanh 变换：

$$a = \tanh(\tilde{a}), \quad \tilde{a} \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$$

**对数概率的修正：**

$$\log \pi(a|s) = \log \pi(\tilde{a}|s) - \sum_{i=1}^d \log(1 - \tanh^2(\tilde{a}_i))$$

### 2.4 SAC 的损失函数

**1. Critic 损失（与 TD3 类似）**

$$L_Q(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( Q_\phi(s, a) - y \right)^2 \right]$$

其中目标值：

$$y = r + \gamma \left( \min_{i=1,2} Q_{\phi_i'}(s', a') - \alpha \log \pi_\theta(a'|s') \right)$$

注意：目标值中**减去了熵项** $\alpha \log \pi_\theta(a'|s')$

**2. Actor 损失**

$$L_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta} \left[ \alpha \log \pi_\theta(a|s) - Q_\phi(s, a) \right]$$

**直观理解：**
- 第一项：最大化熵（鼓励探索）
- 第二项：最大化 Q 值（最大化奖励）
- 两者的平衡由 $\alpha$ 控制

**3. 温度参数损失（自动调节）**

$$L_\alpha = \mathbb{E}_{a \sim \pi_\theta} \left[ -\alpha \left( \log \pi_\theta(a|s) + \bar{\mathcal{H}} \right) \right]$$

其中 $\bar{\mathcal{H}}$ 是目标熵（通常设为 $-\dim(\mathcal{A})$）。

**直观理解：**
- 如果当前熵 $< \bar{\mathcal{H}}$，增大 $\alpha$（鼓励更多探索）
- 如果当前熵 $> \bar{\mathcal{H}}$，减小 $\alpha$（减少探索）

### 2.5 SAC 完整算法

```
初始化：
  - Actor 网络 π_θ(a|s)
  - 两个 Critic 网络 Q_φ₁(s,a), Q_φ₂(s,a)
  - 目标 Critic 网络 Q_φ₁'(s,a), Q_φ₂'(s,a)
  - 温度参数 α（或 log α）
  - 回放缓冲区 D

对于每个时间步 t：
  1. 选择动作：
     a ~ π_θ(·|s)  (从策略采样)
  
  2. 执行动作 a，存储 (s, a, r, s', done) 到 D
  
  3. 从 D 采样 mini-batch
  
  4. 更新 Critic：
     a) 采样下一个动作：a' ~ π_θ(·|s')
     b) 计算目标 Q 值：
        y = r + γ (min(Q_φ₁'(s', a'), Q_φ₂'(s', a')) - α log π_θ(a'|s'))
     c) 更新两个 Critic：
        L_Q = E[(Q_φᵢ(s, a) - y)²], i = 1, 2
  
  5. 更新 Actor：
     a) 采样动作：a ~ π_θ(·|s)
     b) 计算损失：
        L_π = E[α log π_θ(a|s) - Q_φ₁(s, a)]
     c) 梯度下降更新 θ
  
  6. 更新温度参数 α：
     L_α = E[-α (log π_θ(a|s) + H̄)]
  
  7. 软更新目标网络：
     φᵢ' ← τφᵢ + (1-τ)φᵢ', i = 1, 2
```


## 3. SAC 的关键技术

### 3.1 自动温度调节

**为什么需要自动调节？**

- 不同任务的最优温度不同
- 训练过程中最优温度可能变化
- 手动调节温度很困难

**实现：**

将 $\alpha$ 视为可学习参数（实际上学习 $\log \alpha$）：

```python
self.log_alpha = torch.zeros(1, requires_grad=True)
self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

# 更新 alpha
alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
self.alpha_optimizer.zero_grad()
alpha_loss.backward()
self.alpha_optimizer.step()

self.alpha = self.log_alpha.exp()
```

**目标熵的选择：**

通常设为 $\bar{\mathcal{H}} = -\dim(\mathcal{A})$

- 对于 1 维动作：$\bar{\mathcal{H}} = -1$
- 对于 3 维动作：$\bar{\mathcal{H}} = -3$

### 3.2 Squashed Gaussian 策略

**标准高斯策略的问题：**
- 动作可能超出环境限制
- 需要手动裁剪

**Squashed Gaussian 的优势：**
- 自动将动作限制到 $[-1, 1]$
- 保持可微性
- 更稳定的训练

**实现：**

```python
def sample_action(self, state):
    mean, log_std = self.forward(state)
    std = log_std.exp()
    
    # 重参数化采样
    normal = torch.distributions.Normal(mean, std)
    x = normal.rsample()  # rsample 支持梯度
    
    # tanh 压缩
    action = torch.tanh(x)
    
    # 计算 log 概率（修正）
    log_prob = normal.log_prob(x)
    log_prob -= torch.log(1 - action.pow(2) + 1e-6)
    log_prob = log_prob.sum(dim=-1, keepdim=True)
    
    return action, log_prob
```

### 3.3 Twin Q-networks

与 TD3 相同，SAC 也使用两个 Q 网络：

- 减少高估偏差
- 取最小值作为目标

### 3.4 不使用目标策略平滑

**与 TD3 的区别：**

- TD3：给目标动作添加噪声
- SAC：不需要，因为策略本身就是随机的

SAC 的随机策略天然提供了平滑效果。


## 4. SAC vs TD3 详细对比

| 特性 | TD3 | SAC |
|------|-----|-----|
| **策略类型** | 确定性 $\mu_\theta(s)$ | 随机 $\pi_\theta(a\|s)$ |
| **探索方式** | 添加高斯噪声 | 策略本身随机 |
| **熵正则化** | 无 | 有（最大熵目标） |
| **温度参数** | 无 | 自动调节 $\alpha$ |
| **目标策略平滑** | 有（添加噪声） | 无（策略已随机） |
| **延迟更新** | 有（Actor 每 d 步更新） | 无（每步都更新） |
| **样本效率** | 高 | 高 |
| **训练稳定性** | 高 | 高 |
| **超参数敏感性** | 中 | 低 |
| **适用场景** | 连续控制 | 连续控制 |

**性能对比：**

- **样本效率**：SAC 通常略优于 TD3
- **最终性能**：两者相当，取决于任务
- **稳定性**：SAC 更稳定，对超参数更鲁棒
- **实现复杂度**：SAC 稍复杂（需要处理随机策略和温度调节）


## 5. 关键超参数

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| Actor LR | 3e-4 | Actor 学习率 |
| Critic LR | 3e-4 | Critic 学习率 |
| Alpha LR | 3e-4 | 温度参数学习率 |
| γ | 0.99 | 折扣因子 |
| τ | 0.005 | 软更新系数 |
| Buffer Size | 1e6 | 回放缓冲区大小 |
| Batch Size | 256 | 采样批次大小 |
| 目标熵 | $-\dim(\mathcal{A})$ | 自动温度调节的目标 |
| 初始 α | 0.2 | 初始温度（如果不自动调节） |


## 6. SAC 的优势与局限

### 6.1 优势

1. **样本效率高**：
   - Off-policy 算法
   - 可以重复使用经验

2. **训练稳定**：
   - 最大熵目标提供正则化
   - 对超参数不敏感

3. **探索能力强**：
   - 策略本身随机
   - 自动平衡探索与利用

4. **鲁棒性好**：
   - 对奖励函数扰动鲁棒
   - 学到的策略更多样化

5. **易于调试**：
   - 超参数少
   - 温度自动调节

### 6.2 局限

1. **仅适用于连续动作空间**：
   - 离散动作需要特殊处理
   - 不如 DQN/PPO 通用

2. **计算开销较大**：
   - 需要采样动作并计算 log 概率
   - 比确定性策略慢

3. **需要大量样本**：
   - 虽然样本效率高，但仍需数十万步
   - 不如 model-based 方法


## 7. 实践建议

### 7.1 何时使用 SAC

**推荐使用 SAC 的场景：**
- 连续控制任务
- 需要高样本效率
- 需要稳定训练
- 探索很重要

**不推荐使用 SAC 的场景：**
- 离散动作空间（用 DQN/PPO）
- 样本获取成本极低（用 PPO）
- 需要确定性策略

### 7.2 调试技巧

1. **监控温度参数 α**：
   - α 应该逐渐稳定
   - 如果 α 持续增大或减小，检查目标熵设置

2. **监控策略熵**：
   - 熵应该接近目标熵
   - 如果熵过低，策略可能过早收敛

3. **监控 Q 值**：
   - Q 值应该逐渐增大
   - 如果 Q 值爆炸，检查学习率和奖励尺度

4. **对比 TD3**：
   - 如果 SAC 表现不如 TD3，检查温度调节
   - 尝试固定 α 或调整目标熵

### 7.3 常见问题

**问题 1：训练不收敛**
- 检查目标熵设置
- 尝试降低学习率
- 检查奖励尺度

**问题 2：策略过于随机**
- α 可能太大
- 降低目标熵（如 $-0.5 \times \dim(\mathcal{A})$）

**问题 3：策略过于确定**
- α 可能太小
- 增大目标熵
- 检查 log_std 的裁剪范围


## 8. 总结

### 8.1 核心要点

1. **最大熵 RL**：
   - 在最大化奖励的同时最大化熵
   - 鼓励探索，避免过早收敛
   - 学到的策略更鲁棒

2. **SAC 算法**：
   - 随机策略（高斯）
   - Twin Q-networks
   - 自动温度调节
   - Squashed Gaussian

3. **SAC vs TD3**：
   - SAC：随机策略，最大熵，自动探索
   - TD3：确定性策略，添加噪声探索
   - 两者性能相当，SAC 更稳定

4. **实践建议**：
   - 优先使用 SAC（连续控制）
   - 监控温度参数和策略熵
   - 对比 TD3 验证实现

### 8.2 SAC 的地位

**SAC 是当前连续控制任务的 SOTA 算法之一：**
- 样本效率高
- 训练稳定
- 对超参数鲁棒
- 广泛应用于机器人控制、自动驾驶等

### 8.3 下节预告

下节课我们将学习 **模型基础 RL 简介**：
- Dyna 架构
- 世界模型（World Model）
- MBPO
- DreamerV3 概览

### 8.4 扩展阅读

**必读论文：**
1. Haarnoja et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
2. Haarnoja et al. (2018). "Soft Actor-Critic Algorithms and Applications"

**推荐资源：**
- Spinning Up in Deep RL: SAC
- CleanRL SAC 实现
- Stable-Baselines3 SAC 文档
