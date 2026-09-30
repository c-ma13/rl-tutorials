# Lesson 15: PPO (上) - Proximal Policy Optimization

## 1. 从 Value-Based 到 Policy-Based

### 1.1 回顾：我们学过的算法

到目前为止，我们学习了两大类强化学习算法：

**Value-Based 方法（Lesson 7-14）：**
- Q-Learning, SARSA, DQN, Double DQN, Dueling DQN
- 学习价值函数 $Q(s, a)$ 或 $V(s)$
- 策略是隐式的：$\pi(s) = \arg\max_a Q(s, a)$

**Policy-Based 方法（Lesson 10-11）：**
- REINFORCE, Actor-Critic, A2C
- 直接学习策略 $\pi(s, a; \theta)$
- 使用策略梯度优化

### 1.2 Policy-Based 方法的优势

**为什么需要 Policy-Based 方法？**

1. **可以处理连续动作空间**
   - Value-based 需要 $\arg\max_a Q(s, a)$，连续空间无法枚举
   - Policy-based 直接输出动作分布

2. **可以学习随机策略**
   - 某些问题需要随机策略（如石头剪刀布）
   - Value-based 通常是确定性策略

3. **更好的收敛性**
   - 策略梯度在某些情况下收敛性更好
   - 避免了 value-based 的高估偏差问题

4. **更适合高维动作空间**
   - 机器人控制、游戏 AI 等

### 1.3 Policy-Based 方法的挑战

**但是，Policy-Based 方法也有问题：**

1. **样本效率低**
   - REINFORCE 需要大量样本
   - 方差大，学习慢

2. **训练不稳定**
   - 策略更新步长难以控制
   - 容易陷入局部最优或发散

3. **对超参数敏感**
   - 学习率、折扣因子等需要仔细调节

**PPO 的目标：解决这些问题，成为实践中最好用的策略梯度算法！**


## 2. 信赖域方法（Trust Region Methods）

### 2.1 策略梯度的问题

**回顾策略梯度（Lesson 10）：**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) A^{\pi_\theta}(s_t, a_t) \right]$$

**标准梯度下降更新：**

$$\theta_{new} = \theta_{old} + \alpha \nabla_\theta J(\theta)$$

**问题：步长 $\alpha$ 难以选择！**

- **太小**：学习慢，需要大量样本
- **太大**：策略变化剧烈，性能崩溃，难以恢复

**核心矛盾：**
- 我们用旧策略 $\pi_{\theta_{old}}$ 采样数据
- 但用这些数据更新到新策略 $\pi_{\theta_{new}}$
- 如果新旧策略差异太大，数据就不再适用（off-policy 问题）

### 2.2 信赖域的思想

**核心思想：限制策略更新的幅度**

在优化目标的同时，约束新旧策略的距离：

$$\max_\theta \mathbb{E}_{s,a \sim \pi_{\theta_{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A^{\pi_{\theta_{old}}}(s, a) \right]$$

$$\text{subject to: } D_{KL}(\pi_{\theta_{old}} || \pi_\theta) \leq \delta$$

其中：
- $D_{KL}$：KL 散度，衡量两个分布的距离
- $\delta$：信赖域半径，控制策略变化幅度

**直观理解：**
- 在"信赖域"内优化策略（保证数据有效性）
- 类似于"小步快跑"，每次更新不要太激进

### 2.3 重要性采样（Importance Sampling）

**问题：如何用旧策略的数据评估新策略？**

**重要性采样比率（Importance Sampling Ratio）：**

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

**目标函数改写：**

$$L^{CPI}(\theta) = \mathbb{E}_{s,a \sim \pi_{\theta_{old}}} \left[ r_t(\theta) A^{\pi_{\theta_{old}}}(s, a) \right]$$

其中 CPI = Conservative Policy Iteration。

**理解：**
- $r_t(\theta) > 1$：新策略更倾向于选择该动作
- $r_t(\theta) < 1$：新策略更不倾向于选择该动作
- $r_t(\theta) \approx 1$：新旧策略相似

**问题：$r_t(\theta)$ 可能非常大或非常小，导致估计方差爆炸！**


## 3. TRPO (Trust Region Policy Optimization)

### 3.1 TRPO 的目标函数

**TRPO（Schulman et al., 2015）是第一个成功的信赖域策略优化算法：**

$$\max_\theta \mathbb{E}_{s,a \sim \pi_{\theta_{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A^{\pi_{\theta_{old}}}(s, a) \right]$$

$$\text{subject to: } \mathbb{E}_{s \sim \pi_{\theta_{old}}} [D_{KL}(\pi_{\theta_{old}}(\cdot|s) || \pi_\theta(\cdot|s))] \leq \delta$$

**关键点：**
1. 目标函数：最大化优势函数的加权和
2. 约束条件：KL 散度不超过 $\delta$（通常 $\delta = 0.01$）

### 3.2 TRPO 的优化方法

**TRPO 使用二阶优化方法求解约束优化问题：**

1. **线性化目标函数**（一阶泰勒展开）
2. **二次化约束**（二阶泰勒展开）
3. **共轭梯度法**求解
4. **线搜索**确保约束满足

**数学细节（了解即可）：**

目标函数的梯度：

$$g = \nabla_\theta \mathbb{E}_{s,a} [r_t(\theta) A(s, a)]|_{\theta=\theta_{old}}$$

KL 散度的 Hessian 矩阵（Fisher Information Matrix）：

$$H = \nabla_\theta^2 D_{KL}(\pi_{\theta_{old}} || \pi_\theta)|_{\theta=\theta_{old}}$$

更新方向：

$$\Delta \theta = \sqrt{\frac{2\delta}{g^T H^{-1} g}} H^{-1} g$$

### 3.3 TRPO 的优缺点

**优点：**
- ✅ 单调性改进保证（理论上每次更新都不会变差）
- ✅ 训练稳定
- ✅ 对超参数不敏感

**缺点：**
- ❌ 实现复杂（共轭梯度、线搜索）
- ❌ 计算开销大（二阶优化）
- ❌ 难以与其他技巧结合（如 dropout、batch normalization）

**PPO 的目标：保留 TRPO 的优点，简化实现！**


## 4. PPO-Clip：简化的信赖域方法

### 4.1 PPO 的核心思想

**PPO（Schulman et al., 2017）的创新：用裁剪代替 KL 约束**

**不再使用显式的 KL 约束，而是直接裁剪重要性采样比率！**

### 4.2 PPO-Clip 目标函数

**定义重要性采样比率：**

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

**PPO-Clip 目标函数：**

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]$$

其中：
- $\epsilon$：裁剪参数（通常 $\epsilon = 0.2$）
- $\text{clip}(x, a, b)$：将 $x$ 裁剪到 $[a, b]$ 范围内

**展开写：**

$$\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) = \begin{cases}
1 - \epsilon & \text{if } r_t(\theta) < 1 - \epsilon \\
r_t(\theta) & \text{if } 1 - \epsilon \leq r_t(\theta) \leq 1 + \epsilon \\
1 + \epsilon & \text{if } r_t(\theta) > 1 + \epsilon
\end{cases}$$

### 4.3 PPO-Clip 的直观理解

**分情况讨论：**

**情况 1：优势 $A_t > 0$（好动作）**

- 目标：增大 $\pi_\theta(a_t | s_t)$，即增大 $r_t(\theta)$
- 但是：$r_t(\theta)$ 被裁剪到 $[1-\epsilon, 1+\epsilon]$
- 结果：$r_t(\theta)$ 最多增大到 $1+\epsilon$，防止过度优化

**情况 2：优势 $A_t < 0$（坏动作）**

- 目标：减小 $\pi_\theta(a_t | s_t)$，即减小 $r_t(\theta)$
- 但是：$r_t(\theta)$ 被裁剪到 $[1-\epsilon, 1+\epsilon]$
- 结果：$r_t(\theta)$ 最多减小到 $1-\epsilon$，防止过度惩罚

**核心：限制策略更新幅度，保持在信赖域内！**

### 4.4 PPO-Clip 的可视化

**目标函数的形状：**

```
L^CLIP(r)

A > 0 (好动作):
    |
    |     ___________  (裁剪后：斜率为 0)
    |    /
    |   /  (未裁剪：斜率为 A)
    |  /
    |_/_______________|_____
      1-ε    1    1+ε      r

A < 0 (坏动作):
    |
    |  \
    |   \  (未裁剪：斜率为 A)
    |    \___________  (裁剪后：斜率为 0)
    |
    |_________________|_____
      1-ε    1    1+ε      r
```

**关键观察：**
- 在 $[1-\epsilon, 1+\epsilon]$ 范围内，目标函数是线性的
- 超出范围后，目标函数变平（梯度为 0），停止更新

### 4.5 PPO-Clip vs TRPO

| 特性 | TRPO | PPO-Clip |
|------|------|----------|
| 约束方式 | KL 散度约束 | 裁剪重要性采样比率 |
| 优化方法 | 二阶优化（共轭梯度） | 一阶优化（SGD/Adam） |
| 实现复杂度 | 高 | 低 |
| 计算开销 | 大 | 小 |
| 性能 | 优秀 | 优秀（略逊于 TRPO） |
| 实践中使用 | 较少 | 非常广泛 |


## 5. PPO 算法完整流程

### 5.1 PPO-Clip 伪代码

```
初始化：
  - 策略网络 π(a|s; θ)
  - 价值网络 V(s; φ)
  - 优化器（Adam）
  - 超参数：ε, γ, λ (GAE), K (更新轮数), batch_size

对于每个 iteration：
  1. 收集数据：
     用当前策略 π_θ_old 采样 N 步轨迹
     存储 (s_t, a_t, r_t, s_{t+1}, log π_θ_old(a_t|s_t))
  
  2. 计算优势函数：
     用 GAE 计算 A_t (Lesson 11)
     计算回报 R_t = A_t + V(s_t)
  
  3. 更新策略（K 轮）：
     对于每个 mini-batch：
       a) 计算重要性采样比率：
          r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
       
       b) 计算 PPO-Clip 损失：
          L^CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
       
       c) 计算价值函数损失：
          L^VF = E[(V(s_t) - R_t)²]
       
       d) 计算熵正则化（可选）：
          L^ENT = -E[H(π_θ(·|s_t))]
       
       e) 总损失：
          L = -L^CLIP + c_1 * L^VF - c_2 * L^ENT
       
       f) 梯度下降更新 θ 和 φ
  
  4. θ_old ← θ
```

### 5.2 关键超参数

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| ε (clip) | 0.1 - 0.2 | 裁剪范围，控制策略更新幅度 |
| γ (gamma) | 0.99 | 折扣因子 |
| λ (lambda) | 0.95 | GAE 参数 |
| K (epochs) | 3 - 10 | 每次数据重复使用次数 |
| N (steps) | 2048 - 4096 | 每次采样步数 |
| batch_size | 64 - 256 | Mini-batch 大小 |
| learning_rate | 3e-4 | 学习率 |
| c_1 (value coef) | 0.5 | 价值函数损失系数 |
| c_2 (entropy coef) | 0.01 | 熵正则化系数 |

### 5.3 PPO 的技巧

**1. 优势函数归一化**

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

- 稳定训练
- 减少方差

**2. 价值函数裁剪（可选）**

```python
value_pred_clipped = value_old + torch.clamp(value - value_old, -clip_range, clip_range)
value_loss = torch.max((value - returns)**2, (value_pred_clipped - returns)**2).mean()
```

- 防止价值函数更新过大
- 进一步稳定训练

**3. 学习率衰减（可选）**

```python
lr = lr_init * (1 - current_step / total_steps)
```

- 后期更精细的调整
- 提高最终性能

**4. 梯度裁剪**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm=0.5)
```

- 防止梯度爆炸
- 稳定训练


## 6. PPO 的理论分析

### 6.1 单调性改进

**TRPO 的理论保证：**

在 KL 约束下，策略单调改进（每次更新都不会变差）。

**PPO 的理论保证：**

PPO 没有严格的单调性保证，但实践中表现优异。

**原因：**
- 裁剪机制隐式地限制了策略变化
- 多轮更新（K epochs）进一步稳定训练

### 6.2 样本效率

**PPO 是 on-policy 算法，但可以重复使用数据：**

- 每次采样 N 步数据
- 用这些数据更新 K 轮
- 总样本效率：比 REINFORCE 高，比 DQN 低

**为什么可以重复使用？**

- 裁剪机制保证新旧策略不会差异太大
- 数据仍然"近似"来自当前策略

### 6.3 收敛性

**PPO 的收敛性：**

- 没有严格的收敛性证明
- 实践中收敛性良好
- 对超参数相对鲁棒


## 7. PPO 的成功案例

### 7.1 OpenAI Five (Dota 2)

**2018 年，OpenAI 用 PPO 训练 AI 击败人类职业选手：**

- 5v5 团队游戏
- 连续动作空间（移动、技能释放）
- 长期规划（游戏时长 30-45 分钟）
- 训练规模：128,000 CPU 核心，256 GPU，10 个月

**关键技术：**
- PPO 作为核心算法
- 大规模分布式训练
- 课程学习（从简单到复杂）

### 7.2 ChatGPT / InstructGPT (RLHF)

**2022 年，OpenAI 用 PPO 对齐大语言模型：**

- RLHF (Reinforcement Learning from Human Feedback)
- 奖励模型：人类偏好
- PPO 优化语言模型策略

**为什么选择 PPO？**
- 训练稳定
- 实现简单
- 对超参数不敏感

### 7.3 其他应用

- **机器人控制**：MuJoCo 连续控制任务
- **游戏 AI**：Atari、StarCraft II
- **自动驾驶**：决策规划
- **推荐系统**：用户交互优化


## 8. PPO vs 其他算法

| 算法 | 类型 | 样本效率 | 训练稳定性 | 实现复杂度 | 适用场景 |
|------|------|----------|-----------|-----------|----------|
| DQN | Value-based | 高 | 中 | 中 | 离散动作 |
| A2C | Policy-based | 低 | 中 | 低 | 通用 |
| TRPO | Policy-based | 中 | 高 | 高 | 通用 |
| PPO | Policy-based | 中 | 高 | 低 | 通用（推荐） |
| SAC | Off-policy | 高 | 高 | 中 | 连续动作 |

**PPO 的定位：**
- 训练稳定性和实现简单性的最佳平衡
- 实践中的首选算法（尤其是连续控制）


## 9. 实现要点

### 9.1 网络结构

**策略网络（连续动作）：**

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        x = self.fc(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std
```

**价值网络：**

```python
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        return self.fc(state)
```

### 9.2 PPO-Clip 损失计算

```python
def compute_ppo_loss(states, actions, old_log_probs, advantages, returns, 
                     policy_net, value_net, clip_epsilon=0.2):
    # 计算新策略的 log 概率
    mean, std = policy_net(states)
    dist = torch.distributions.Normal(mean, std)
    new_log_probs = dist.log_prob(actions).sum(dim=-1)
    
    # 重要性采样比率
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # PPO-Clip 损失
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 价值函数损失
    values = value_net(states).squeeze()
    value_loss = (values - returns).pow(2).mean()
    
    # 熵正则化
    entropy = dist.entropy().sum(dim=-1).mean()
    
    return policy_loss, value_loss, entropy
```


## 10. 总结

### 10.1 核心要点

1. **信赖域方法**：
   - 限制策略更新幅度
   - 保证训练稳定性

2. **TRPO**：
   - 使用 KL 散度约束
   - 二阶优化，实现复杂

3. **PPO-Clip**：
   - 用裁剪代替 KL 约束
   - 一阶优化，实现简单
   - 性能优异，广泛应用

4. **PPO 的优势**：
   - 训练稳定
   - 实现简单
   - 对超参数鲁棒
   - 适用于连续和离散动作空间

### 10.2 下节预告

下节课我们将学习 **PPO (下)**：
- PPO 实现细节
- 超参数调优
- 在连续控制任务上的应用
- PPO 的变体和改进

### 10.3 扩展阅读

**必读论文：**
1. Schulman et al. (2015). "Trust Region Policy Optimization" (TRPO)
2. Schulman et al. (2017). "Proximal Policy Optimization Algorithms" (PPO)

**推荐资源：**
- Spinning Up in Deep RL: PPO
- OpenAI Baselines: PPO2
- CleanRL PPO 实现
