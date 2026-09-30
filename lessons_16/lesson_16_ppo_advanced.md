# Lesson 16: PPO (下) - 实现细节与实践

## 1. 回顾：PPO 核心思想

在上节课中，我们学习了 PPO 的理论基础：

**核心公式：**

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$

**关键优势：**
- 简单易实现（一阶优化）
- 训练稳定（裁剪机制）
- 性能优异（接近 TRPO）

本节课将深入 PPO 的实现细节，学习如何在连续控制任务上应用 PPO。


## 2. 连续动作空间的 PPO

### 2.1 离散 vs 连续动作空间

**离散动作空间（如 CartPole）：**
- 动作：有限个离散选项（左、右）
- 策略：Categorical 分布
- 输出：每个动作的概率

**连续动作空间（如 Pendulum、机器人控制）：**
- 动作：连续值（如力矩 $\in [-2, 2]$）
- 策略：高斯分布（Gaussian）
- 输出：均值 $\mu$ 和标准差 $\sigma$

### 2.2 高斯策略网络

**网络结构：**

```python
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        # 输出均值
        self.mean = nn.Linear(64, action_dim)
        # 输出 log 标准差（可学习参数）
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        x = self.fc(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std
```

**关键设计：**

1. **均值 $\mu$**：由网络输出，依赖于状态
2. **标准差 $\sigma$**：
   - 方案 1：固定值（如 0.5）
   - 方案 2：可学习参数（与状态无关）
   - 方案 3：由网络输出（依赖于状态）
   - **推荐方案 2**：平衡灵活性和稳定性

3. **为什么输出 log_std？**
   - 保证 $\sigma > 0$：$\sigma = \exp(\log \sigma)$
   - 数值稳定性更好

### 2.3 动作采样和 log 概率

**采样动作：**

```python
def get_action(self, state):
    mean, std = self.forward(state)
    dist = torch.distributions.Normal(mean, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1)  # 多维动作求和
    return action, log_prob
```

**为什么多维动作要求和？**

假设动作是 2 维：$a = [a_1, a_2]$

如果各维度独立：

$$\log \pi(a|s) = \log \pi(a_1|s) + \log \pi(a_2|s)$$

### 2.4 动作裁剪

**问题：采样的动作可能超出环境限制**

例如 Pendulum 环境：动作范围 $[-2, 2]$，但高斯分布可能采样到 $-5$ 或 $5$。

**解决方案：**

```python
action = torch.tanh(action_raw) * action_bound
```

- $\tanh$ 将输出限制到 $[-1, 1]$
- 乘以 `action_bound` 缩放到实际范围

**注意：裁剪会影响 log 概率的计算！**

需要使用变换后的分布（Squashed Gaussian），或者简单地裁剪后不修正（实践中常用）。


## 3. PPO 实现细节

### 3.1 数据收集

**标准流程：**

```python
def collect_trajectories(env, policy, value_net, num_steps):
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
    
    state = env.reset()
    for _ in range(num_steps):
        # 采样动作
        action, log_prob = policy.get_action(state)
        value = value_net(state)
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        
        # 存储数据
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(value)
        
        state = next_state if not done else env.reset()
    
    return states, actions, rewards, dones, log_probs, values
```

**关键点：**
- 用**当前策略**采样数据
- 存储 **old_log_probs**（用于计算重要性采样比率）
- 存储 **values**（用于计算 GAE）

### 3.2 优势函数计算（GAE）

**GAE 公式（Lesson 11）：**

$$A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

**实现：**

```python
def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns
```

**技巧：优势函数归一化**

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

- 减少方差
- 稳定训练
- 几乎总是有益

### 3.3 PPO 更新

**完整损失函数：**

$$L_{total} = L^{CLIP} - c_1 L^{VF} + c_2 S[\pi_\theta](s)$$

其中：
- $L^{CLIP}$：PPO-Clip 损失（策略损失）
- $L^{VF}$：价值函数损失
- $S[\pi_\theta](s)$：熵正则化

**实现：**

```python
def ppo_update(policy, value_net, optimizer, states, actions, old_log_probs, 
               advantages, returns, clip_eps=0.2, epochs=10, batch_size=64):
    
    for _ in range(epochs):
        # 随机打乱数据
        indices = np.random.permutation(len(states))
        
        for start in range(0, len(states), batch_size):
            # Mini-batch
            batch_idx = indices[start:start + batch_size]
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]
            
            # 计算新策略的 log 概率
            new_log_probs, entropy = policy.evaluate_actions(batch_states, batch_actions)
            
            # 重要性采样比率
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            
            # PPO-Clip 损失
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值函数损失
            values = value_net(batch_states)
            value_loss = (values - batch_returns).pow(2).mean()
            
            # 熵正则化
            entropy_loss = -entropy.mean()
            
            # 总损失
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            
            # 梯度下降
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_grad_norm=0.5)
            optimizer.step()
```

### 3.4 价值函数裁剪（可选）

**动机：防止价值函数更新过大**

```python
value_pred_clipped = value_old + torch.clamp(
    value - value_old, -clip_range, clip_range
)
value_loss = torch.max(
    (value - returns)**2,
    (value_pred_clipped - returns)**2
).mean()
```

**效果：**
- 进一步稳定训练
- OpenAI Baselines 使用
- 但不是必需的


## 4. PPO 超参数调优

### 4.1 关键超参数

| 超参数 | 典型值 | 调优建议 |
|--------|--------|----------|
| **clip_epsilon** | 0.1 - 0.2 | 越小越保守，越大学习越快但可能不稳定 |
| **learning_rate** | 3e-4 | 可以尝试 1e-4 到 1e-3 |
| **gamma** | 0.99 | 长期任务用 0.99，短期任务用 0.95 |
| **gae_lambda** | 0.95 | 0.9 - 0.99，影响偏差-方差权衡 |
| **epochs** | 3 - 10 | 越多越充分利用数据，但可能过拟合 |
| **batch_size** | 64 - 256 | 越大越稳定，但计算开销大 |
| **steps_per_iteration** | 2048 - 4096 | 越多越稳定，但采样开销大 |
| **value_coef** | 0.5 | 价值函数损失的权重 |
| **entropy_coef** | 0.01 | 熵正则化的权重，鼓励探索 |

### 4.2 调优策略

**1. 从默认值开始**

```python
# 推荐的默认配置
config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'epochs': 10,
    'batch_size': 64,
    'steps_per_iteration': 2048,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
}
```

**2. 逐步调整**

- **学习率**：如果训练不稳定，降低学习率
- **clip_epsilon**：如果学习太慢，增大 epsilon
- **epochs**：如果数据利用不充分，增加 epochs
- **entropy_coef**：如果探索不足，增大熵系数

**3. 监控指标**

- **平均奖励**：主要优化目标
- **策略损失**：应该逐渐减小
- **价值损失**：应该逐渐减小
- **熵**：应该逐渐减小（但不要太快）
- **KL 散度**：监控策略变化幅度（应该较小）

### 4.3 学习率调度

**线性衰减（推荐）：**

```python
def linear_schedule(initial_lr, current_step, total_steps):
    return initial_lr * (1 - current_step / total_steps)
```

**余弦退火：**

```python
def cosine_schedule(initial_lr, current_step, total_steps):
    return initial_lr * 0.5 * (1 + np.cos(np.pi * current_step / total_steps))
```

**效果：**
- 后期更精细的调整
- 提高最终性能
- 但不是必需的


## 5. PPO 的工程技巧

### 5.1 奖励归一化

**问题：不同环境的奖励尺度差异很大**

**解决方案：运行时奖励归一化**

```python
class RunningMeanStd:
    def __init__(self):
        self.mean = 0
        self.var = 1
        self.count = 1e-4
    
    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean += delta * batch_count / total_count
        self.var = (self.var * self.count + batch_var * batch_count + 
                    delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count
    
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
```

### 5.2 观测归一化

**同样的思路应用于状态观测：**

```python
normalized_state = (state - running_mean) / (running_std + 1e-8)
```

**效果：**
- 加速训练
- 提高稳定性
- 几乎总是有益

### 5.3 梯度裁剪

**防止梯度爆炸：**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

**典型值：0.5 - 1.0**

### 5.4 正交初始化

**网络权重初始化：**

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0)

model.apply(init_weights)
```

**效果：**
- 更好的初始性能
- 更快的收敛
- OpenAI Baselines 使用

### 5.5 帧堆叠（Frame Stacking）

**对于视觉输入（如 Atari）：**

```python
# 堆叠最近 4 帧
stacked_frames = np.concatenate([frame_t-3, frame_t-2, frame_t-1, frame_t], axis=-1)
```

**效果：**
- 隐式表示速度信息
- 提高性能


## 6. PPO 的变体

### 6.1 PPO-Penalty

**使用 KL 惩罚代替裁剪：**

$$L^{KLPEN}(\theta) = \mathbb{E}_t \left[ r_t(\theta) A_t - \beta \cdot KL[\pi_{\theta_{old}} || \pi_\theta] \right]$$

**自适应调整 $\beta$：**
- 如果 KL 散度太大，增大 $\beta$
- 如果 KL 散度太小，减小 $\beta$

**对比 PPO-Clip：**
- PPO-Clip 更简单，更常用
- PPO-Penalty 更接近 TRPO 的思想

### 6.2 PPO with Auxiliary Tasks

**添加辅助任务提高样本效率：**

1. **像素重建**（视觉输入）
2. **奖励预测**
3. **状态预测**

**效果：**
- 提高样本效率
- 学习更好的特征表示
- 但增加了实现复杂度

### 6.3 Recurrent PPO (R-PPO)

**使用 LSTM/GRU 处理部分可观测环境：**

```python
class RecurrentPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state, hidden):
        x, hidden = self.lstm(state, hidden)
        return self.fc(x), hidden
```

**应用场景：**
- 部分可观测环境（POMDP）
- 需要记忆的任务


## 7. PPO 在实践中的应用

### 7.1 连续控制任务

**MuJoCo 环境：**
- Hopper, Walker2d, HalfCheetah, Ant
- 高维连续动作空间
- PPO 是标准基线

**典型性能：**
- Hopper: ~3000 奖励
- Walker2d: ~4000 奖励
- HalfCheetah: ~6000 奖励

### 7.2 机器人控制

**真实机器人：**
- 抓取、操作、导航
- PPO 的稳定性至关重要
- 通常先在仿真中训练，再迁移到真实机器人

**Sim-to-Real 技巧：**
- 域随机化（Domain Randomization）
- 系统辨识（System Identification）
- 渐进式训练

### 7.3 游戏 AI

**成功案例：**
- Dota 2 (OpenAI Five)
- StarCraft II (AlphaStar 的一部分)
- Hide and Seek (OpenAI)

**关键技术：**
- 大规模分布式训练
- 课程学习
- 自博弈

### 7.4 RLHF (Reinforcement Learning from Human Feedback)

**ChatGPT / InstructGPT：**
- 用 PPO 对齐大语言模型
- 奖励模型：人类偏好
- 策略：语言模型

**为什么选择 PPO？**
- 训练稳定（语言模型参数量大）
- 实现简单（易于集成到现有框架）
- 对超参数鲁棒


## 8. PPO vs 其他算法（实践视角）

### 8.1 PPO vs A2C

| 特性 | A2C | PPO |
|------|-----|-----|
| 样本效率 | 低 | 中 |
| 训练稳定性 | 中 | 高 |
| 实现复杂度 | 低 | 中 |
| 超参数敏感性 | 高 | 低 |
| 推荐场景 | 快速原型 | 生产环境 |

**结论：PPO 几乎总是更好的选择**

### 8.2 PPO vs SAC

| 特性 | PPO | SAC |
|------|-----|-----|
| 样本效率 | 中 | 高 |
| 训练稳定性 | 高 | 高 |
| 适用场景 | 通用 | 连续控制 |
| 探索能力 | 中 | 高（最大熵） |

**结论：**
- 连续控制 + 样本效率重要 → SAC
- 其他场景 → PPO

### 8.3 PPO vs DQN

| 特性 | DQN | PPO |
|------|-----|-----|
| 样本效率 | 高 | 中 |
| 动作空间 | 离散 | 离散 + 连续 |
| 训练稳定性 | 中 | 高 |
| 实现复杂度 | 中 | 中 |

**结论：**
- 离散动作 + 样本效率重要 → DQN
- 连续动作 → PPO


## 9. 调试 PPO 的常见问题

### 9.1 训练不收敛

**可能原因：**
1. 学习率太大
2. clip_epsilon 太大
3. 价值函数学习不好

**解决方案：**
- 降低学习率（3e-4 → 1e-4）
- 减小 clip_epsilon（0.2 → 0.1）
- 增大 value_coef（0.5 → 1.0）
- 检查价值函数损失是否下降

### 9.2 性能崩溃

**可能原因：**
1. 策略更新太激进
2. 奖励尺度不合适
3. 梯度爆炸

**解决方案：**
- 减小 clip_epsilon
- 奖励归一化
- 梯度裁剪
- 减少 epochs

### 9.3 探索不足

**可能原因：**
1. 熵系数太小
2. 策略标准差太小

**解决方案：**
- 增大 entropy_coef（0.01 → 0.05）
- 增大初始 log_std
- 使用 ε-greedy 探索（离散动作）

### 9.4 样本效率低

**可能原因：**
1. epochs 太少
2. batch_size 太小
3. 价值函数不准确

**解决方案：**
- 增加 epochs（10 → 20）
- 增大 batch_size（64 → 128）
- 改进价值网络结构


## 10. 总结

### 10.1 核心要点

1. **连续动作空间**：
   - 使用高斯策略
   - 输出均值和标准差
   - log_std 作为可学习参数

2. **实现细节**：
   - GAE 计算优势函数
   - 优势函数归一化
   - Mini-batch 更新
   - 梯度裁剪

3. **超参数调优**：
   - 从默认值开始
   - 监控关键指标
   - 逐步调整

4. **工程技巧**：
   - 奖励归一化
   - 观测归一化
   - 正交初始化
   - 学习率调度

5. **PPO 的优势**：
   - 训练稳定
   - 实现简单
   - 对超参数鲁棒
   - 广泛应用

### 10.2 实践建议

1. **从简单环境开始**：
   - CartPole → Pendulum → MuJoCo
   - 逐步增加复杂度

2. **使用成熟的实现**：
   - Stable-Baselines3
   - CleanRL
   - 学习最佳实践

3. **监控训练过程**：
   - 使用 TensorBoard / Weights & Biases
   - 记录所有关键指标
   - 及时发现问题

4. **耐心调试**：
   - PPO 通常需要较长时间训练
   - 不要过早放弃
   - 系统性地排查问题

### 10.3 下节预告

下节课我们将学习 **DDPG & TD3**：
- 确定性策略梯度
- DDPG 算法
- Twin Delayed DDPG (TD3)
- 在连续控制任务上的应用

### 10.4 扩展阅读

**必读论文：**
1. Schulman et al. (2017). "Proximal Policy Optimization Algorithms"
2. Engstrom et al. (2020). "Implementation Matters in Deep RL: A Case Study on PPO and TRPO"

**推荐资源：**
- Stable-Baselines3 文档：https://stable-baselines3.readthedocs.io/
- CleanRL PPO 实现：https://github.com/vwxyzjn/cleanrl
- Spinning Up in Deep RL: PPO
- 37 Implementation Details of PPO: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
