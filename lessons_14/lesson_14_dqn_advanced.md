# Lesson 14: DQN 进阶 (Advanced DQN)

## 1. 回顾：DQN 的局限性

在上节课中，我们学习了 DQN 的核心思想：
- **经验回放（Experience Replay）**：打破样本相关性
- **目标网络（Target Network）**：稳定训练目标

但 DQN 仍然存在一些问题：

1. **高估偏差（Overestimation Bias）**：DQN 倾向于高估 Q 值
2. **样本效率低**：需要大量样本才能收敛
3. **网络结构单一**：没有充分利用状态价值和优势函数的分离

本节课将介绍三个重要的 DQN 改进算法：
- **Double DQN**：解决高估偏差
- **Dueling DQN**：分离状态价值和优势函数
- **Prioritized Experience Replay**：优先采样重要经验


## 2. Double DQN

### 2.1 DQN 的高估偏差问题

**问题：DQN 为什么会高估 Q 值？**

回顾 DQN 的 TD 目标：

$$y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)$$

这里使用了 $\max$ 操作来选择下一个状态的最优动作。

**关键问题：$\max$ 操作会放大估计误差！**

假设 Q 值估计存在随机误差：

$$Q(s, a) = Q^*(s, a) + \epsilon_a$$

其中 $\epsilon_a$ 是零均值的随机误差。

那么：

$$\max_a Q(s, a) = \max_a [Q^*(s, a) + \epsilon_a] \geq Q^*(s, a^*) + \epsilon_{a^*}$$

由于 $\max$ 操作会选择误差最大的动作，导致：

$$\mathbb{E}[\max_a Q(s, a)] > \max_a \mathbb{E}[Q(s, a)] = \max_a Q^*(s, a)$$

**结论：DQN 系统性地高估 Q 值！**

### 2.2 Double DQN 的解决方案

**核心思想：解耦动作选择和动作评估**

在 DQN 中，同一个网络既用于选择动作，又用于评估动作：

$$y_t^{DQN} = r_t + \gamma Q(s_{t+1}, \arg\max_{a'} Q(s_{t+1}, a'; \theta^-); \theta^-)$$

**Double DQN 的改进：**

1. **用在线网络选择动作**：$a^* = \arg\max_{a'} Q(s_{t+1}, a'; \theta)$
2. **用目标网络评估动作**：$y_t = r_t + \gamma Q(s_{t+1}, a^*; \theta^-)$

完整的 Double DQN 目标：

$$y_t^{DoubleDQN} = r_t + \gamma Q(s_{t+1}, \arg\max_{a'} Q(s_{t+1}, a'; \theta); \theta^-)$$

**为什么这样可以减少高估？**

- 在线网络 $\theta$ 和目标网络 $\theta^-$ 的误差是独立的
- 即使在线网络选择了高估的动作，目标网络也不太可能同时高估该动作
- 两个网络的误差相互抵消，减少了高估偏差

### 2.3 Double DQN 实现

**代码修改非常简单，只需改变 TD 目标的计算：**

```python
# DQN
with torch.no_grad():
    next_q_values = target_net(next_states).max(1)[0]
    targets = rewards + gamma * next_q_values * (1 - dones)

# Double DQN
with torch.no_grad():
    # 用在线网络选择动作
    next_actions = online_net(next_states).argmax(1)
    # 用目标网络评估动作
    next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
    targets = rewards + gamma * next_q_values * (1 - dones)
```

### 2.4 Double DQN 的效果

**实验结果（Atari 游戏）：**
- Double DQN 显著减少了 Q 值的高估
- 在大多数游戏上性能优于 DQN
- 训练更稳定，收敛更快

**理论保证：**
- Double DQN 可能会轻微低估 Q 值，但低估程度远小于 DQN 的高估
- 在实践中，轻微低估比严重高估更好


## 3. Dueling DQN

### 3.1 动机：分离状态价值和优势函数

**回顾：Q 函数的分解**

在 Lesson 11（Actor-Critic）中，我们学习了优势函数：

$$A(s, a) = Q(s, a) - V(s)$$

其中：
- $V(s)$：状态价值，表示"在状态 $s$ 下的期望回报"
- $A(s, a)$：优势函数，表示"动作 $a$ 相对于平均动作的优势"

因此：

$$Q(s, a) = V(s) + A(s, a)$$

**关键洞察：**

在很多状态下，**不同动作的价值差异很小**！

例如：
- 在 Atari 游戏中，大部分时间智能体只是在等待，不同动作的价值几乎相同
- 只有在关键时刻（如躲避敌人、射击目标），动作选择才重要

**问题：**

传统 DQN 对每个动作都独立估计 Q 值，没有利用这个结构。

### 3.2 Dueling DQN 的网络结构

**核心思想：显式地分离 $V(s)$ 和 $A(s, a)$**

**网络结构：**

```
输入状态 s
    ↓
共享特征提取层（卷积层或全连接层）
    ↓
    ├─────────────┬─────────────┐
    ↓             ↓             ↓
价值流 V(s)    优势流 A(s,a)
(单个输出)    (num_actions 个输出)
    ↓             ↓
    └─────────────┴─────────────┘
              ↓
        Q(s, a) = V(s) + A(s, a)
```

**数学表示：**

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + A(s, a; \theta, \alpha)$$

其中：
- $\theta$：共享特征提取层的参数
- $\beta$：价值流的参数
- $\alpha$：优势流的参数

### 3.3 可识别性问题（Identifiability Problem）

**问题：上述分解不唯一！**

给定 $Q(s, a)$，可以有无穷多种方式分解为 $V(s)$ 和 $A(s, a)$：

$$Q(s, a) = V(s) + A(s, a) = [V(s) + C] + [A(s, a) - C]$$

对于任意常数 $C$，上式都成立。

**后果：**
- 网络无法学习到唯一的 $V(s)$ 和 $A(s, a)$
- 训练不稳定

**解决方案：强制约束**

**方案 1：减去平均优势**

$$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a') \right)$$

这样强制 $\mathbb{E}_a[A(s, a)] = 0$。

**方案 2：减去最大优势（推荐）**

$$Q(s, a) = V(s) + \left( A(s, a) - \max_{a'} A(s, a') \right)$$

这样强制最优动作的优势为 0。

**优势：**
- 方案 2 更稳定（只依赖一个动作，而不是所有动作的平均）
- 方案 2 在实践中效果更好

### 3.4 Dueling DQN 的优势

**1. 更好的状态价值估计**

- 价值流 $V(s)$ 在每次更新时都会被训练
- 即使某些动作很少被选择，$V(s)$ 也能得到充分训练

**2. 更快的学习**

- 在动作价值差异小的状态下，网络主要学习 $V(s)$
- 在动作价值差异大的状态下，网络主要学习 $A(s, a)$
- 分工明确，学习效率更高

**3. 更好的泛化**

- $V(s)$ 和 $A(s, a)$ 的分离使得网络更容易泛化到新状态

### 3.5 Dueling DQN 实现

```python
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        
        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # 方案 2：减去最大优势
        q_values = value + (advantage - advantage.max(dim=1, keepdim=True)[0])
        return q_values
```


## 4. Prioritized Experience Replay (PER)

### 4.1 动机：不是所有经验都同等重要

**问题：均匀采样的局限**

在标准 DQN 中，从回放缓冲区**均匀随机采样**经验。

但是：
- 有些经验包含更多信息（例如，TD 误差大的经验）
- 有些经验已经被充分学习（例如，TD 误差小的经验）

**核心思想：优先采样 TD 误差大的经验**

TD 误差大 → 模型预测不准 → 包含更多信息 → 应该更频繁地学习

### 4.2 优先级的定义

**TD 误差作为优先级：**

$$\delta_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-) - Q(s_t, a_t; \theta)$$

**优先级：**

$$p_t = |\delta_t| + \epsilon$$

其中 $\epsilon > 0$ 是一个小常数，确保所有经验都有非零概率被采样。

**采样概率：**

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

其中 $\alpha$ 控制优先级的强度：
- $\alpha = 0$：均匀采样（标准 DQN）
- $\alpha = 1$：完全按优先级采样
- 典型值：$\alpha = 0.6$

### 4.3 重要性采样（Importance Sampling）

**问题：优先采样引入偏差**

优先采样改变了数据分布，导致估计有偏。

**解决方案：重要性采样权重**

$$w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta$$

其中：
- $N$：回放缓冲区大小
- $\beta$：重要性采样指数，从 $\beta_0$ 线性增长到 1
- 典型值：$\beta_0 = 0.4$，在训练结束时 $\beta = 1$

**修正后的损失函数：**

$$L(\theta) = \frac{1}{B} \sum_{i=1}^B w_i \cdot \delta_i^2$$

### 4.4 高效实现：Sum Tree

**问题：如何高效地按优先级采样？**

朴素实现：
- 存储所有优先级，计算累积和
- 采样时二分查找
- 时间复杂度：$O(N)$ 更新，$O(\log N)$ 采样

**Sum Tree 数据结构：**

```
        总和 (root)
       /          \
    左子树和      右子树和
    /    \        /    \
  p1    p2      p3    p4  (叶子节点：优先级)
```

**操作：**
- **采样**：从根节点开始，按概率选择左/右子树，递归到叶子节点 → $O(\log N)$
- **更新**：更新叶子节点，向上传播更新父节点 → $O(\log N)$

### 4.5 PER 的效果

**实验结果（Atari 游戏）：**
- PER 显著提高样本效率（减少 50% 的样本量）
- 在大多数游戏上性能优于 DQN
- 训练速度更快

**缺点：**
- 实现复杂度高
- 计算开销增加（约 20-30%）


## 5. Rainbow DQN：集成所有改进

### 5.1 Rainbow 的组成

**Rainbow DQN（2017）集成了 6 个 DQN 改进：**

1. **Double DQN**：减少高估偏差
2. **Dueling DQN**：分离状态价值和优势函数
3. **Prioritized Experience Replay**：优先采样重要经验
4. **Multi-step Learning**：使用 n-step TD 目标
5. **Distributional RL (C51)**：学习价值分布而不是期望
6. **Noisy Nets**：参数空间探索

### 5.2 Multi-step Learning

**核心思想：使用 n-step TD 目标**

标准 DQN 使用 1-step TD 目标：

$$y_t^{(1)} = r_t + \gamma \max_{a'} Q(s_{t+1}, a')$$

n-step TD 目标：

$$y_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q(s_{t+n}, a')$$

**优势：**
- 更快地传播奖励信号
- 减少偏差（bias）
- 典型值：$n = 3$

### 5.3 Distributional RL (C51)

**核心思想：学习价值分布，而不是期望**

传统 RL 学习期望回报：

$$Q(s, a) = \mathbb{E}[G_t | s_t = s, a_t = a]$$

Distributional RL 学习整个分布：

$$Z(s, a) = G_t | s_t = s, a_t = a$$

**C51 算法：**
- 将价值范围离散化为 51 个 atoms
- 用分类分布表示价值分布
- 使用 KL 散度作为损失函数

**优势：**
- 更丰富的价值表示
- 更稳定的训练
- 更好的性能

### 5.4 Noisy Nets

**核心思想：在参数空间进行探索**

传统 ε-greedy 在动作空间探索：

$$a = \begin{cases} 
\text{random action} & \text{with prob } \epsilon \\
\arg\max_a Q(s, a) & \text{with prob } 1 - \epsilon
\end{cases}$$

Noisy Nets 在参数空间探索：

$$y = (W + \sigma \odot \epsilon) x + b$$

其中：
- $W, b$：可学习的参数
- $\sigma$：可学习的噪声标准差
- $\epsilon \sim \mathcal{N}(0, 1)$：随机噪声

**优势：**
- 自适应探索（网络自己学习探索策略）
- 状态相关的探索（不同状态探索程度不同）
- 无需手动调节 ε

### 5.5 Rainbow 的性能

**实验结果（Atari 游戏）：**
- Rainbow 在 57 个 Atari 游戏上的平均性能是 DQN 的 4 倍
- 样本效率提高 10 倍
- 是当时（2017）最强的 value-based RL 算法

**消融实验（Ablation Study）：**
- 每个组件都有贡献
- Prioritized Replay 和 Multi-step Learning 贡献最大
- 组件之间有协同效应


## 6. DQN 系列算法对比

| 算法 | 核心改进 | 优势 | 缺点 |
|------|----------|------|------|
| DQN | 经验回放 + 目标网络 | 稳定训练 | 高估偏差、样本效率低 |
| Double DQN | 解耦动作选择和评估 | 减少高估 | 可能轻微低估 |
| Dueling DQN | 分离 V(s) 和 A(s,a) | 更快学习、更好泛化 | 实现稍复杂 |
| PER | 优先采样重要经验 | 样本效率高 | 实现复杂、计算开销大 |
| Rainbow | 集成 6 个改进 | 性能最强 | 实现非常复杂 |


## 7. 实现要点

### 7.1 Double DQN 实现

```python
def train_double_dqn(self):
    # ... 采样 batch ...
    
    # 计算当前 Q 值
    q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Double DQN 目标
    with torch.no_grad():
        # 用在线网络选择动作
        next_actions = self.online_net(next_states).argmax(1)
        # 用目标网络评估动作
        next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards + self.gamma * next_q_values * (1 - dones)
    
    # 计算损失
    loss = nn.MSELoss()(q_values, targets)
    
    # ... 梯度下降 ...
```

### 7.2 Dueling DQN 实现

```python
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        
        # 共享层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # 减去最大优势
        q_values = value + (advantage - advantage.max(dim=1, keepdim=True)[0])
        return q_values
```

### 7.3 调试技巧

1. **先实现 Double DQN**：
   - 改动最小，容易调试
   - 验证是否减少了高估

2. **再实现 Dueling DQN**：
   - 可视化 V(s) 和 A(s,a)
   - 检查是否合理分离

3. **最后实现 PER**：
   - 先用简单的优先级（不用 Sum Tree）
   - 验证效果后再优化实现


## 8. 总结

### 8.1 核心要点

1. **Double DQN**：
   - 解耦动作选择和评估
   - 减少高估偏差
   - 实现简单，效果显著

2. **Dueling DQN**：
   - 分离状态价值和优势函数
   - 更快学习，更好泛化
   - 网络结构改进

3. **Prioritized Experience Replay**：
   - 优先采样 TD 误差大的经验
   - 提高样本效率
   - 需要重要性采样修正偏差

4. **Rainbow DQN**：
   - 集成 6 个改进
   - 当时最强的 value-based RL 算法
   - 实现复杂，但性能优异

### 8.2 实践建议

1. **从 Double DQN 开始**：
   - 改动最小，收益明显
   - 几乎没有额外计算开销

2. **Dueling DQN 适合复杂环境**：
   - 状态空间大、动作价值差异小的环境
   - 例如 Atari 游戏

3. **PER 适合样本效率敏感的场景**：
   - 环境交互成本高
   - 愿意牺牲计算开销换取样本效率

4. **Rainbow 适合追求极致性能**：
   - 有充足的计算资源
   - 需要最好的性能

### 8.3 下节预告

下节课我们将学习 **PPO (Proximal Policy Optimization)**：
- 从 value-based 转向 policy-based
- 信赖域方法的思想
- PPO-Clip 目标函数推导
- PPO 在实践中的主导地位

### 8.4 扩展阅读

**必读论文：**
1. van Hasselt et al. (2016). "Deep Reinforcement Learning with Double Q-learning"
2. Wang et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning"
3. Schaul et al. (2016). "Prioritized Experience Replay"
4. Hessel et al. (2018). "Rainbow: Combining Improvements in Deep Reinforcement Learning"

**推荐资源：**
- Spinning Up in Deep RL: DQN Variants
- CleanRL 实现：https://github.com/vwxyzjn/cleanrl
