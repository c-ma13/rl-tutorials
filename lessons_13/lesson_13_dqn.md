# Lesson 13: DQN (Deep Q-Network)

## 1. 为什么需要 DQN？

### 1.1 传统 Q-Learning 的局限

在前面的课程中，我们学习了 Q-Learning 算法：

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

这个算法在**表格形式**（tabular）下工作良好，即我们用一个表格存储每个状态-动作对的 Q 值。

**但是，当状态空间很大或连续时，表格方法失效：**

- **Atari 游戏**：状态是 84×84×4 的图像，状态空间约 $256^{84 \times 84 \times 4} \approx 10^{67970}$
- **围棋**：状态空间约 $10^{170}$
- **机器人控制**：连续状态空间（关节角度、速度等）

**问题：**
1. 存储：无法为每个状态-动作对存储一个 Q 值
2. 泛化：无法从见过的状态泛化到未见过的状态
3. 学习效率：需要访问每个状态多次才能学习

### 1.2 函数逼近的思路

**核心思想：用函数逼近器（function approximator）来表示 Q 函数**

$$Q(s, a) \approx Q(s, a; \theta)$$

其中 $\theta$ 是函数逼近器的参数（例如神经网络的权重）。

**优势：**
- 参数数量远小于状态空间大小
- 自动泛化到相似状态
- 可以处理连续状态空间

**DQN = Q-Learning + Deep Neural Network**


## 2. Naive Deep Q-Learning 的问题

### 2.1 直接应用神经网络的尝试

最直接的想法：用神经网络 $Q(s, a; \theta)$ 替换 Q 表格，然后用梯度下降最小化 TD 误差：

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta) \right)^2 \right]$$

**但这样做会导致训练不稳定甚至发散！**

### 2.2 不稳定的原因

**问题 1：样本相关性（Sample Correlation）**

在线 RL 中，连续的样本 $(s_t, a_t, r_t, s_{t+1})$ 高度相关：
- 相邻状态相似
- 动作由当前策略决定，存在偏向性
- 违反了监督学习中"独立同分布"（i.i.d.）的假设

**后果：**
- 神经网络容易过拟合到最近的样本
- 梯度更新方向不稳定
- 容易陷入局部最优或发散

**问题 2：目标不稳定（Moving Target）**

TD 目标是：

$$y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta)$$

注意：**目标 $y_t$ 本身依赖于参数 $\theta$**！

每次更新 $\theta$ 后，目标 $y_t$ 也会改变，就像"追逐一个移动的靶子"。

**后果：**
- 训练过程像"狗追尾巴"
- Q 值容易震荡或发散
- 难以收敛

**问题 3：奖励和状态分布的非平稳性**

随着策略改进，智能体访问的状态分布会变化，导致训练数据分布不断变化。


## 3. DQN 的两大创新

DeepMind 在 2013 年提出的 DQN 算法通过两个关键技术解决了上述问题：

### 3.1 经验回放（Experience Replay）

**核心思想：将经验存储在回放缓冲区中，训练时随机采样**

**算法流程：**

1. **存储经验**：将每个转移 $(s_t, a_t, r_t, s_{t+1}, done_t)$ 存入回放缓冲区 $\mathcal{D}$
2. **随机采样**：从 $\mathcal{D}$ 中随机采样一个 mini-batch 进行训练
3. **梯度更新**：用采样的 batch 更新网络参数

**优势：**

✅ **打破样本相关性**：随机采样使得 batch 中的样本近似独立

✅ **提高样本效率**：每个经验可以被多次使用（off-policy）

✅ **稳定训练**：平滑了数据分布的变化

**实现细节：**

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

### 3.2 目标网络（Target Network）

**核心思想：用一个独立的网络来计算 TD 目标，定期更新**

**算法流程：**

1. **维护两个网络**：
   - **在线网络（Online Network）** $Q(s, a; \theta)$：每步更新，用于选择动作
   - **目标网络（Target Network）** $Q(s, a; \theta^-)$：固定参数，用于计算 TD 目标

2. **TD 目标计算**：
   $$y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)$$
   
   注意：目标网络参数 $\theta^-$ 是固定的！

3. **损失函数**：
   $$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( y - Q(s, a; \theta) \right)^2 \right]$$

4. **定期更新目标网络**：每隔 $C$ 步，令 $\theta^- \leftarrow \theta$

**优势：**

✅ **稳定目标**：在 $C$ 步内，TD 目标保持不变

✅ **减少震荡**：避免"追逐移动靶子"的问题

✅ **提高收敛性**：类似于监督学习中的固定标签


## 4. DQN 算法完整流程

### 4.1 伪代码

```
初始化：
  - 在线网络 Q(s, a; θ) 随机初始化
  - 目标网络 Q(s, a; θ⁻) ← θ
  - 回放缓冲区 D，容量 N
  - 超参数：batch_size, γ, ε, learning_rate, C (目标网络更新频率)

对于每个 episode：
  初始化状态 s
  
  对于每个时间步 t：
    1. 用 ε-greedy 选择动作：
       a = argmax_a Q(s, a; θ)  (概率 1-ε)
       a = random action        (概率 ε)
    
    2. 执行动作 a，观察 r, s'
    
    3. 存储转移 (s, a, r, s', done) 到 D
    
    4. 如果 |D| >= batch_size：
       a) 从 D 中随机采样 mini-batch
       b) 对于 batch 中每个样本 (sⱼ, aⱼ, rⱼ, s'ⱼ, doneⱼ)：
          如果 doneⱼ：
            yⱼ = rⱼ
          否则：
            yⱼ = rⱼ + γ max_a' Q(s'ⱼ, a'; θ⁻)
       
       c) 计算损失：L(θ) = (1/batch_size) Σⱼ (yⱼ - Q(sⱼ, aⱼ; θ))²
       d) 梯度下降更新 θ
    
    5. 每隔 C 步：θ⁻ ← θ
    
    6. s ← s'
```

### 4.2 网络结构

**对于 Atari 游戏（图像输入）：**

```
输入：84×84×4 的灰度图像（4 帧堆叠）
  ↓
Conv1: 32 filters, 8×8, stride 4, ReLU
  ↓
Conv2: 64 filters, 4×4, stride 2, ReLU
  ↓
Conv3: 64 filters, 3×3, stride 1, ReLU
  ↓
Flatten
  ↓
FC1: 512 units, ReLU
  ↓
FC2: num_actions units (输出每个动作的 Q 值)
```

**对于简单环境（向量输入，如 CartPole）：**

```
输入：状态向量（例如 4 维）
  ↓
FC1: 128 units, ReLU
  ↓
FC2: 128 units, ReLU
  ↓
FC3: num_actions units
```

### 4.3 关键超参数

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| 回放缓冲区大小 | 10,000 - 1,000,000 | Atari 用 1M，简单环境用 10k-100k |
| Batch size | 32 - 128 | 平衡训练速度和稳定性 |
| 学习率 | 1e-4 - 1e-3 | Adam 优化器 |
| 折扣因子 γ | 0.99 | 标准值 |
| ε (初始) | 1.0 | 完全探索 |
| ε (最终) | 0.01 - 0.1 | 保留少量探索 |
| ε 衰减步数 | 10,000 - 1,000,000 | 线性衰减 |
| 目标网络更新频率 C | 1,000 - 10,000 | 硬更新（hard update） |


## 5. DQN 的训练技巧

### 5.1 ε-Greedy 探索策略

DQN 使用 ε-greedy 进行探索，通常采用**线性衰减**：

$$\epsilon_t = \max(\epsilon_{end}, \epsilon_{start} - \frac{t}{T}(\epsilon_{start} - \epsilon_{end}))$$

- $\epsilon_{start} = 1.0$：初始完全随机探索
- $\epsilon_{end} = 0.01$：最终保留 1% 的探索
- $T$：衰减步数（例如 100,000 步）

### 5.2 奖励裁剪（Reward Clipping）

在 Atari 游戏中，不同游戏的奖励尺度差异很大（有的是 1 分，有的是 1000 分）。

**技巧：将所有奖励裁剪到 [-1, 1]**

$$r_{clipped} = \text{sign}(r) = \begin{cases} 
+1 & \text{if } r > 0 \\
0 & \text{if } r = 0 \\
-1 & \text{if } r < 0
\end{cases}$$

**优势：**
- 统一不同游戏的奖励尺度
- 稳定训练（避免大奖励导致的梯度爆炸）

### 5.3 梯度裁剪（Gradient Clipping）

防止梯度爆炸：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

### 5.4 帧堆叠（Frame Stacking）

对于 Atari 游戏，单帧图像无法表示速度信息。

**技巧：将连续 4 帧堆叠作为输入**

- 输入形状：(84, 84, 4)
- 可以隐式表示速度和方向

### 5.5 帧跳过（Frame Skipping）

Atari 游戏运行在 60 FPS，但智能体不需要每帧都做决策。

**技巧：每 4 帧执行一次动作，重复执行该动作**

- 减少计算量
- 加速训练


## 6. DQN 的理论分析

### 6.1 收敛性

**定理（Mnih et al., 2015）：**

在以下条件下，DQN 可以收敛到最优 Q 函数的近似：

1. 函数逼近器（神经网络）有足够的表达能力
2. 回放缓冲区足够大
3. 目标网络更新频率适当
4. 学习率足够小

**注意：**
- DQN 的收敛性**没有严格的理论保证**（非线性函数逼近 + off-policy + bootstrapping）
- 实践中，DQN 在很多任务上表现良好，但也可能不稳定或发散

### 6.2 样本效率

**DQN 是 off-policy 算法**：
- 可以重复使用旧经验
- 样本效率高于 on-policy 算法（如 A2C、PPO）

**但是：**
- 需要大量样本才能收敛（Atari 游戏通常需要 10M - 200M 帧）
- 不如 model-based RL 方法样本效率高


## 7. DQN 的成功案例

### 7.1 Atari 游戏（2013-2015）

**论文：**
- "Playing Atari with Deep Reinforcement Learning" (2013, NIPS Workshop)
- "Human-level control through deep reinforcement learning" (2015, Nature)

**成果：**
- 在 49 个 Atari 游戏中，DQN 在 29 个游戏上超过人类水平
- 使用**相同的网络结构和超参数**（端到端学习）
- 仅从像素输入学习，无需手工特征

**代表性游戏：**
- Breakout：学会了"挖隧道"策略
- Pong：完美击败内置 AI
- Space Invaders：学会了躲避和射击

### 7.2 DQN 的局限性

尽管 DQN 取得了巨大成功，但也存在一些问题：

1. **高估偏差（Overestimation Bias）**：
   - $\max_{a'} Q(s', a')$ 倾向于高估 Q 值
   - 导致训练不稳定
   - 解决方案：Double DQN（下节课）

2. **样本效率低**：
   - 需要数百万甚至上亿帧才能收敛
   - 解决方案：Prioritized Experience Replay（下节课）

3. **无法处理连续动作空间**：
   - DQN 需要枚举所有动作来计算 $\max_{a'} Q(s', a')$
   - 解决方案：DDPG、TD3、SAC（后续课程）

4. **训练不稳定**：
   - 对超参数敏感
   - 可能发散或陷入局部最优
   - 解决方案：Rainbow DQN（下节课）


## 8. 实现要点

### 8.1 PyTorch 实现框架

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

def train_dqn(env, num_episodes, ...):
    # 初始化网络
    online_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(online_net.state_dict())
    
    optimizer = optim.Adam(online_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for t in range(max_steps):
            # ε-greedy 选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = online_net(torch.FloatTensor(state))
                    action = q_values.argmax().item()
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            
            # 训练
            if len(replay_buffer) >= batch_size:
                # 采样 batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # 转换为 tensor
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)
                
                # 计算 Q(s, a)
                q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # 计算目标 y = r + γ max_a' Q(s', a'; θ⁻)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    targets = rewards + gamma * next_q_values * (1 - dones)
                
                # 计算损失
                loss = nn.MSELoss()(q_values, targets)
                
                # 梯度下降
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 更新目标网络
            if t % target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # 衰减 epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
```

### 8.2 调试技巧

1. **监控 Q 值**：
   - 绘制平均 Q 值曲线
   - Q 值应该逐渐增大并稳定

2. **监控损失**：
   - TD 损失应该逐渐减小
   - 如果损失爆炸，检查学习率和梯度裁剪

3. **监控奖励**：
   - 绘制 episode 奖励曲线
   - 使用滑动平均平滑曲线

4. **可视化策略**：
   - 定期评估策略（关闭探索）
   - 录制视频观察智能体行为


## 9. 总结

### 9.1 核心要点

1. **DQN = Q-Learning + 深度神经网络 + 经验回放 + 目标网络**

2. **经验回放**：
   - 打破样本相关性
   - 提高样本效率
   - 稳定训练

3. **目标网络**：
   - 稳定 TD 目标
   - 减少震荡
   - 提高收敛性

4. **DQN 的成功**：
   - 首次在 Atari 游戏上达到人类水平
   - 开启了深度强化学习的时代

5. **DQN 的局限**：
   - 高估偏差
   - 样本效率低
   - 无法处理连续动作
   - 训练不稳定

### 9.2 下节预告

下节课我们将学习 **DQN 的改进算法**：

- **Double DQN**：解决高估偏差
- **Dueling DQN**：分离状态价值和优势函数
- **Prioritized Experience Replay**：优先采样重要经验
- **Rainbow DQN**：集成多种改进

### 9.3 扩展阅读

**必读论文：**
1. Mnih et al. (2013). "Playing Atari with Deep Reinforcement Learning"
2. Mnih et al. (2015). "Human-level control through deep reinforcement learning" (Nature)

**推荐资源：**
- David Silver RL Course Lecture 6: Value Function Approximation
- Spinning Up in Deep RL: DQN
- CleanRL DQN 实现：https://github.com/vwxyzjn/cleanrl
