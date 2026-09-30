# Lesson 19: 模型基础 RL 简介 (Model-Based RL)

## 1. Model-Free vs Model-Based RL

### 1.1 回顾：Model-Free RL

到目前为止，我们学习的所有算法都是 **Model-Free（无模型）** 的：

- DQN, TD3, SAC, PPO, A2C, REINFORCE...
- 直接学习策略 $\pi(a|s)$ 或价值函数 $Q(s, a)$
- 不学习环境动态模型

**优点：**
- 实现简单
- 适用于复杂环境
- 不需要建模环境

**缺点：**
- 样本效率低（需要大量交互）
- 无法进行规划（planning）
- 无法利用模型进行想象（imagination）

### 1.2 Model-Based RL 的核心思想

**Model-Based RL 学习环境的动态模型：**

$$s_{t+1} = f(s_t, a_t) \quad \text{或} \quad s_{t+1} \sim p(s_{t+1} | s_t, a_t)$$

**有了模型后可以：**

1. **规划（Planning）**：
   - 在模型中搜索最优动作序列
   - 不需要与真实环境交互

2. **想象（Imagination）**：
   - 在模型中生成虚拟经验
   - 用虚拟经验训练策略

3. **样本效率**：
   - 减少与真实环境的交互
   - 特别适合交互成本高的场景

### 1.3 Model-Based RL 的挑战

**1. 模型误差（Model Error）**

- 学到的模型不完美
- 误差会累积（compounding error）
- 长期规划时误差放大

**2. 计算开销**

- 学习模型需要额外计算
- 在模型中规划也需要计算

**3. 适用性**

- 某些环境难以建模（如复杂的物理系统）
- 高维状态空间建模困难


## 2. Dyna 架构

### 2.1 Dyna 的核心思想

**Dyna（Sutton, 1990）是最早的 Model-Based RL 框架：**

**核心思想：结合真实经验和模拟经验**

1. **与环境交互**：收集真实经验
2. **学习模型**：用真实经验学习环境模型
3. **规划**：用模型生成模拟经验
4. **学习策略**：用真实经验 + 模拟经验学习策略

### 2.2 Dyna-Q 算法

**Dyna-Q 是 Dyna 的一个简单实现：**

```
初始化：
  - Q 表 Q(s, a)
  - 模型 Model(s, a) → (r, s')
  - 经验池 Memory

对于每个时间步：
  1. 与环境交互：
     a) 选择动作：a = ε-greedy(Q(s, ·))
     b) 执行动作：s' = env.step(a), 获得 r
     c) 更新 Q 表（Q-learning）：
        Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]
     d) 更新模型：Model(s, a) ← (r, s')
     e) 存储经验：Memory.add((s, a, r, s'))
  
  2. 规划（Planning）：
     重复 n 次：
       a) 从 Memory 随机采样 (s, a)
       b) 用模型预测：(r, s') = Model(s, a)
       c) 更新 Q 表：
          Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]
```

**关键点：**
- 真实经验用于学习 Q 和模型
- 模拟经验用于学习 Q
- 规划步数 $n$ 控制模拟经验的数量

### 2.3 Dyna-Q 的优势

**1. 样本效率**

- 每个真实经验可以生成多个模拟经验
- 加速学习

**2. 简单**

- 基于表格 Q-learning
- 模型也是表格形式（确定性）

**3. 有效**

- 在简单环境中表现优异
- 如迷宫、网格世界

### 2.4 Dyna-Q 的局限

**1. 仅适用于离散状态-动作空间**

- 表格方法无法扩展到连续空间

**2. 确定性模型**

- 假设环境是确定性的
- 无法处理随机环境

**3. 模型误差**

- 如果模型不准确，会学到错误的策略


## 3. 深度 Model-Based RL

### 3.1 用神经网络建模

**现代 Model-Based RL 使用神经网络作为模型：**

$$s_{t+1} = f_\theta(s_t, a_t) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \Sigma)$$

或者学习分布：

$$p_\theta(s_{t+1} | s_t, a_t)$$

**常见模型类型：**

1. **确定性模型**：$s_{t+1} = f_\theta(s_t, a_t)$
2. **高斯模型**：$s_{t+1} \sim \mathcal{N}(\mu_\theta(s_t, a_t), \Sigma_\theta(s_t, a_t))$
3. **集成模型**：多个模型的集成，估计不确定性

### 3.2 世界模型（World Model）

**Ha & Schmidhuber (2018) 提出了世界模型框架：**

**核心组件：**

1. **视觉模型（V）**：
   - 编码器：将高维观测（如图像）压缩到低维表示
   - 解码器：从低维表示重建观测

2. **记忆模型（M）**：
   - RNN/LSTM：预测下一个状态的表示
   - $z_{t+1} = M(z_t, a_t, h_t)$

3. **控制器（C）**：
   - 策略网络：在低维表示空间中学习策略
   - $a_t = C(z_t, h_t)$

**训练流程：**

1. 收集真实经验
2. 训练 V 和 M（无监督学习）
3. 在学到的世界模型中训练 C（强化学习）

**优势：**
- 在低维表示空间中学习
- 可以在模型中进行大量模拟
- 样本效率高

### 3.3 MBPO (Model-Based Policy Optimization)

**Janner et al. (2019) 提出了 MBPO：**

**核心思想：短期模型展开 + Model-Free 算法**

**算法流程：**

```
初始化：
  - 策略 π_θ
  - 动态模型 p_φ(s'|s,a)
  - 真实经验池 D_env
  - 模型经验池 D_model

对于每个 epoch：
  1. 与环境交互：
     用当前策略收集真实经验 → D_env
  
  2. 训练模型：
     用 D_env 训练动态模型 p_φ
  
  3. 生成模拟经验：
     对于 D_env 中的每个状态 s：
       用模型展开 k 步（k=1-5）：
         a ~ π_θ(·|s)
         s' ~ p_φ(·|s,a)
         存储 (s, a, r, s') → D_model
  
  4. 训练策略：
     用 D_env + D_model 训练策略（如 SAC）
```

**关键技术：**

1. **短期展开（Short Horizon）**：
   - 只展开 1-5 步
   - 减少模型误差累积

2. **集成模型（Ensemble）**：
   - 训练多个模型
   - 估计模型不确定性

3. **分支展开（Branching Rollout）**：
   - 从真实状态开始展开
   - 避免在模型生成的状态上继续展开

**优势：**
- 样本效率高（比 SAC 提高 10-100 倍）
- 性能好（接近或超过 Model-Free）
- 实现相对简单

### 3.4 DreamerV3

**Hafner et al. (2023) 提出了 DreamerV3：**

**核心思想：在潜在空间中学习世界模型和策略**

**主要组件：**

1. **世界模型**：
   - 编码器：观测 → 潜在状态
   - 动态模型：预测潜在状态转移
   - 奖励模型：预测奖励
   - 继续模型：预测 episode 是否结束

2. **Actor-Critic**：
   - 在潜在空间中学习策略和价值函数
   - 使用想象的轨迹训练

**训练流程：**

1. 收集真实经验
2. 训练世界模型（重建观测、预测奖励）
3. 在世界模型中想象轨迹
4. 用想象的轨迹训练 Actor-Critic

**优势：**
- 统一框架（适用于多种任务）
- 样本效率极高
- 可以处理视觉输入


## 4. Model-Based RL 的关键技术

### 4.1 模型学习

**监督学习目标：**

$$L(\theta) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}} \left[ \|s' - f_\theta(s, a)\|^2 \right]$$

或者负对数似然：

$$L(\theta) = -\mathbb{E}_{(s,a,s') \sim \mathcal{D}} \left[ \log p_\theta(s' | s, a) \right]$$

**技巧：**
- 归一化输入输出
- 预测差分：$\Delta s = s' - s$
- 使用集成模型

### 4.2 模型不确定性估计

**为什么需要不确定性？**

- 模型在某些区域可能不准确
- 需要知道何时信任模型

**方法：**

1. **集成模型（Ensemble）**：
   - 训练多个模型
   - 方差表示不确定性

2. **贝叶斯神经网络**：
   - 学习参数分布
   - 计算开销大

3. **Dropout**：
   - 训练时使用 Dropout
   - 推理时多次采样估计不确定性

### 4.3 模型误差缓解

**1. 短期展开**

- 只展开少数步（1-5 步）
- 减少误差累积

**2. 模型选择**

- 在不确定性高的区域少用模型
- 在不确定性低的区域多用模型

**3. 对抗训练**

- 训练鲁棒的策略
- 对模型误差不敏感

### 4.4 规划方法

**1. 随机射击（Random Shooting）**

- 采样多个动作序列
- 在模型中评估
- 选择最好的序列

**2. 交叉熵方法（CEM）**

- 迭代优化动作序列分布
- 采样 → 评估 → 更新分布

**3. 模型预测控制（MPC）**

- 在每个时间步规划
- 只执行第一个动作
- 重新规划（receding horizon）


## 5. Model-Based vs Model-Free

| 特性 | Model-Free | Model-Based |
|------|------------|-------------|
| **样本效率** | 低 | 高 |
| **渐近性能** | 高 | 中（受模型误差限制） |
| **计算开销** | 低 | 高 |
| **实现复杂度** | 低 | 高 |
| **适用场景** | 通用 | 交互成本高的场景 |
| **典型算法** | SAC, PPO, TD3 | MBPO, Dreamer |

**何时使用 Model-Based？**

- 真实环境交互成本高（如机器人）
- 需要快速适应新任务
- 可以获得准确的模型

**何时使用 Model-Free？**

- 环境交互成本低
- 追求最优性能
- 环境难以建模


## 6. 实践建议

### 6.1 从简单开始

1. **先实现 Dyna-Q**：
   - 理解 Model-Based 的基本思想
   - 在简单环境（如网格世界）测试

2. **再尝试深度方法**：
   - 学习神经网络模型
   - 在连续控制任务测试

### 6.2 模型学习技巧

1. **数据归一化**：
   - 输入输出都归一化
   - 提高训练稳定性

2. **预测差分**：
   - 预测 $\Delta s = s' - s$ 而不是 $s'$
   - 更容易学习

3. **集成模型**：
   - 训练 5-10 个模型
   - 估计不确定性

### 6.3 调试技巧

1. **监控模型误差**：
   - 在验证集上评估模型
   - 误差应该逐渐减小

2. **可视化预测**：
   - 对比真实轨迹和预测轨迹
   - 检查模型是否合理

3. **短期展开**：
   - 从 1 步开始
   - 逐渐增加展开步数


## 7. 总结

### 7.1 核心要点

1. **Model-Based RL**：
   - 学习环境动态模型
   - 用模型生成模拟经验
   - 提高样本效率

2. **Dyna 架构**：
   - 结合真实经验和模拟经验
   - 简单有效的框架

3. **现代方法**：
   - MBPO：短期展开 + Model-Free
   - Dreamer：潜在空间世界模型
   - 样本效率提高 10-100 倍

4. **关键挑战**：
   - 模型误差累积
   - 计算开销
   - 实现复杂度

### 7.2 Model-Based RL 的未来

**当前趋势：**
- 更好的模型架构（Transformer）
- 更好的不确定性估计
- 与 Model-Free 的混合方法

**应用前景：**
- 机器人控制
- 自动驾驶
- 游戏 AI
- 工业控制

### 7.3 下节预告

下节课我们将学习 **多智能体 RL 简介**：
- MARL 基本范式
- 自博弈（Self-play）
- CTDE（Centralized Training, Decentralized Execution）
- MAPPO

### 7.4 扩展阅读

**必读论文：**
1. Sutton (1990). "Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming" (Dyna)
2. Ha & Schmidhuber (2018). "World Models"
3. Janner et al. (2019). "When to Trust Your Model: Model-Based Policy Optimization" (MBPO)
4. Hafner et al. (2023). "Mastering Diverse Domains through World Models" (DreamerV3)

**推荐资源：**
- Spinning Up in Deep RL: Model-Based RL
- Berkeley CS285: Model-Based RL lectures
- DreamerV3 官方实现
