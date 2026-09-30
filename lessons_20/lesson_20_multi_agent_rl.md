# Lesson 20: 多智能体 RL 简介 (Multi-Agent RL)

## 1. 从单智能体到多智能体

### 1.1 为什么需要多智能体？

到目前为止，我们所有算法都假设**只有一个智能体**在环境中学习。但现实世界中，许多场景涉及多个决策者：

- **自动驾驶**：多辆车在道路上交互
- **机器人协作**：多个机器人协作搬运物体
- **游戏 AI**：多个玩家对抗（如 StarCraft、Dota2）
- **经济系统**：多个交易者在市场中博弈
- **通信网络**：多个节点共享资源

### 1.2 MARL 的形式化

**多智能体 MDP（Markov Game / Stochastic Game）：**

$$\mathcal{G} = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}_{i \in \mathcal{N}}, \mathcal{T}, \{R_i\}_{i \in \mathcal{N}}, \gamma \rangle$$

其中：
- $\mathcal{N} = \{1, 2, ..., n\}$：智能体集合
- $\mathcal{S}$：全局状态空间
- $\mathcal{A}_i$：智能体 $i$ 的动作空间
- $\mathcal{T}: \mathcal{S} \times \mathcal{A}_1 \times ... \times \mathcal{A}_n \to \Delta(\mathcal{S})$：状态转移函数
- $R_i: \mathcal{S} \times \mathcal{A}_1 \times ... \times \mathcal{A}_n \to \mathbb{R}$：智能体 $i$ 的奖励函数
- $\gamma$：折扣因子

**关键区别：**
- 转移和奖励依赖于**所有智能体的联合动作**
- 每个智能体有自己的奖励函数
- 环境对每个智能体来说都是**非平稳的**（因为其他智能体也在学习）


## 2. MARL 的基本范式

### 2.1 按合作关系分类

**1. 完全合作（Fully Cooperative）**

$$R_1 = R_2 = ... = R_n = R$$

- 所有智能体共享同一个奖励
- 目标：最大化团队总回报
- 例子：多机器人搬运、组队游戏

**2. 完全竞争（Fully Competitive / Zero-Sum）**

$$\sum_{i=1}^{n} R_i = 0$$

- 一个智能体的收益等于其他智能体的损失
- 典型：两人零和博弈
- 例子：围棋、象棋、格斗游戏

**3. 混合（Mixed / General-Sum）**

- 奖励既不完全相同，也不零和
- 智能体之间既有合作也有竞争
- 例子：交通系统、经济博弈

### 2.2 按信息结构分类

**1. 完全可观测（Full Observability）**

- 每个智能体都能看到全局状态
- 对应 Markov Game

**2. 部分可观测（Partial Observability）**

- 每个智能体只能看到局部观测 $o_i$
- 对应 Dec-POMDP（分散化部分可观测 MDP）

$$o_i = O_i(s) \quad \text{观测函数}$$


## 3. MARL 的核心挑战

### 3.1 非平稳性（Non-Stationarity）

**问题：** 从单个智能体的视角看，环境是非平稳的。

- 智能体 $i$ 的最优策略取决于其他智能体的策略
- 其他智能体也在学习，策略不断变化
- 违反了 MDP 的马尔可夫性假设

**影响：**
- 单智能体 RL 算法的收敛性保证不再成立
- Q-learning 在多智能体设置中可能不收敛

### 3.2 信用分配（Credit Assignment）

**问题：** 在合作场景中，如何判断每个智能体对团队奖励的贡献？

- 团队获得奖励 $R$，但哪个智能体的动作是关键的？
- 简单的共享奖励可能导致"搭便车"问题
- "懒惰智能体"问题（lazy agent problem）

### 3.3 联合动作空间爆炸

**问题：** 联合动作空间随智能体数量指数增长。

$$|\mathcal{A}| = |\mathcal{A}_1| \times |\mathcal{A}_2| \times ... \times |\mathcal{A}_n|$$

- 3 个智能体，每个 10 个动作 → 联合动作空间 $10^3 = 1000$
- 10 个智能体 → $10^{10}$ = 100亿

### 3.4 均衡概念

**纳什均衡（Nash Equilibrium）：**

没有任何智能体能通过单方面改变策略来提高自身收益。

$$\forall i, \forall \pi_i': V_i(\pi_i^*, \pi_{-i}^*) \geq V_i(\pi_i', \pi_{-i}^*)$$

- 纳什均衡不一定是最优解
- 可能存在多个纳什均衡
- 找到纳什均衡本身就是困难问题


## 4. 独立学习者（Independent Learners）

### 4.1 Independent Q-Learning (IQL)

**最简单的方法：忽略其他智能体的存在**

每个智能体独立运行自己的 Q-learning：

$$Q_i(s, a_i) \leftarrow Q_i(s, a_i) + \alpha \left[ r_i + \gamma \max_{a_i'} Q_i(s', a_i') - Q_i(s, a_i) \right]$$

**优点：**
- 实现简单
- 可扩展到大量智能体
- 每个智能体可以使用任何单智能体算法

**缺点：**
- 不考虑其他智能体的策略
- 环境非平稳 → 不保证收敛
- 实际中在简单问题上效果还可以

### 4.2 Independent PPO

同理，每个智能体独立运行 PPO：

- 各自收集经验
- 各自更新策略
- 互不通信

虽然理论上有问题，但实践中常作为简单基线。


## 5. 自博弈（Self-Play）

### 5.1 核心思想

**用自己作为对手来训练：**

1. 训练一个智能体
2. 复制它作为对手
3. 与对手博弈并更新
4. 重复

**直觉：** 如果你能打败自己的所有历史版本，你就在不断进步。

### 5.2 经典成功案例

**1. TD-Gammon (Tesauro, 1995)**
- 西洋双陆棋
- 自博弈 + TD 学习
- 达到世界级水平

**2. AlphaGo / AlphaZero (Silver et al., 2016/2017)**
- 围棋、国际象棋、将棋
- MCTS + 自博弈
- 超越人类冠军

**3. OpenAI Five (2019)**
- Dota 2
- 自博弈 + PPO
- 击败世界冠军队

### 5.3 自博弈的实现

**简单自博弈：**

```
初始化策略 π
重复：
  1. 对手 π_opp ← 复制 π
  2. 用 π 与 π_opp 对弈，收集经验
  3. 用收集的经验更新 π
```

**带历史池的自博弈（Fictitious Self-Play）：**

```
初始化策略 π
初始化对手池 Π = {π}
重复：
  1. 从 Π 中随机选择对手 π_opp
  2. 用 π 与 π_opp 对弈，收集经验
  3. 用收集的经验更新 π
  4. 定期将 π 的快照加入 Π
```

**优势：**
- 避免"循环策略"问题（A 打败 B，B 打败 C，C 打败 A）
- 更鲁棒的策略

### 5.4 自博弈的挑战

1. **策略崩溃**：训练不稳定，策略可能退化
2. **评估困难**：Elo 评分等方法来追踪进步
3. **计算开销**：需要同时运行多个智能体


## 6. CTDE：集中训练，分散执行

### 6.1 核心思想

**CTDE（Centralized Training, Decentralized Execution）：**

- **训练时**：可以利用全局信息（所有智能体的观测、动作）
- **执行时**：每个智能体只用自己的局部观测做决策

$$\text{训练：} \quad \pi_i(a_i | o_i; \theta_i) \quad \text{用全局信息辅助训练}$$
$$\text{执行：} \quad a_i = \pi_i(o_i) \quad \text{只用局部观测}$$

**为什么这样设计？**
- 训练时有全局信息 → 解决信用分配问题
- 执行时只用局部信息 → 可扩展，通信开销小

### 6.2 CTDE 的具体实现

**1. 集中式 Critic（Centralized Critic）**

- Actor 只接收局部观测
- Critic 接收全局状态或所有观测

$$\pi_i(a_i | o_i) \quad \text{（分散策略）}$$
$$V(s) \text{ 或 } Q(s, a_1, ..., a_n) \quad \text{（集中价值函数）}$$

**2. 参数共享（Parameter Sharing）**

- 所有智能体共享同一个网络
- 输入中加入智能体 ID 来区分

$$\pi(a | o, \text{id})$$

**优势：**
- 减少参数量
- 隐式共享经验
- 适合同质智能体


## 7. 值分解方法

### 7.1 VDN（Value Decomposition Networks）

**核心思想：** 团队 Q 值等于个体 Q 值之和。

$$Q_{tot}(s, \mathbf{a}) = \sum_{i=1}^{n} Q_i(o_i, a_i)$$

**约束：** 加法分解

**优点：**
- 简单，易于实现
- 保证 IGM（Individual-Global-Max）：
  $$\arg\max_{\mathbf{a}} Q_{tot} = \left( \arg\max_{a_1} Q_1, ..., \arg\max_{a_n} Q_n \right)$$

**缺点：**
- 假设太强，表达能力有限
- 无法表示需要协调的任务

### 7.2 QMIX

**核心思想：** 团队 Q 值是个体 Q 值的**单调混合**。

$$Q_{tot}(s, \mathbf{a}) = f_{\text{mix}}(Q_1, Q_2, ..., Q_n; s)$$

**约束：** 混合网络的权重非负

$$\frac{\partial Q_{tot}}{\partial Q_i} \geq 0, \quad \forall i$$

**实现：**
- 混合网络（Mixing Network）由超网络（Hypernetwork）生成
- 超网络以全局状态 $s$ 为输入
- 生成混合网络的（非负）权重

**优势：**
- 比 VDN 表达能力更强
- 仍保证 IGM 条件
- 在 SMAC（StarCraft Multi-Agent Challenge）上效果显著


## 8. MAPPO

### 8.1 Multi-Agent PPO

**MAPPO（Yu et al., 2022）是 PPO 在多智能体设置中的扩展：**

**关键设计：**

1. **集中式价值函数**：
   $$V(s) \quad \text{以全局状态为输入}$$

2. **分散式策略**：
   $$\pi_i(a_i | o_i) \quad \text{以局部观测为输入}$$

3. **参数共享（可选）**：
   - 所有智能体共享 Actor 和 Critic 参数
   - 用智能体 ID 区分

### 8.2 MAPPO 算法

```
初始化：
  - 共享策略 π_θ(a|o, id)
  - 共享价值函数 V_φ(s)

对于每个 epoch：
  1. 数据收集：
     对于每个时间步：
       对于每个智能体 i：
         a_i ~ π_θ(·|o_i, id_i)
       执行联合动作，获得奖励和下一状态
  
  2. 计算优势：
     对于每个智能体 i：
       用 GAE 计算 A_i(s, a_i)（使用集中式 V(s)）
  
  3. 策略更新（PPO-Clip）：
     对于每个智能体 i：
       L_i = E[min(r_i A_i, clip(r_i, 1-ε, 1+ε) A_i)]
     L = Σ_i L_i / n
     更新 θ
  
  4. 价值更新：
     L_V = E[Σ_i (V_φ(s) - R_i)²]
     更新 φ
```

### 8.3 MAPPO 的优势

1. **简单有效**：基于成熟的 PPO，易于实现和调试
2. **可扩展**：参数共享使其适用于大量智能体
3. **强大性能**：在 SMAC、MPE 等基准上接近或超过 QMIX

**实践发现：**
- 参数共享在同质智能体中效果好
- 集中式 Critic 比分散式 Critic 更稳定
- 适当的输入归一化很重要


## 9. 通信与涌现

### 9.1 学习通信（Learned Communication）

**CommNet（Sukhbaatar et al., 2016）：**
- 智能体之间传递连续消息
- 消息通过平均池化聚合

**TarMAC（Das et al., 2019）：**
- 基于注意力的通信
- 智能体学习"向谁发送什么信息"

### 9.2 涌现行为

**多智能体系统中常观察到的涌现行为：**

1. **分工（Division of Labor）**：
   - 智能体自发地承担不同角色
   - 无需显式设计角色

2. **协调（Coordination）**：
   - 智能体学会在时间和空间上协调
   - 如：一个吸引注意力，另一个从背后进攻

3. **通信（Communication）**：
   - 智能体发展出简单的"语言"
   - 用于共享信息和协调行为


## 10. 实用环境与工具

### 10.1 常用多智能体环境

| 环境 | 类型 | 智能体数 | 特点 |
|------|------|----------|------|
| **MPE** | 混合 | 2-6 | 简单连续空间，适合调试 |
| **SMAC** | 合作 | 2-27 | StarCraft 微操 |
| **Hanabi** | 合作 | 2-5 | 不完全信息 |
| **MuJoCo Multi-Agent** | 合作 | 2-6 | 连续控制 |
| **Overcooked** | 合作 | 2 | 人机协作 |
| **PettingZoo** | 多种 | 多种 | 统一 API |

### 10.2 常用库

- **PettingZoo**：多智能体环境的 Gymnasium 接口
- **EPyMARL**：MARL 算法库（QMIX, MAPPO 等）
- **MARLlib**：更全面的 MARL 算法库


## 11. 总结

### 11.1 核心要点

1. **MARL 范式**：
   - 合作、竞争、混合
   - 完全可观测 vs 部分可观测

2. **核心挑战**：
   - 非平稳性、信用分配、联合动作空间爆炸

3. **关键方法**：
   - 独立学习者（IQL）：简单但有限
   - 自博弈（Self-Play）：竞争场景的利器
   - CTDE：训练用全局信息，执行用局部信息
   - 值分解（VDN, QMIX）：分解团队 Q 值
   - MAPPO：简单有效的多智能体 PPO

4. **实用建议**：
   - 从 IQL 开始作为基线
   - MAPPO 是强大且易用的方法
   - 合作场景考虑 QMIX
   - 竞争场景考虑自博弈

### 11.2 下节预告

下节课我们将学习 **离线 RL（Offline RL）**：
- 离线 RL 基本概念
- CQL（Conservative Q-Learning）
- IQL（Implicit Q-Learning）
- Decision Transformer

### 11.3 扩展阅读

**必读论文：**
1. Tan (1993). "Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents"
2. Silver et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (AlphaZero)
3. Rashid et al. (2018). "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning"
4. Yu et al. (2022). "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (MAPPO)

**推荐资源：**
- PettingZoo 文档和教程
- MARL 综述：A Survey of Multi-Agent Reinforcement Learning (Busoniu et al.)
- Berkeley CS285 Multi-Agent RL lectures
