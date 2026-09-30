# 强化学习系统学习路线 (Reinforcement Learning Curriculum)

> 适用对象：深度学习背景，数学基础扎实，熟悉 PyTorch，目标掌握 RLHF/DPO/GRPO
> 每日学习时间：约 2 小时（讲义 + 代码实践）
> 预计总时长：8-10 周（约 30 课时）

---

## 阶段一：基础理论（第 1-2 周，6 课时）

掌握强化学习的数学框架和核心思想，建立从"环境-智能体"交互到价值函数的完整认知。

| 课时 | 主题 | 核心内容 | 代码实践 | 预计时间 |
|------|------|----------|----------|----------|
| 01 | RL 概览与 MDP | RL 定义、与监督/无监督学习区别、MDP 五元组 (S,A,P,R,γ)、Bellman 方程 | 实现 GridWorld 环境，手动计算状态价值 | 2h |
| 02 | Bellman 方程深入 | Bellman 期望方程、Bellman 最优方程、状态价值 V(s) vs 动作价值 Q(s,a)、收缩映射 | 编程求解小规模 MDP 的精确解（矩阵求逆法） | 2h |
| 03 | 动态规划：策略评估与策略迭代 | 迭代策略评估、策略改进定理、策略迭代算法 | 实现策略迭代解 Frozen Lake | 2h |
| 04 | 动态规划：价值迭代 | 价值迭代算法、异步 DP、DP 的局限性 | 实现价值迭代，对比策略迭代效率 | 2h |
| 05 | Monte Carlo 方法 | MC 策略评估、首次访问 vs 每次访问、MC 控制、探索-利用、ε-greedy | 实现 MC 控制玩 Blackjack (OpenAI Gym) | 2h |
| 06 | 时序差分学习 (TD) | TD(0)、TD 与 MC 对比、偏差-方差权衡、TD(λ)、资格迹 | 实现 TD(0) 和 TD(λ)，对比学习曲线 | 2h |

---

## 阶段二：经典控制算法（第 3-4 周，6 课时）

掌握 model-free 控制的核心算法，理解 on-policy vs off-policy 的本质区别。

| 课时 | 主题 | 核心内容 | 代码实践 | 预计时间 |
|------|------|----------|----------|----------|
| 07 | SARSA | On-policy TD 控制、SARSA 更新规则、SARSA(λ) | 实现 SARSA 解 CliffWalking | 2h |
| 08 | Q-Learning | Off-policy TD 控制、Q-Learning 更新规则、与 SARSA 对比 | 实现 Q-Learning，与 SARSA 在 CliffWalking 上对比 | 2h |
| 09 | 函数逼近基础 | 为什么需要函数逼近、线性逼近、特征工程、收敛性分析 | 用线性函数逼近实现 Mountain Car | 2h |
| 10 | 策略梯度入门 (REINFORCE) | 策略参数化、策略梯度定理推导、REINFORCE 算法、基线减方差 | 实现 REINFORCE 解 CartPole | 2h |
| 11 | Actor-Critic | 优势函数 A(s,a)、A2C 算法、GAE (Generalized Advantage Estimation) | 实现 A2C 解 CartPole，对比 REINFORCE | 2h |
| 12 | 探索策略专题 | ε-greedy 局限、UCB、Thompson Sampling、好奇心驱动探索、RND | 实现多臂老虎机比较不同探索策略 | 2h |

---

## 阶段三：深度强化学习（第 5-7 周，10 课时）

用深度神经网络驱动 RL，掌握现代 Deep RL 的核心算法族。

| 课时 | 主题 | 核心内容 | 代码实践 | 预计时间 |
|------|------|----------|----------|----------|
| 13 | DQN | 经验回放、目标网络、DQN 训练流程、Atari 突破 | PyTorch 实现 DQN 解 CartPole | 2h |
| 14 | DQN 进阶 | Double DQN、Dueling DQN、Prioritized Experience Replay、Rainbow | 实现 Double DQN + Dueling DQN | 2h |
| 15 | PPO (上) | 信赖域方法思想、TRPO 简介、PPO-Clip 目标函数推导 | 从零实现 PPO-Clip | 2.5h |
| 16 | PPO (下) | PPO 实现细节、超参调优、PPO 在实践中的主导地位 | 用 PPO 解连续控制任务 (Pendulum) | 2h |
| 17 | DDPG & TD3 | 确定性策略梯度、DDPG 算法、Twin Delayed DDPG (TD3) | 实现 DDPG/TD3 解 Pendulum | 2h |
| 18 | SAC (Soft Actor-Critic) | 最大熵 RL 框架、温度参数自动调节、SAC 算法 | 实现 SAC，对比 TD3 | 2h |
| 19 | 模型基础 RL 简介 | Dyna 架构、世界模型 (World Model)、MBPO、DreamerV3 概览 | 实现简单 Dyna-Q | 2h |
| 20 | 多智能体 RL 简介 | MARL 基本范式、自博弈 (Self-play)、CTDE、MAPPO | 实现简单的多智能体博弈 | 2h |
| 21 | 离线 RL | Batch RL 问题、CQL、IQL、Decision Transformer | 实现简单的离线 Q-learning + CQL | 2h |
| 22 | RL 工程实践 | 超参敏感性、reward shaping、调试技巧、常用库 (Stable-Baselines3, CleanRL) | 用 Stable-Baselines3 快速训练 + 用 W&B 记录实验 | 2h |

---

## 阶段四：RL for LLMs — RLHF 与前沿（第 8-10 周，8 课时）

这是你的最终目标区域，将 RL 与大模型训练深度结合。

| 课时 | 主题 | 核心内容 | 代码实践 | 预计时间 |
|------|------|----------|----------|----------|
| 23 | 从 RL 到 LLM 对齐 | LLM 训练三阶段 (Pretrain→SFT→RLHF)、为什么需要对齐、奖励建模 | 分析 reward model 的数据格式和训练流程 | 2h |
| 24 | RLHF 详解 | InstructGPT 论文精读、PPO 在语言模型上的应用、KL 惩罚、reward hacking | 用 TRL 库对小模型做 RLHF 微调 | 2.5h |
| 25 | DPO (Direct Preference Optimization) | DPO 推导（从 RLHF 目标到隐式奖励）、DPO vs RLHF 对比 | 用 TRL 实现 DPO 训练 | 2h |
| 26 | DPO 变体与发展 | IPO、KTO、ORPO、SimPO、各方法优劣对比 | 实现 SimPO 训练对比 DPO | 2h |
| 27 | GRPO (Group Relative Policy Optimization) | DeepSeek 的 GRPO 方法、无需 value model、组内相对排序 | 实现简化版 GRPO | 2h |
| 28 | RL 用于推理 (RL for Reasoning) | OpenAI o1/o3 思路、过程奖励模型 (PRM)、MCTS + LLM、STaR/Quiet-STaR | 实现简单的过程奖励训练 | 2h |
| 29 | 在线 RL 与 LLM Agent | LLM 作为策略网络、工具调用 RL、WebAgent、SWE-agent 思路 | 构建简单的 LLM Agent + RL 反馈循环 | 2h |
| 30 | 前沿综述与展望 | Scaling RL、Constitutional AI、RLVR、多模态 RL、开源生态 | 复现一篇近期论文的核心实验 | 2.5h |

---

## 推荐资源

### 教材
- **Sutton & Barto** "Reinforcement Learning: An Introduction" (2nd ed.) — 经典圣经，阶段一二必读
- **Szepesvari** "Algorithms for Reinforcement Learning" — 数学推导更严谨的补充

### 课程
- **David Silver RL Course** (UCL, 2015) — 最经典的 RL 入门课，10 讲
- **CS285 (UC Berkeley)** — Deep RL 最佳课程，Sergey Levine 主讲
- **Hugging Face Deep RL Course** — 免费，实操性强

### 代码库
- **CleanRL** — 单文件实现各算法，最适合学习
- **Stable-Baselines3** — 工程级实现，适合快速实验
- **TRL (Transformer Reinforcement Learning)** — Hugging Face 出品，RLHF/DPO/GRPO 必备
- **OpenRLHF** — 分布式 RLHF 训练框架

### 论文 (按阶段)
- 阶段三：DQN (2015), A3C (2016), PPO (2017), SAC (2018), DreamerV3 (2023)
- 阶段四：InstructGPT (2022), DPO (2023), GRPO (2024), DeepSeek-R1 (2025)

---

## 学习建议

1. **数学推导要亲手做** — 数学基础很重要，Bellman 方程、策略梯度定理等要自己推一遍
2. **代码要从零写** — 前期不要用库，理解每行代码的含义；后期可以用 SB3/TRL 加速
3. **画学习曲线** — 每个算法都要画 reward curve，建立对收敛行为的直觉
4. **关联已有知识** — 语音增强中的序列决策、损失函数设计思想与 RL 有很多共通之处
5. **阶段四是重点** — 作为目标方向，阶段四的论文要精读，代码要能跑通

---

## Exercise 代码配置规范

每个 lesson 的 exercise 代码提供两个版本：

1. **`lesson_XX_exercise.py`** — 完整参考实现（包含所有 TODO 的答案）
2. **`lesson_XX_exercise_practice.py`** — 练习版本（TODO 部分只有 `pass`，需要自己实现）

**推荐学习流程**：
1. 阅读讲义 `lesson_XX_xxx.md`
2. 在 `lesson_XX_exercise_practice.py` 中独立实现所有 TODO
3. 运行验证，对照 `lesson_XX_exercise.py` 中的参考实现进行学习和改进

这样既保留了"动手写"的学习价值，又有完整参考可以验证正确性。
