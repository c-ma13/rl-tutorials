# Lesson 22: RL 工程实践

## 1. 从研究到工程

### 1.1 理论与实践的鸿沟

**前 21 课我们学了什么：**
- 算法原理（Q-learning, DQN, PPO, SAC...）
- 数学推导（Bellman 方程、策略梯度定理...）
- 简单实现（CartPole, GridWorld...）

**实际应用中的挑战：**
- 算法不收敛或收敛很慢
- 超参数难以调优
- 训练不稳定，方差大
- 难以复现论文结果
- 调试困难，不知道哪里出错

**本课目标：**
- 掌握 RL 工程实践技巧
- 学会使用成熟的 RL 库
- 建立系统的实验流程
- 提高调试和优化效率

### 1.2 RL 工程的核心要素

```
成功的 RL 项目 = 算法 + 工程 + 实验管理
```

1. **算法实现**：正确、高效、可复现
2. **超参数调优**：系统化搜索和优化
3. **实验管理**：版本控制、日志记录、可视化
4. **调试技巧**：快速定位问题
5. **工具链**：使用成熟的库和框架


## 2. 超参数敏感性

### 2.1 RL 中的超参数

**RL 算法通常有大量超参数：**

| 类别 | 超参数 | 典型范围 |
|------|--------|----------|
| **学习率** | actor_lr, critic_lr | 1e-5 ~ 1e-2 |
| **网络结构** | hidden_sizes, activation | [64,64] ~ [512,512] |
| **训练** | batch_size, num_epochs | 32~512, 3~20 |
| **探索** | epsilon, entropy_coef | 0.01~1.0, 0.001~0.1 |
| **折扣** | gamma | 0.9~0.999 |
| **PPO** | clip_range, gae_lambda | 0.1~0.3, 0.9~0.99 |
| **SAC** | alpha (temperature) | 0.01~0.5 |
| **缓冲区** | buffer_size | 1e4~1e6 |

### 2.2 超参数的影响

**学习率（Learning Rate）：**
- **太大**：训练不稳定，策略崩溃
- **太小**：收敛慢，可能陷入局部最优
- **技巧**：使用学习率衰减或自适应优化器（Adam）

**批大小（Batch Size）：**
- **太大**：梯度估计准确，但样本效率低
- **太小**：梯度噪声大，训练不稳定
- **技巧**：PPO 通常用 64-256，SAC 用 256-512

**折扣因子（Gamma）：**
- **接近 1**：长期规划，但训练慢
- **较小**：短视，但训练快
- **技巧**：长 episode 用 0.99，短 episode 用 0.95

**熵系数（Entropy Coefficient）：**
- **太大**：过度探索，策略随机
- **太小**：过早收敛，陷入局部最优
- **技巧**：从大到小衰减（如 0.1 → 0.01）

### 2.3 超参数调优策略

**1. 从默认值开始**

使用论文或库的默认超参数作为起点：
- Stable-Baselines3 提供了经过验证的默认值
- RL Baselines3 Zoo 提供了针对不同环境的最佳超参数

**2. 网格搜索（Grid Search）**

```python
learning_rates = [1e-4, 3e-4, 1e-3]
batch_sizes = [64, 128, 256]

for lr in learning_rates:
    for bs in batch_sizes:
        train(lr=lr, batch_size=bs)
```

**缺点**：组合爆炸，计算开销大

**3. 随机搜索（Random Search）**

```python
for trial in range(n_trials):
    lr = np.random.uniform(1e-5, 1e-2)
    bs = np.random.choice([32, 64, 128, 256])
    train(lr=lr, batch_size=bs)
```

**优势**：更高效，覆盖更广

**4. 贝叶斯优化（Bayesian Optimization）**

使用 Optuna、Hyperopt 等工具：
- 根据历史结果智能选择下一组超参数
- 平衡探索与利用

**5. 群体训练（Population-Based Training, PBT）**

- 同时训练多个智能体
- 动态调整超参数
- 性能差的复制性能好的超参数


## 3. Reward Shaping

### 3.1 为什么需要 Reward Shaping？

**稀疏奖励问题：**

```python
# 稀疏奖励：只有到达目标才有奖励
def sparse_reward(state):
    if state == goal:
        return 1.0
    else:
        return 0.0
```

**问题**：
- 智能体很难发现正奖励
- 探索效率低
- 训练时间长

**解决方案：Reward Shaping**

### 3.2 Reward Shaping 技术

**1. 距离奖励（Distance-Based Reward）**

```python
def shaped_reward(state, goal):
    distance = np.linalg.norm(state - goal)
    return -distance  # 越接近目标，奖励越高
```

**2. 进度奖励（Progress Reward）**

```python
def progress_reward(prev_state, curr_state, goal):
    prev_dist = np.linalg.norm(prev_state - goal)
    curr_dist = np.linalg.norm(curr_state - goal)
    return prev_dist - curr_dist  # 奖励接近目标的行为
```

**3. 子目标奖励（Subgoal Reward）**

```python
def subgoal_reward(state, subgoals):
    for subgoal in subgoals:
        if reached(state, subgoal):
            return 1.0
    return 0.0
```

**4. 潜在函数（Potential-Based Shaping）**

**理论保证（Ng et al., 1999）：**

$$F(s, s') = \gamma \Phi(s') - \Phi(s)$$

其中 $\Phi(s)$ 是潜在函数。这种 shaping 不改变最优策略。

**例子：**

```python
def potential(state, goal):
    return -np.linalg.norm(state - goal)

def shaped_reward(state, next_state, reward, gamma=0.99):
    F = gamma * potential(next_state, goal) - potential(state, goal)
    return reward + F
```

### 3.3 Reward Shaping 的陷阱

**1. 过度 Shaping**

- 智能体学会"作弊"，利用 shaped reward
- 例：机器人学会在目标附近晃动而不是真正到达

**2. 改变最优策略**

- 不当的 shaping 可能导致次优策略
- 使用 potential-based shaping 避免此问题

**3. 调试困难**

- Shaped reward 可能掩盖真实性能
- 始终监控原始任务的成功率


## 4. 调试技巧

### 4.1 RL 调试的挑战

**RL 调试比监督学习更难：**
- 非平稳性：数据分布随策略变化
- 延迟反馈：错误可能在很久后才显现
- 随机性：结果方差大
- 多因素：算法、环境、超参数都可能出错

### 4.2 系统化调试流程

**1. 验证环境**

```python
# 检查环境是否正确
env = gym.make('CartPole-v1')
obs = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")

# 随机策略测试
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
```

**2. 测试简单基线**

- 随机策略：性能下界
- 专家策略（如果有）：性能上界
- 简单启发式：合理性检查

**3. 过拟合单个 Episode**

```python
# 如果算法连一个 episode 都学不会，说明有 bug
# 用固定的 episode 训练，看能否过拟合
```

**4. 检查梯度**

```python
# 检查梯度是否合理
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
```

**5. 可视化学习曲线**

- 奖励曲线：是否上升？
- 损失曲线：是否下降？
- 熵：是否合理衰减？
- Q 值：是否合理？

### 4.3 常见问题与解决方案

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| **不学习** | 学习率太小 | 增大学习率 |
| | 奖励尺度不对 | 归一化奖励 |
| | 探索不足 | 增大探索率 |
| **训练不稳定** | 学习率太大 | 减小学习率 |
| | 批大小太小 | 增大批大小 |
| | 没有归一化 | 添加观测/奖励归一化 |
| **过拟合** | 网络太大 | 减小网络 |
| | 训练太久 | 早停 |
| **欠拟合** | 网络太小 | 增大网络 |
| | 训练不够 | 增加训练步数 |
| **Q 值爆炸** | 目标网络更新太快 | 减小 tau 或增大更新间隔 |
| | Gamma 太接近 1 | 减小 gamma |


## 5. 常用 RL 库

### 5.1 Stable-Baselines3 (SB3)

**特点：**
- PyTorch 实现
- 高质量、经过验证的算法
- 统一的 API
- 丰富的文档和示例

**支持的算法：**
- A2C, PPO
- DQN, DDPG, TD3, SAC
- HER (Hindsight Experience Replay)

**基本用法：**

```python
from stable_baselines3 import PPO

# 创建模型
model = PPO("MlpPolicy", "CartPole-v1", verbose=1)

# 训练
model.learn(total_timesteps=10000)

# 保存
model.save("ppo_cartpole")

# 加载
model = PPO.load("ppo_cartpole")

# 评估
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

### 5.2 CleanRL

**特点：**
- 单文件实现
- 易于理解和修改
- 注重可复现性
- 与 W&B 集成

**哲学：**
- 简单胜于复杂
- 透明胜于抽象
- 每个算法一个文件

### 5.3 RLlib (Ray)

**特点：**
- 分布式训练
- 支持大规模并行
- 多种算法
- 适合生产环境

### 5.4 库的选择

| 库 | 适用场景 |
|------|----------|
| **Stable-Baselines3** | 快速原型、标准任务、学习 |
| **CleanRL** | 理解算法、修改算法、研究 |
| **RLlib** | 大规模训练、生产部署 |
| **自己实现** | 深入理解、特殊需求 |


## 6. 实验管理：Weights & Biases

### 6.1 为什么需要实验管理？

**RL 实验的挑战：**
- 实验多：不同算法、超参数、环境
- 训练长：几小时到几天
- 结果随机：需要多次运行
- 难以比较：不同实验的结果

**实验管理工具的作用：**
- 自动记录超参数和指标
- 可视化学习曲线
- 比较不同实验
- 团队协作
- 可复现性

### 6.2 Weights & Biases (W&B) 基础

**安装：**

```bash
pip install wandb
```

**基本用法：**

```python
import wandb

# 初始化
wandb.init(
    project="my-rl-project",
    config={
        "learning_rate": 3e-4,
        "batch_size": 64,
        "gamma": 0.99,
    }
)

# 训练循环
for step in range(num_steps):
    # ... 训练代码 ...
    
    # 记录指标
    wandb.log({
        "reward": episode_reward,
        "loss": loss,
        "epsilon": epsilon,
    })

# 结束
wandb.finish()
```

### 6.3 W&B 高级功能

**1. 超参数扫描（Sweep）**

```yaml
# sweep.yaml
program: train.py
method: bayes
metric:
  name: reward
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.01
  batch_size:
    values: [32, 64, 128, 256]
```

```bash
wandb sweep sweep.yaml
wandb agent <sweep_id>
```

**2. 视频记录**

```python
# 记录 episode 视频
frames = []
obs = env.reset()
for _ in range(500):
    frames.append(env.render(mode='rgb_array'))
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        break

wandb.log({"video": wandb.Video(np.array(frames), fps=30)})
```

**3. 模型保存**

```python
# 保存最佳模型
if episode_reward > best_reward:
    model.save("best_model")
    wandb.save("best_model.zip")
```

### 6.4 SB3 + W&B 集成

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import wandb

class WandbCallback(BaseCallback):
    def __init__(self):
        super().__init__()
    
    def _on_step(self):
        # 记录指标
        if len(self.model.ep_info_buffer) > 0:
            wandb.log({
                "reward": self.model.ep_info_buffer[-1]['r'],
                "length": self.model.ep_info_buffer[-1]['l'],
            })
        return True

# 初始化 W&B
wandb.init(project="sb3-ppo")

# 训练
model = PPO("MlpPolicy", "CartPole-v1")
model.learn(total_timesteps=10000, callback=WandbCallback())

wandb.finish()
```


## 7. 可复现性

### 7.1 随机种子

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确定性算法（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 7.2 环境种子

```python
env.seed(42)
env.action_space.seed(42)
```

### 7.3 版本控制

**记录关键信息：**
- 代码版本（git commit hash）
- 库版本（requirements.txt）
- 超参数
- 环境配置

```python
import subprocess

git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
wandb.config.update({"git_hash": git_hash})
```


## 8. 性能优化

### 8.1 向量化环境

**并行收集经验：**

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# 创建多个环境并行运行
env = SubprocVecEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])

model = PPO("MlpPolicy", env)
model.learn(total_timesteps=10000)
```

**加速比：** 接近线性（4 个环境 ≈ 4 倍速度）

### 8.2 GPU 加速

```python
# 自动使用 GPU（如果可用）
model = PPO("MlpPolicy", "CartPole-v1", device="cuda")
```

### 8.3 经验回放优化

- 使用高效的缓冲区实现
- 优先经验回放（PER）
- 压缩存储（图像）


## 9. 实用建议

### 9.1 开始新项目的检查清单

- [ ] 理解任务和环境
- [ ] 选择合适的算法
- [ ] 从简单基线开始
- [ ] 使用成熟的库（SB3）
- [ ] 设置实验跟踪（W&B）
- [ ] 固定随机种子
- [ ] 记录所有超参数
- [ ] 多次运行取平均
- [ ] 可视化学习曲线
- [ ] 保存最佳模型

### 9.2 调试时的优先级

1. **先验证环境**：确保环境正确
2. **测试简单情况**：能否过拟合单个 episode？
3. **检查数据流**：观测、奖励、动作是否合理？
4. **验证算法**：使用已知有效的超参数
5. **逐步复杂化**：从简单到复杂

### 9.3 超参数调优的经验法则

1. **学习率**：从 3e-4 开始
2. **批大小**：从 64 或 128 开始
3. **网络大小**：从 [64, 64] 开始
4. **Gamma**：长 episode 用 0.99，短用 0.95
5. **先调学习率**：影响最大
6. **一次调一个**：便于理解影响


## 10. 总结

### 10.1 核心要点

1. **工程实践至关重要**：
   - 算法正确 ≠ 训练成功
   - 需要系统化的工程方法

2. **超参数敏感性**：
   - RL 对超参数非常敏感
   - 使用默认值 + 系统化搜索

3. **Reward Shaping**：
   - 稀疏奖励需要 shaping
   - 使用 potential-based shaping 保证理论性质

4. **调试技巧**：
   - 系统化流程
   - 从简单到复杂
   - 可视化一切

5. **工具链**：
   - Stable-Baselines3：快速原型
   - W&B：实验管理
   - 向量化环境：加速训练

6. **可复现性**：
   - 固定随机种子
   - 记录所有配置
   - 版本控制

### 10.2 下节预告

下节课我们将进入 **RLHF（人类反馈强化学习）**：
- RLHF 基本流程
- 奖励模型训练
- PPO 微调
- InstructGPT 和 ChatGPT

### 10.3 扩展阅读

**必读资源：**
1. Stable-Baselines3 文档：https://stable-baselines3.readthedocs.io/
2. RL Baselines3 Zoo：https://github.com/DLR-RM/rl-baselines3-zoo
3. Weights & Biases 文档：https://docs.wandb.ai/
4. Henderson et al. (2018). "Deep Reinforcement Learning that Matters"
5. Engstrom et al. (2020). "Implementation Matters in Deep RL"

**实用工具：**
- Gymnasium：https://gymnasium.farama.org/
- CleanRL：https://github.com/vwxyzjn/cleanrl
- Optuna：https://optuna.org/（超参数优化）
- TensorBoard：可视化工具
