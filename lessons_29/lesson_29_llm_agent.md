# Lesson 29: 在线 RL 与 LLM Agent

## 1. LLM 作为智能体

### 1.1 从对话到行动

**传统 LLM：** 输入文本 → 输出文本

**LLM Agent：** 观察环境 → 思考 → 执行动作 → 获得反馈 → 循环

```
传统 LLM:
  用户: "今天天气怎么样？"
  LLM:  "我无法获取实时天气信息。"

LLM Agent:
  用户: "今天天气怎么样？"
  Agent 思考: "我需要查询天气 API"
  Agent 动作: call_weather_api(city="北京")
  API 返回: {"temp": 25, "weather": "晴"}
  Agent: "今天北京天气晴朗，气温 25°C。"
```

### 1.2 RL 视角下的 LLM Agent

| RL 概念 | LLM Agent 对应 |
|---------|---------------|
| **环境** | 外部世界（网页、API、终端、文件系统） |
| **状态** | 对话历史 + 环境观察 |
| **动作空间** | 工具调用 / 文本生成 / 代码执行 |
| **策略** | LLM（$\pi_\theta$） |
| **奖励** | 任务完成度、用户反馈 |
| **轨迹** | 多步交互序列 |

### 1.3 与标准 RL 的区别

| | 标准 RL | LLM Agent |
|--|---------|-----------|
| 动作空间 | 离散/连续，低维 | 文本，极高维 |
| 状态表示 | 向量 | 自然语言 |
| 策略 | 从零学习 | 预训练 + 微调 |
| 环境 | 模拟器 | 真实世界 |
| 交互成本 | 低 | 高（API 调用等） |


## 2. Agent 架构

### 2.1 ReAct 框架

**Yao et al. (2022)：** Reasoning + Acting

```
循环:
  1. Thought (思考): 分析当前状态，决定下一步
  2. Action (行动): 调用工具或执行操作
  3. Observation (观察): 获取环境反馈
  重复直到任务完成
```

**示例：**

```
问题: "苹果公司的 CEO 出生在哪个城市？"

Thought 1: 我需要先查找苹果公司的 CEO 是谁。
Action 1: Search("Apple CEO")
Observation 1: Tim Cook is the CEO of Apple Inc.

Thought 2: 现在我需要查找 Tim Cook 的出生地。
Action 2: Search("Tim Cook birthplace")
Observation 2: Tim Cook was born in Mobile, Alabama.

Thought 3: 我找到答案了。
Answer: Tim Cook 出生在美国阿拉巴马州的 Mobile 市。
```

### 2.2 工具调用（Tool Use）

**常见工具类型：**

| 工具 | 功能 | 示例 |
|------|------|------|
| **搜索** | 信息检索 | Google, Wikipedia |
| **代码执行** | 运算、数据处理 | Python REPL |
| **API 调用** | 外部服务 | 天气、地图、数据库 |
| **文件操作** | 读写文件 | 编辑代码、文档 |
| **终端** | 系统命令 | bash, git |

**工具调用的 RL 建模：**

$$a_t = \begin{cases}
\text{tool\_call}(\text{name}, \text{args}) & \text{需要外部信息} \\
\text{text\_response} & \text{可以直接回答}
\end{cases}$$

### 2.3 Agent 循环

```python
def agent_loop(llm, tools, task, max_steps=10):
    history = [{"role": "user", "content": task}]
    
    for step in range(max_steps):
        # LLM 决策
        response = llm.generate(history)
        
        if response.has_tool_call:
            # 执行工具
            tool_name = response.tool_call.name
            tool_args = response.tool_call.args
            observation = tools[tool_name](**tool_args)
            
            history.append({"role": "assistant", "content": response})
            history.append({"role": "tool", "content": observation})
        else:
            # 最终回答
            history.append({"role": "assistant", "content": response})
            return response
    
    return "达到最大步数限制"
```


## 3. 用 RL 训练 Agent

### 3.1 为什么需要 RL？

**SFT 的局限：**
- 只能从专家轨迹学习
- 无法从错误中学习
- 分布偏移问题（训练 vs 实际）

**RL 的优势：**
- 从试错中学习
- 探索新的解决策略
- 最大化长期回报

### 3.2 奖励设计

**任务完成奖励：**

```python
def task_reward(task, result):
    if task_completed_successfully(result):
        return 1.0
    elif partial_completion(result):
        return 0.5
    else:
        return 0.0
```

**过程奖励：**

```python
def step_reward(action, observation):
    reward = 0.0
    
    # 有效的工具调用
    if action.is_valid_tool_call:
        reward += 0.1
    
    # 获取有用信息
    if observation.is_informative:
        reward += 0.2
    
    # 冗余操作惩罚
    if action.is_redundant:
        reward -= 0.1
    
    # 效率奖励
    reward -= 0.01  # 每步小惩罚，鼓励高效
    
    return reward
```

### 3.3 训练方法

**1. 在线 RL（Online RL）**

```
循环:
  1. Agent 与环境交互，收集轨迹
  2. 计算奖励
  3. 用 PPO/GRPO 更新策略
  4. 重复
```

**2. 拒绝采样微调（Rejection Sampling Fine-tuning）**

```
循环:
  1. 对每个任务，生成 N 个轨迹
  2. 筛选成功的轨迹
  3. 在成功轨迹上 SFT
  4. 重复
```

**3. 离线 RL（Offline RL）**

- 从历史日志学习
- 不需要在线交互
- 使用 CQL/IQL/DPO


## 4. 实际案例

### 4.1 WebAgent

**任务：** 在网页上完成操作（购物、预订等）

**环境：**
- 状态：DOM 树 / 截图
- 动作：点击、输入、滚动
- 奖励：任务完成度

**训练：**
1. 收集人类操作轨迹
2. SFT 学习基本操作
3. RL 优化成功率

### 4.2 SWE-Agent

**任务：** 自动修复 GitHub Issues

**环境：**
- 状态：代码库、Issue 描述、终端输出
- 动作：编辑文件、运行命令、搜索代码
- 奖励：测试通过率

**流程：**

```
1. 读取 Issue 描述
2. 搜索相关代码
3. 定位 bug
4. 编辑代码修复
5. 运行测试验证
6. 提交 patch
```

### 4.3 数据分析 Agent

**任务：** 自动数据分析

**环境：**
- 状态：数据集、分析历史
- 动作：执行 Python 代码（pandas, matplotlib）
- 奖励：分析质量评分

### 4.4 Claude Code / Codex

**任务：** 辅助编程

**环境：**
- 状态：代码上下文、用户需求
- 动作：读/写/编辑文件、运行命令
- 奖励：代码正确性、用户满意度


## 5. Agent 评估

### 5.1 基准测试

| 基准 | 任务类型 | 评估指标 |
|------|----------|----------|
| **WebArena** | 网页操作 | 任务成功率 |
| **SWE-bench** | 代码修复 | 测试通过率 |
| **GAIA** | 通用 Agent | 任务完成率 |
| **AgentBench** | 多领域 | 综合评分 |
| **ToolBench** | 工具使用 | 调用准确率 |

### 5.2 评估维度

1. **成功率**：任务完成的比例
2. **效率**：完成任务的步数
3. **鲁棒性**：面对异常情况的表现
4. **安全性**：是否执行危险操作
5. **泛化性**：新任务的表现


## 6. 挑战与前沿

### 6.1 核心挑战

**1. 安全性**
- Agent 能执行真实操作
- 错误操作可能造成损害
- 需要安全护栏

**2. 长程规划**
- 复杂任务需要多步规划
- 当前 LLM 在长程任务上表现差
- 需要更好的记忆和规划机制

**3. 环境多样性**
- 每个环境（网页、API 等）都不同
- 需要泛化能力
- 工具描述和 API 文档的理解

**4. 奖励设计**
- 真实任务的奖励难以定义
- 中间奖励设计困难
- 稀疏奖励下的探索问题

### 6.2 前沿方向

1. **多模态 Agent**：结合视觉和语言
2. **多 Agent 协作**：多个 Agent 分工合作
3. **持续学习**：从交互中不断改进
4. **安全对齐**：确保 Agent 行为安全


## 7. 总结

### 7.1 核心要点

1. **LLM Agent = LLM + 工具 + 环境交互**
2. **ReAct 框架**：Thought → Action → Observation 循环
3. **RL 训练 Agent**：从试错中学习，超越模仿
4. **奖励设计**：任务奖励 + 过程奖励 + 效率惩罚
5. **实际应用**：WebAgent、SWE-Agent、数据分析
6. **核心挑战**：安全性、长程规划、泛化

### 7.2 下节预告

下节课（最后一课）我们将学习 **前沿综述与展望**：
- Scaling RL
- Constitutional AI
- RLVR
- 多模态 RL
- 开源生态

### 7.3 扩展阅读

**必读论文：**
1. Yao et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models"
2. Yang et al. (2024). "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering"
3. Zhou et al. (2023). "WebArena: A Realistic Web Environment for Building Autonomous Agents"
4. Schick et al. (2023). "Toolformer: Language Models Can Teach Themselves to Use Tools"

**实用资源：**
- LangChain Agent 文档
- AutoGPT / MetaGPT
- OpenAI Function Calling API
