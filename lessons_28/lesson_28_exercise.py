"""
Lesson 28: RL 用于推理 (RL for Reasoning)
练习代码 - 参考实现

本练习实现简单的过程奖励训练，包括：
1. 多步推理环境（简化数学计算）
2. 结果奖励模型 (ORM)
3. 过程奖励模型 (PRM)
4. Best-of-N 采样
5. ORM vs PRM 引导对比
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: 多步推理环境 (提供，无 TODO)
# ============================================================

class MathReasoningEnv:
    """
    简化的多步数学推理环境

    问题形式: 计算 a ○ b ○ c ○ d，其中 ○ 是 +, -, *
    需要按运算优先级分步计算

    每步: 选择两个数和一个运算符计算
    """

    def __init__(self, num_steps=3):
        self.num_steps = num_steps
        self.reset()

    def reset(self):
        """生成一个新问题"""
        # 生成随机数字 (1-9 的小整数)
        self.numbers = np.random.randint(1, 10, size=4).tolist()
        # 生成随机运算符 (0=+, 1=-, 2=*)
        self.operators = np.random.randint(0, 3, size=3).tolist()
        self.op_symbols = ['+', '-', '*']

        # 按运算优先级逐步计算正确答案
        self.correct_steps, self.correct_answer = self._solve()

        # 当前状态
        self.current_step = 0
        self.step_results = []
        self.step_correct = []

        return self._get_problem_str()

    def _solve(self):
        """按优先级求解"""
        nums = self.numbers.copy()
        ops = self.operators.copy()
        steps = []

        # 第一遍: 处理乘法
        i = 0
        while i < len(ops):
            if ops[i] == 2:  # 乘法
                result = nums[i] * nums[i + 1]
                steps.append((nums[i], '*', nums[i + 1], result))
                nums = nums[:i] + [result] + nums[i + 2:]
                ops = ops[:i] + ops[i + 1:]
            else:
                i += 1

        # 第二遍: 处理加法和减法
        while len(ops) > 0:
            op = ops.pop(0)
            a = nums.pop(0)
            b = nums[0]
            if op == 0:
                result = a + b
                steps.append((a, '+', b, result))
            else:
                result = a - b
                steps.append((a, '-', b, result))
            nums[0] = result

        return steps, nums[0]

    def _get_problem_str(self):
        """返回问题字符串"""
        parts = [str(self.numbers[0])]
        for i, op in enumerate(self.operators):
            parts.append(self.op_symbols[op])
            parts.append(str(self.numbers[i + 1]))
        return ' '.join(parts)

    def submit_step(self, operand_a, operator, operand_b, result):
        """
        提交一步计算结果

        返回:
            step_correct: 这一步是否正确
            done: 是否完成所有步骤
            final_correct: 最终答案是否正确（仅 done 时有意义）
        """
        if self.current_step < len(self.correct_steps):
            correct = self.correct_steps[self.current_step]
            step_correct = (result == correct[3])  # 只检查结果
        else:
            step_correct = False

        self.step_results.append(result)
        self.step_correct.append(step_correct)
        self.current_step += 1

        done = self.current_step >= len(self.correct_steps)
        final_correct = (result == self.correct_answer) if done else False

        return step_correct, done, final_correct


# ============================================================
# Part 2: 推理智能体 (提供，无 TODO)
# ============================================================

class ReasoningAgent:
    """简化的推理智能体 -- 用概率模拟推理质量"""

    def __init__(self, accuracy=0.7):
        """accuracy: 每步计算正确的概率"""
        self.accuracy = accuracy

    def solve(self, env):
        """
        尝试解决问题

        返回:
            steps: [(a, op, b, result, correct), ...] 每步的结果
            final_correct: 最终答案是否正确
        """
        problem = env._get_problem_str()
        correct_steps = env.correct_steps
        steps = []

        all_correct = True
        for i, (a, op, b, correct_result) in enumerate(correct_steps):
            # 以 accuracy 概率计算正确
            if np.random.random() < self.accuracy:
                result = correct_result
                step_correct = True
            else:
                # 生成一个错误结果
                result = correct_result + np.random.choice([-2, -1, 1, 2])
                step_correct = False
                all_correct = False

            steps.append((a, op, b, result, step_correct))

        return steps, all_correct


# ============================================================
# Part 3: 结果奖励模型 - ORM (TODO 1)
# ============================================================

class OutcomeRewardModel(nn.Module):
    """
    结果奖励模型 (ORM)
    只根据最终答案是否正确给奖励
    """

    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        # 网络结构: input_dim -> hidden_dim -> hidden_dim -> 1
        # 使用 ReLU 激活，输出用 Sigmoid 得到概率
        # ---- 在这里实现 ----
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # ---- 代码结束 ----

    def forward(self, features):
        """features: 问题和答案的特征向量"""
        # ---- 在这里实现 ----
        return self.network(features)
        # ---- 代码结束 ----

    @staticmethod
    def extract_features(numbers, operators, step_results, num_steps=3, max_nums=4):
        """
        提取特征向量（将问题+结果编码为固定长度向量）
        features = numbers(4) + operators(3) + step_results(3) = 10维
        归一化: numbers/10, operators/3, results/100
        """
        feats = []
        feats.extend([n / 10.0 for n in numbers])
        feats.extend([o / 3.0 for o in operators])
        results_padded = step_results + [0] * (num_steps - len(step_results))
        feats.extend([r / 100.0 for r in results_padded[:num_steps]])
        return np.array(feats, dtype=np.float32)


# ============================================================
# Part 4: 过程奖励模型 - PRM (TODO 2)
# ============================================================

class ProcessRewardModel(nn.Module):
    """
    过程奖励模型 (PRM)
    对每一步推理给出正确概率
    """

    def __init__(self, input_dim=10, hidden_dim=64, max_steps=3):
        super().__init__()
        self.max_steps = max_steps
        # 共享特征提取器: input_dim -> hidden_dim -> hidden_dim (ReLU)
        # 每步独立的预测头: nn.ModuleList，每个头 Linear(hidden_dim, 1) + Sigmoid
        # ---- 在这里实现 ----
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.step_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            for _ in range(max_steps)
        ])
        # ---- 代码结束 ----

    def forward(self, features, step_idx):
        """
        预测第 step_idx 步的正确概率
        """
        # ---- 在这里实现 ----
        h = self.feature_net(features)
        return self.step_heads[step_idx](h)
        # ---- 代码结束 ----

    def score_all_steps(self, features):
        """预测所有步骤的正确概率"""
        # ---- 在这里实现 ----
        return [self.forward(features, i) for i in range(self.max_steps)]
        # ---- 代码结束 ----

    @staticmethod
    def extract_step_features(numbers, operators, step_results_so_far,
                              num_steps=3, max_nums=4):
        """与 ORM 类似，但 step_results_so_far 可能是部分结果"""
        feats = []
        feats.extend([n / 10.0 for n in numbers])
        feats.extend([o / 3.0 for o in operators])
        results_padded = step_results_so_far + [0] * (num_steps - len(step_results_so_far))
        feats.extend([r / 100.0 for r in results_padded[:num_steps]])
        return np.array(feats, dtype=np.float32)


# ============================================================
# Part 5: 训练 ORM 和 PRM (TODO 3)
# ============================================================

def collect_training_data(num_episodes=1000, agent_accuracy=0.7):
    """
    收集训练数据

    返回:
        data: list of dict, 每个 dict 包含:
            - numbers: 问题中的数字
            - operators: 问题中的运算符
            - step_results: 每步计算的结果列表
            - step_labels: 每步是否正确的布尔值列表
            - final_correct: 最终答案是否正确
    """
    # ---- 在这里实现 ----
    env = MathReasoningEnv()
    agent = ReasoningAgent(accuracy=agent_accuracy)
    data = []

    for _ in range(num_episodes):
        env.reset()
        steps, final_correct = agent.solve(env)

        step_results = [s[3] for s in steps]       # 每步的结果值
        step_labels = [s[4] for s in steps]         # 每步是否正确

        data.append({
            'numbers': env.numbers,
            'operators': env.operators,
            'step_results': step_results,
            'step_labels': step_labels,
            'final_correct': final_correct
        })

    return data
    # ---- 代码结束 ----


def train_orm(orm, data, num_epochs=30, lr=1e-3):
    """
    训练结果奖励模型 (ORM)

    返回:
        losses: 每个 epoch 的平均损失
        accuracies: 每个 epoch 的准确率
    """
    # ---- 在这里实现 ----
    optimizer = optim.Adam(orm.parameters(), lr=lr)
    criterion = nn.BCELoss()
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_count = 0

        for sample in data:
            features = OutcomeRewardModel.extract_features(
                sample['numbers'], sample['operators'], sample['step_results']
            )
            features_tensor = torch.tensor(features).unsqueeze(0)  # (1, 10)
            label = torch.tensor([[1.0 if sample['final_correct'] else 0.0]])

            pred = orm(features_tensor)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted_label = (pred.item() > 0.5)
            correct_count += (predicted_label == sample['final_correct'])

        avg_loss = epoch_loss / len(data)
        accuracy = correct_count / len(data)
        losses.append(avg_loss)
        accuracies.append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f"  ORM Epoch {epoch+1}/{num_epochs}: "
                  f"Loss={avg_loss:.4f}, Acc={accuracy:.2%}")

    return losses, accuracies
    # ---- 代码结束 ----


def train_prm(prm, data, num_epochs=30, lr=1e-3):
    """
    训练过程奖励模型 (PRM)

    返回:
        losses: 每个 epoch 的平均损失
        accuracies: 每个 epoch 的准确率
    """
    # ---- 在这里实现 ----
    optimizer = optim.Adam(prm.parameters(), lr=lr)
    criterion = nn.BCELoss()
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_count = 0
        total_steps = 0

        for sample in data:
            num_steps = len(sample['step_results'])

            for step in range(num_steps):
                # 使用到当前步为止的结果
                results_so_far = sample['step_results'][:step + 1]
                features = ProcessRewardModel.extract_step_features(
                    sample['numbers'], sample['operators'], results_so_far
                )
                features_tensor = torch.tensor(features).unsqueeze(0)
                label = torch.tensor(
                    [[1.0 if sample['step_labels'][step] else 0.0]]
                )

                pred = prm(features_tensor, step)
                loss = criterion(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                predicted_label = (pred.item() > 0.5)
                correct_count += (predicted_label == sample['step_labels'][step])
                total_steps += 1

        avg_loss = epoch_loss / total_steps
        accuracy = correct_count / total_steps
        losses.append(avg_loss)
        accuracies.append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f"  PRM Epoch {epoch+1}/{num_epochs}: "
                  f"Loss={avg_loss:.4f}, Acc={accuracy:.2%}")

    return losses, accuracies
    # ---- 代码结束 ----


# ============================================================
# Part 6: Best-of-N 采样 (TODO 4)
# ============================================================

def best_of_n_orm(env, agent, orm, N=8):
    """
    用 ORM 选择最佳推理路径

    参数:
        env: 推理环境 (已 reset)
        agent: 推理智能体
        orm: 训练好的 ORM
        N: 采样次数

    返回:
        best_steps: 最佳路径的步骤
        best_correct: 最佳路径是否最终正确
    """
    # ---- 在这里实现 ----
    best_score = -1.0
    best_steps = None
    best_correct = False

    for _ in range(N):
        # 对同一道题让 agent 多次求解
        steps, final_correct = agent.solve(env)

        step_results = [s[3] for s in steps]
        features = OutcomeRewardModel.extract_features(
            env.numbers, env.operators, step_results
        )
        features_tensor = torch.tensor(features).unsqueeze(0)

        with torch.no_grad():
            score = orm(features_tensor).item()

        if score > best_score:
            best_score = score
            best_steps = steps
            best_correct = final_correct

    return best_steps, best_correct
    # ---- 代码结束 ----


def best_of_n_prm(env, agent, prm, N=8):
    """
    用 PRM 选择最佳推理路径（步骤得分的乘积）

    参数:
        env: 推理环境 (已 reset)
        agent: 推理智能体
        prm: 训练好的 PRM
        N: 采样次数

    返回:
        best_steps: 最佳路径的步骤
        best_correct: 最佳路径是否最终正确
    """
    # ---- 在这里实现 ----
    best_score = -1.0
    best_steps = None
    best_correct = False

    for _ in range(N):
        steps, final_correct = agent.solve(env)

        step_results = [s[3] for s in steps]

        # 计算每步得分的乘积
        total_score = 1.0
        for step_idx in range(len(steps)):
            results_so_far = step_results[:step_idx + 1]
            features = ProcessRewardModel.extract_step_features(
                env.numbers, env.operators, results_so_far
            )
            features_tensor = torch.tensor(features).unsqueeze(0)

            with torch.no_grad():
                step_score = prm(features_tensor, step_idx).item()
            total_score *= step_score

        if total_score > best_score:
            best_score = total_score
            best_steps = steps
            best_correct = final_correct

    return best_steps, best_correct
    # ---- 代码结束 ----


def compare_orm_vs_prm(num_problems=200, N=8, agent_accuracy=0.7):
    """
    对比 ORM vs PRM 引导的 Best-of-N

    返回:
        no_search_acc: 无搜索准确率
        orm_acc: ORM 引导的 Best-of-N 准确率
        prm_acc: PRM 引导的 Best-of-N 准确率
    """
    # ---- 在这里实现 ----
    # 先收集数据训练模型
    print("  收集训练数据...")
    train_data = collect_training_data(num_episodes=1000, agent_accuracy=agent_accuracy)

    print("  训练 ORM...")
    orm = OutcomeRewardModel()
    train_orm(orm, train_data, num_epochs=20, lr=1e-3)
    orm.eval()

    print("  训练 PRM...")
    prm = ProcessRewardModel()
    train_prm(prm, train_data, num_epochs=20, lr=1e-3)
    prm.eval()

    # 对比评估
    env = MathReasoningEnv()
    agent = ReasoningAgent(accuracy=agent_accuracy)

    no_search_correct = 0
    orm_correct = 0
    prm_correct = 0

    for i in range(num_problems):
        if (i + 1) % 50 == 0:
            print(f"  评估进度: {i+1}/{num_problems}")

        # 生成一个新问题
        env.reset()
        # 保存问题信息，因为 agent.solve 不会改变 env 的 numbers/operators
        numbers_saved = env.numbers.copy()
        operators_saved = env.operators.copy()
        correct_steps_saved = list(env.correct_steps)
        correct_answer_saved = env.correct_answer

        # 无搜索: 单次尝试
        _, final_correct = agent.solve(env)
        if final_correct:
            no_search_correct += 1

        # ORM Best-of-N
        _, orm_final = best_of_n_orm(env, agent, orm, N=N)
        if orm_final:
            orm_correct += 1

        # PRM Best-of-N
        _, prm_final = best_of_n_prm(env, agent, prm, N=N)
        if prm_final:
            prm_correct += 1

    no_search_acc = no_search_correct / num_problems
    orm_acc = orm_correct / num_problems
    prm_acc = prm_correct / num_problems

    return no_search_acc, orm_acc, prm_acc
    # ---- 代码结束 ----


# ============================================================
# Part 7: 可视化 (提供，无 TODO)
# ============================================================

def plot_training(orm_losses, orm_accs, prm_losses, prm_accs):
    """绘制 ORM 和 PRM 训练过程"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ORM 损失
    axes[0, 0].plot(orm_losses, 'b-', linewidth=2)
    axes[0, 0].set_title('ORM 训练损失', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].grid(True, alpha=0.3)

    # ORM 准确率
    axes[0, 1].plot(orm_accs, 'b-', linewidth=2)
    axes[0, 1].set_title('ORM 训练准确率', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    # PRM 损失
    axes[1, 0].plot(prm_losses, 'r-', linewidth=2)
    axes[1, 0].set_title('PRM 训练损失', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('损失')
    axes[1, 0].grid(True, alpha=0.3)

    # PRM 准确率
    axes[1, 1].plot(prm_accs, 'r-', linewidth=2)
    axes[1, 1].set_title('PRM 训练准确率', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('准确率')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)

    plt.suptitle('奖励模型训练过程', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_28/reward_model_training.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  训练曲线已保存到 reward_model_training.png")


def plot_comparison(no_search_acc, orm_acc, prm_acc):
    """对比 Best-of-N 结果"""
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ['无搜索\n(单次尝试)', 'ORM\nBest-of-8', 'PRM\nBest-of-8']
    accs = [no_search_acc, orm_acc, prm_acc]
    colors = ['#95a5a6', '#3498db', '#e74c3c']

    bars = ax.bar(methods, accs, color=colors, width=0.5, edgecolor='black',
                  linewidth=1.2)

    # 在柱子上方添加数值标签
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=14,
                fontweight='bold')

    ax.set_ylabel('准确率', fontsize=14)
    ax.set_title('ORM vs PRM 引导的 Best-of-N 采样', fontsize=16,
                 fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=no_search_acc, color='gray', linestyle='--', alpha=0.5,
               label=f'基线: {no_search_acc:.1%}')
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('e:/rl_lessons/lessons_28/orm_vs_prm.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  对比图已保存到 orm_vs_prm.png")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # 步骤 1: 收集数据
    print("=" * 60)
    print("步骤 1: 收集推理数据")
    print("=" * 60)
    data = collect_training_data(num_episodes=2000, agent_accuracy=0.7)

    # 统计信息
    correct_count = sum(1 for d in data if d['final_correct'])
    print(f"  总样本数: {len(data)}")
    print(f"  最终正确数: {correct_count} ({correct_count/len(data):.2%})")
    avg_step_acc = np.mean([
        np.mean(d['step_labels']) for d in data
    ])
    print(f"  平均步骤准确率: {avg_step_acc:.2%}")

    # 步骤 2: 训练 ORM 和 PRM
    print("\n" + "=" * 60)
    print("步骤 2: 训练 ORM 和 PRM")
    print("=" * 60)
    orm = OutcomeRewardModel()
    prm = ProcessRewardModel()
    orm_losses, orm_accs = train_orm(orm, data, num_epochs=30)
    prm_losses, prm_accs = train_prm(prm, data, num_epochs=30)
    plot_training(orm_losses, orm_accs, prm_losses, prm_accs)

    # 步骤 3: Best-of-N 对比
    print("\n" + "=" * 60)
    print("步骤 3: Best-of-N 对比 (ORM vs PRM)")
    print("=" * 60)
    no_search_acc, orm_bon_acc, prm_bon_acc = compare_orm_vs_prm(
        num_problems=200, N=8
    )
    plot_comparison(no_search_acc, orm_bon_acc, prm_bon_acc)

    print(f"\n无搜索准确率: {no_search_acc:.2%}")
    print(f"ORM Best-of-8 准确率: {orm_bon_acc:.2%}")
    print(f"PRM Best-of-8 准确率: {prm_bon_acc:.2%}")
    print("\n关键发现：")
    print("1. PRM 提供步骤级别的密集奖励信号")
    print("2. PRM 引导的搜索通常优于 ORM")
    print("3. Best-of-N 采样能显著提升推理准确率")
