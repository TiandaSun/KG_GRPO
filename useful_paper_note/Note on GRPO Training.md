# Note on GRPO Training

本文记录了使用GRPO 训练基金助手 SubAgent 的完整过程，包括 20+ 个版本的迭代以及最终找到有效方案的思考过程。

## **一、项目背景**

### **1.1 模型选择**

项目目标是用 7B 小模型代替已有的 Mixtral 8X22B，实现低延迟低成本的特定场景 SubAgent。

核心挑战：**如何在 7B 模型的能力边界内，把工具选择准确率做到极致？**

事实证明，通过精心设计的 GRPO 训练流程，7B 模型完全可以在特定任务上达到很高的准确率。

### **1.2 任务定义**

训练一个 Mutual Fund Assistant SubAgent，根据用户请求准确选择工具（API）。

可用工具：`check_balance`、`transfer_funds`、`withdraw_funds`、`place_order`、`get_transaction_history`、`check_market_status`

场景分级：

- 简单："查一下我的余额" → `check_balance`
- 复杂："我今天早上卖了 AMZN 股票，钱什么时候到账？" → 需理解 T+1 结算规则
- 组合："转账 10 万再提现 5 万" → 需规划多步操作

---

## **二、基础实验与问题发现**

### **2.1 基线模型**

标准 GRPO 训练结果不理想，平均分 63%，Tool 准确率 67%。

各类别得分差异明显：history/market/balance 在 82-89%，但 settlement 仅 18%、intraday 23%、deposit 31%、combo 12%。

直觉告诉我们弱类别是数据不够，于是 3 倍过采样，结果全面退化。查看日志发现近 50% 步数梯度为零，过采样让简单样本重复出现，没有提供新的学习信号。

### **2.2 领域数据扩充**

考虑到弱类别是领域知识问题，我们准备了 100+ 条新样本覆盖 T+1 结算、周末处理等场景。

结果完全无效。日志显示 `frac_reward_zero_std` 几乎 100%——模型已经"会"了，没有学习信号。GRPO 需要奖励方差才能学习。

---

## **三、[NGRPO](https://zhida.zhihu.com/search?content_id=270296059&content_type=Article&match_order=1&q=NGRPO&zhida_source=entity) 突破**

核心问题：所有生成的 reward 相同时，advantage = 0，模型无法学习。

受 NGRPO 论文启发，我们添加虚拟满分样本：

`rewards = [0.8, 0.8, 0.8, 0.8]
augmented = rewards + [1.0]  # 虚拟满分样本
mean = sum(augmented) / len(augmented)  # 0.84
std = sqrt(variance(augmented))          # 0.08
advantage = (r - mean) / std             # 非零！`

即使真实样本得分相同，虚拟样本也能提供参照点，产生非零梯度。

结果：平均分 +13 点，tool 准确率 +14 点，intraday +35 点。关键指标 `frac_reward_zero_std: 0.0`。

---

## **四、各种"高级"方法的尝试**

基于 NGRPO 拿到满意的 baseline 后，尝试了多种论文方法，效果都不如预期。

- *渐进式奖励塑形 (PRS)**：分阶段给予奖励，结果跌了 5 点。模型已能正确产生格式，细粒度帮助不大。
- *难度感知重加权 (GRPO-LEAD)**：难样本高权重，combo 跌了近 20 点。95% 样本正确率太高，几乎没有"难样本"，权重调整反而破坏了有效学习信号。

**SFT 领域知识注入**：300 条 SFT 后 reasoning 提升但 tool 选择退化，1500 条后仍不如 NGRPO。模型学会了正确推理，但 tool 选择反而变差。

结论：算法改进无法解决数据问题，数据集太简单时各种技术都不管用。

---

## **五、[DART](https://zhida.zhihu.com/search?content_id=270296059&content_type=Article&match_order=1&q=DART&zhida_source=entity) 的启示**

SFT 实验中观察到奇怪现象：reasoning 变好但 tool 选择变差。我想到刚读到的一篇DART 论文 (arXiv:2602.00994) 提到：

> Reasoning 和 Tool-use 的梯度方向几乎正交
> 

我们实现了简化版 Token-Level 梯度分离：

`reasoning_tokens = tokens_in("reasoning": "...")
tool_tokens = tokens_in("action": {"tool": ...})
loss = 0.3 * reasoning_loss + 0.7 * tool_loss`

整体略低于 NGRPO，但 Combo 提升到 78%。DART 在复杂场景确实有效，但需要足够训练数据。

---

## **六、数据混合的教训**

大量增加 combo 数据后，transfer/order 等场景性能暴跌，combo 也没提升多少。降低比例后效果依然较差。数据混合比例极其敏感。

---

## **七、ReAct Agent 的突破**

我停下来重新思考：Combo 场景本质是**多步规划**问题，不是简单的工具选择。

原训练数据：

`{"input": "转账10万再提现5万", "output": '{"tool": "check_balance"}'}`

label 本身就错了！测试期望 `transfer_funds`，训练却教 `check_balance`。

重新设计输出格式：

`{
  "thought": "两步操作：先转账再提现。执行转账。",
  "action": {"tool": "transfer_funds", "params": {"amount": 1000000}},
  "plan": ["transfer 10 lakh", "withdraw 5 lakh"]
}`

thought 让模型先思考，action 输出第一步实际操作，plan 规划后续步骤。

结果：combo 91%，平均分 89%，tool 准确率 92%。**问题定义比算法更重要**。

---

## **八、细节优化**

ReAct 成功后继续强化弱类别，但提升有限。深挖日志发现奖励函数问题：

`reasoning: +0.2
action.tool: +0.3
valid_tool: +0.2
params: +0.1
# 最大分: 0.8 ← 问题所在！`

上限 0.8，所有正确输出得分相同，没有区分度。调整到 1.0 后立即提升 6%。

分析错误模式发现模型过度使用 `check_market_status`，添加针对性惩罚 -0.3。

最终：平均分 96%，tool 准确率 97%，最低的 settlement 也到 79%。**针对具体错误模式设计惩罚比通用算法改进更有效**。

---

## **九、核心经验**

### **关键监控指标**

| **指标** | **健康范围** | **问题信号** |
| --- | --- | --- |
| frac_reward_zero_std | < 30% | > 50% 奖励函数有问题 |
| reward | 接近 1.0 | < 0.8 奖励上限太低 |
| grad_norm | 非零 | 持续为 0 没有学习 |

### **奖励函数设计**

最大分必须能达到 1.0，要有足够区分度，针对具体错误模式设计惩罚，持续监控 `frac_reward_zero_std`。

### **数据教训**

简单重复数据会导致灾难性遗忘，LLM 生成数据质量难保证，改变数据格式（ReAct）有效，修正错误 label 是关键。

### **算法选择**

NGRPO 通用解决零梯度问题，DART 适合复杂场景，PRS/难度感知需要足够难样本才有效。

> 问题定义 > 数据质量 > 算法选择
> 

---

## **十、最终方案**

`# 模型
base_model = "[Qwen/Qwen2.5-7B-Instruct](https://zhida.zhihu.com/search?content_id=270296059&content_type=Article&match_order=1&q=Qwen%2FQwen2.5-7B-Instruct&zhida_source=entity)"
lora_r, lora_alpha = 16, 32

# 训练
num_generations, temperature = 8, 0.9
learning_rate, epochs = 5e-6, 4

# 奖励函数
def reward(completion, prompt):
    score = 0.0
    if 'thought' in parsed: score += 0.2
    if 'action' in parsed and 'tool' in action:
        score += 0.2
        if tool in valid_tools: score += 0.2
        if 'params' in action: score += 0.1
    if 'plan' in parsed: score += 0.2
    if tool == 'check_market_status' and not market_keywords_in_prompt:
        score -= 0.3
    return max(0, min(score, 1.0))

use_virtual_max_reward = True  # NGRPO`

---

## **十一、总结**

- 过采样实验发现了零梯度问题
- SFT 实验揭示了 reasoning 和 tool-use 的冲突
- 数据混合实验认识到比例敏感性
- `frac_reward_zero_std` 是关键监控指标

核心建议：先确保问题定义正确，监控 `frac_reward_zero_std`，奖励函数最大分要能达到 1.0，针对错误模式设计惩罚，不要迷信"高级"算法。