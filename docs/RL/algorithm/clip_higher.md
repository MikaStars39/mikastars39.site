# 强化学习深度笔记：权重裁剪（Clip Higher）如何对抗熵崩塌

在长链推理（CoT）任务的强化学习过程中，**熵崩塌（Entropy Collapse）** 是导致训练失败的头号杀手。本文将深入探讨 CISPO 算法如何通过改进裁剪机制来解决这一问题。

---

## 1. 核心概念：熵崩塌（Entropy Collapse）

策略的熵 $H(\pi)$ 代表了模型输出的不确定性（探索能力）：
$$H(\pi) = -\sum_{a \in \mathcal{A}} \pi(a|s) \log \pi(a|s)$$

* **健康状态：** 熵值平缓下降，模型在保持探索的同时逐渐收敛。
* **崩塌状态：** 熵值在短时间内断崖式下跌。模型变得极度“自负”，概率分布过度尖锐，锁死在局部最优解，无法再学习新逻辑。

---

## 2. 传统 PPO/GRPO 的局限：梯度“禁言”

传统 PPO 使用 `min` 算子来限制更新幅度：
$$L_{PPO} = \min \left( r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t \right)$$

### 为什么这会导致崩塌？
1.  **关键 Token 的剧烈变动：** 在数学推理中，某些关键逻辑词（如 "Therefore", "Wait"）在被模型发现能导向正确答案时，其概率比率 $r_t$ 会瞬间激增。
2.  **梯度截断：** 一旦 $r_t > 1+\epsilon$，$L_{PPO}$ 会进入 Clip 区域。由于 Clip 项相对于当前参数 $\theta$ 是**常数**，其导数为 **0**。
3.  **副作用：** 算法直接“没收”了这些关键 Token 的进化机会。模型为了继续降低 Loss，只能强行压缩其他 Token 的空间，导致分布畸变，最终引发熵崩塌。



---

## 3. CISPO 的数学推导：从“截断”到“限速”

CISPO 提出废除 `min` 算子，将裁剪作用于**重要性采样权重**本身，而非整个目标函数：

### 目标函数
$$L_{CISPO} = \underbrace{\text{clip}(r_t, 1-\epsilon_{low}, 1+\epsilon_{high})}_{\text{权重裁剪 (系数)}} \cdot A_t \cdot \underbrace{\log \pi_\theta(a_t|s_t)}_{\text{保留梯度路径}}$$

### 为什么能避免崩塌？
* **方向保留：** 即使 $r_t$ 很大，梯度 $\nabla_\theta L$ 依然正比于 $\nabla_\theta \log \pi_\theta$。这意味着模型始终知道“正确的方向”在哪里。
* **限速更新：** 裁剪后的系数（如 1.2）限制了更新的**步长**。这就像给飞驰的赛车踩了刹车（限速），而不是直接把发动机拆了（梯度归零）。这种平稳的进步防止了概率分布的瞬间坍缩。

---

## 4. PyTorch 代码实现

以下代码展示了如何在计算 Loss 的同时监控熵值，并实现 CISPO 的核心裁剪逻辑。

```python
import torch
import torch.nn.functional as F

class CISPOLoss(torch.nn.Module):
    def __init__(self, eps_high=0.2, eps_low=0.2, entropy_coeff=0.01):
        super().__init__()
        self.eps_high = eps_high
        self.eps_low = eps_low
        self.entropy_coeff = entropy_coeff # 熵正则化系数

    def forward(self, logits, tokens, old_log_probs, advantages):
        """
        Args:
            logits: 当前模型的原始输出 [batch, seq, vocab]
            tokens: 实际选中的 token 序列 [batch, seq]
            old_log_probs: 采样时旧模型的 log 概率 [batch, seq]
            advantages: 组内归一化后的优势值 [batch]
        """
        # 1. 计算当前 log 概率分布
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 2. 提取选定 token 的 log 概率
        current_lp = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        
        # 3. 计算策略熵 (Entropy)
        # 监控此指标：若 ent 快速下降，说明模型正在丧失探索性
        entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
        
        # 4. 计算重要性采样权重 ratio
        ratio = torch.exp(current_lp - old_log_probs)
        
        # 5. CISPO 核心：裁剪权重并 detach 梯度，但保留 log_prob 的梯度
        # 这样即便 ratio 很大，也只会以 (1+eps_high) 的速率稳定更新
        clipped_ratio = torch.clamp(ratio, 1 - self.eps_low, 1 + self.eps_high).detach()
        
        # 6. 计算 Policy Loss
        # A 需要扩展到 token 维度
        A = advantages.unsqueeze(1) 
        policy_loss = -(clipped_ratio * A * current_lp).mean()
        
        # 7. 总 Loss = 策略 Loss - 熵奖励 (鼓励高熵)
        total_loss = policy_loss - self.entropy_coeff * entropy
        
        return total_loss, entropy, ratio.max()

# --- 实验建议 ---
# 在训练初期，如果发现 entropy 下降过快：
# 1. 尝试调大 eps_high (例如从 0.2 调至 0.5)
# 2. 增加 entropy_coeff
