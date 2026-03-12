让我在 slime 代码库中搜索这些指标的计算方式：

[7 tools called]

现在我来详细解释 metrics.csv 中各个指标的含义和计算方式：

## 📊 metrics.csv 指标详解

### 1. **rollout/log_probs** - 当前策略对数概率

**含义**：当前训练策略（policy）对每个生成 token 的对数概率的均值。

**计算方式**：

```python
log_prob = -cross_entropy(logits, tokens)  # 每个 token 的对数概率
# 然后对每个样本的 response 部分求均值
```

```151:158:/mnt/llm-train/users/explore-train/qingyu/slime/slime/utils/ppo_utils.py
def compute_log_probs(logits: torch.Tensor, tokens: torch.Tensor, process_group: dist.ProcessGroup | None):
    # TODO: when megatron is not installed, fall back to naive implementation
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    # convert to [seq_len, batch_size, vocab_size] as expected by fused_vocab_parallel_cross_entropy
    logits = logits.unsqueeze(1)
    tokens = tokens.unsqueeze(1)
    return -fused_vocab_parallel_cross_entropy(logits, tokens, process_group)
```

---

### 2. **rollout/rollout_log_probs** - Rollout 时的对数概率

**含义**：在 rollout（生成）阶段记录的对数概率。与 `log_probs` 的区别在于：
- `rollout_log_probs`：rollout 时用的模型计算的
- `log_probs`：训练前用最新模型重新计算的

**用途**：用于检测 rollout 和训练之间是否有模型不一致的问题。`train_rollout_logprob_abs_diff` 就是两者差异的绝对值。

---

### 3. **rollout/ref_log_probs** - 参考模型对数概率

**含义**：参考模型（Reference Model，通常是 SFT 后的初始模型）对每个 token 的对数概率。

**用途**：用于计算 KL 散度惩罚，防止策略模型偏离参考模型太远。

---

### 4. **rollout/advantages** - 优势函数

**含义**：衡量某个动作（token）相比平均水平好多少。正值表示该 token 比预期好，负值表示比预期差。

**计算方式**（根据不同算法）：

#### GRPO/GSPO（你使用的是这个，因为 `rollout/returns` = `rollout/advantages`）：

```python
# GRPO: 直接用 reward 作为 advantage
advantages = rewards  # 每个 token 的 advantage 都等于该样本的 reward
```

```201:208:/mnt/llm-train/users/explore-train/qingyu/slime/slime/utils/ppo_utils.py
def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns
```

#### PPO（使用 GAE）：

```python
# 广义优势估计 (GAE)
# Adv_t = δ_t + γλ*δ_{t+1} + (γλ)^2*δ_{t+2} + ...
# 其中 δ_t = r_t + γ*V_{t+1} - V_t
for t in reversed(range(response_len)):
    delta = rewards[t] + gamma * values[t+1] - values[t]
    lastgaelam = delta + gamma * lambd * lastgaelam
    advantages.append(lastgaelam)
```

---

### 5. **rollout/returns** - 回报

**含义**：从当前时刻开始的累积折扣奖励。

**计算方式**：
- **GRPO**：`returns = advantages`（相同）
- **PPO**：`returns = advantages + values`

---

### 6. **rollout/rewards** 和 **rollout/raw_reward**

- **raw_reward**：原始奖励，来自奖励模型或规则
- **rewards**：可能经过 KL 惩罚修正后的奖励

```python
# PPO 中，reward 会加入 KL 惩罚
rewards = raw_reward - kl_coef * KL(π_θ || π_ref)
```

---

### 7. **train/entropy_loss** - 熵损失

**含义**：策略输出分布的熵，衡量策略的随机性/探索程度。

**计算方式**：

```python
# 熵 = -Σ p(x) * log(p(x))
# 即 softmax 输出的熵
entropy = log(Σ exp(logits)) - Σ(softmax(logits) * logits)
```

```162:182:/mnt/llm-train/users/explore-train/qingyu/slime/slime/utils/ppo_utils.py
class _VocabParallelEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:

        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=process_group)
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=process_group)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        # ...
```

**用途**：加入总损失鼓励探索（`loss = pg_loss - entropy_coef * entropy_loss`）

---

### 8. **train/pg_loss** - 策略梯度损失

**含义**：PPO 的核心损失函数，即 clipped 策略梯度损失。

**计算方式**：

```python
# 重要性采样比率
ratio = exp(log_π_new - log_π_old)

# Clipped PPO loss
pg_loss1 = -ratio * advantages
pg_loss2 = -clip(ratio, 1-ε, 1+ε) * advantages
pg_loss = max(pg_loss1, pg_loss2)  # 取更保守的
```

```124:148:/mnt/llm-train/users/explore-train/qingyu/slime/slime/utils/ppo_utils.py
@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()
    # ...
```

---

### 9. **train/pg_clipfrac** - 策略梯度裁剪比例

**含义**：被 clip 操作影响的样本比例。

**解读**：
- 值为 0：所有更新都在信任域内
- 值较高：很多更新被裁剪，可能需要调整学习率或 clip 范围

---

### 10. **train/ppo_kl** - PPO KL 散度

**含义**：新策略与旧策略之间的 KL 散度。

**计算方式**：

```python
# KL ≈ log(π_new/π_old) = log_prob_new - log_prob_old
kl = log_probs - old_log_probs
```

---

### 11. **train/kl_loss** - KL 损失

**含义**：新策略与**参考模型**之间的 KL 散度（如果启用 `use_kl_loss`）。

**计算方式**：

```python
kl_loss = KL(π_θ || π_ref) = Σ (log_probs - ref_log_probs)
# 或使用更稳定的 k3 估计器
kl_loss = exp(-log_ratio) - 1 + log_ratio
```

---

### 12. **train/loss** - 总损失

**计算方式**：

```python
loss = pg_loss - entropy_coef * entropy_loss + kl_loss_coef * kl_loss
```

---

### 13. **train/train_rollout_logprob_abs_diff**

**含义**：训练时计算的 log_probs 与 rollout 时记录的 rollout_log_probs 的绝对差异。

**用途**：监控 on-policy vs off-policy 的程度，值越大说明训练与 rollout 之间模型变化越大。

---

## 📈 从你的数据观察

从你的 metrics.csv 可以看出：
1. **raw_reward = 0**：所有样本的原始奖励都是 0（可能是占位或特定任务设计）
2. **advantages = returns**：使用的是 GRPO 风格的优势估计
3. **pg_clipfrac = 0**：没有发生裁剪，策略更新很保守
4. **entropy_loss 逐渐下降**：策略在变得更确定/更少探索
5. **response_lengths 逐渐增加**：模型生成的回复越来越长

6. 让我查看 on-policy distillation 的 advantage 计算逻辑：

[2 tools called]

找到了关键代码！让我详细分析：

## On-Policy Distillation 的 Advantage 计算

从代码可以看到：

```313:339:/mnt/llm-train/users/explore-train/qingyu/slime/slime/backends/megatron_utils/loss.py
    elif args.advantage_estimator == "on_policy_distillation":
        student_log_probs = log_probs
        teacher_log_probs = rollout_data.get("teacher_log_probs")
        response_lengths = rollout_data.get("response_lengths")
        device = student_log_probs[0].device
        teacher_log_probs = [t_log_prob.to(device=device) for t_log_prob in teacher_log_probs]
        teacher_log_probs = [
            t_log_prob[-response_length:]
            for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
        ]
        # KL advantage with stop gradient (MiMo MOPD: sg[log(pi_teacher/pi_student)])
        kl_advantages = [
            (teacher_log_prob - student_log_prob).detach()
            for teacher_log_prob, student_log_prob in zip(teacher_log_probs, student_log_probs, strict=False)
        ]
        # Combine with ORM advantage if opd_orm_weight > 0 (MiMo MOPD: A_KL + alpha * A_ORM)
        opd_orm_weight = getattr(args, "opd_orm_weight", 0.0)
        if opd_orm_weight > 0 and rewards is not None:
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            orm_returns = get_grpo_returns(rewards_tensor, kl)
            advantages = [
                kl_adv + opd_orm_weight * orm_ret
                for kl_adv, orm_ret in zip(kl_advantages, orm_returns, strict=False)
            ]
        else:
            advantages = kl_advantages
        returns = advantages
```

## ✅ 是的，OPD 的 Advantage 等价于 Reverse KL

### 数学推导

**Advantage 定义**：
$$A_t = \log \pi_{\text{teacher}}(a_t|s_t) - \log \pi_{\text{student}}(a_t|s_t) = \log \frac{\pi_{\text{teacher}}}{\pi_{\text{student}}}$$

**Policy Gradient 损失**（在 on-policy 情况下 ratio ≈ 1）：

$$L_{PG} = -\mathbb{E}_{\pi_{\text{student}}}[A] = -\mathbb{E}_{\pi_{\text{student}}}\left[\log \frac{\pi_{\text{teacher}}}{\pi_{\text{student}}}\right]$$

**变换后**：

$$L_{PG} = \mathbb{E}_{\pi_{\text{student}}}\left[\log \frac{\pi_{\text{student}}}{\pi_{\text{teacher}}}\right]$$

这正是 **Reverse KL** 的定义：

$$\boxed{KL(\pi_{\text{student}} || \pi_{\text{teacher}}) = \mathbb{E}_{\pi_{\text{student}}}\left[\log \frac{\pi_{\text{student}}}{\pi_{\text{teacher}}}\right]}$$

### Forward KL vs Reverse KL

| | Forward KL | Reverse KL (你的 OPD) |
|---|---|---|
| 公式 | $KL(\pi_T \|\| \pi_S)$ | $KL(\pi_S \|\| \pi_T)$ |
| 采样分布 | Teacher | **Student** |
| 行为 | **Mean-seeking**：覆盖 teacher 所有模式 | **Mode-seeking**：聚焦 teacher 高概率区域 |
| 特点 | 可能导致 student 过于分散 | 避免 student 在 teacher 低概率处有高概率 |

### 你的 OPD 配置分析

从你的脚本：
```bash
--advantage-estimator on_policy_distillation
--opd-orm-weight 0.0  # 纯 KL 蒸馏，不混合 ORM reward
```

因为 `opd-orm-weight = 0`，所以你的 advantage **完全等于** reverse KL：

$$A = \text{sg}[\log \pi_T - \log \pi_S]$$

其中 `sg` 是 stop gradient，只让 student 的 log_probs 参与梯度计算。

### 为什么用 Reverse KL 做蒸馏？

1. **自然适配 RL 框架**：从 student 采样，用 policy gradient 优化
2. **Mode-seeking**：让 student 学习 teacher 最有信心的输出，避免"胡说八道"
3. **稳定性**：不需要从 teacher 采样，实现更简单

这也是 MiMo 论文中 MOPD（Mode-seeking On-Policy Distillation）的做法。
