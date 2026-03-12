# 深度解析：为什么强化学习中使用 `exp(log_a - log_b)` 而非直接相除？

在 GRPO、PPO 或 CISPO 的代码实现中，计算新旧策略比率（Ratio/Importance Sampling Weight）时，标准写法是：
`ratio = torch.exp(current_log_probs - old_log_probs)`

---

## 1. 数学等价性
从纯数学理论上来说，这两者是完全等价的。根据对数的基本性质：
$$\log\left(\frac{a}{b}\right) = \log(a) - \log(b)$$
两边同时取指数 $e$：
$$\frac{a}{b} = e^{(\log a - \log b)}$$
其中 $a$ 是当前模型的概率 $\pi_\theta$， $b$ 是旧模型的概率 $\pi_{old}$。

---

## 2. 核心原因：数值稳定性（Numerical Stability）

在处理大语言模型（LLM）时，`vocab_size` 通常高达 10w+。经过 Softmax 后，单个 Token 的概率值会变得极其微小。

### A. 直接相除的风险（浮点数下溢）
* **下溢（Underflow）：** 随着训练进行，不被看好的 Token 概率可能低至 $10^{-40}$。在 `Float32` 精度下，这些值会被强制识别为 **0**。
* **崩溃（NaN）：** 如果旧概率 $b$ 变为 0，计算 $a/b$ 就会出现 `DivisionByZero`，产生 `NaN`，直接导致昂贵的训练任务中断。

### B. Log 空间计算的优势
* **保持健康：** 概率 $10^{-40}$ 在 Log 空间里是 $-92.1$。计算机处理 $-92.1$ 就像处理 $1.0$ 一样轻松，不会丢失精度。
* **减法代替除法：** 在 Log 空间进行减法计算，再最后取 `exp`，可以极大地拓宽计算机能够处理的数值范围。



---

## 3. 代码实现逻辑拆解

```python
# 1. 使用 F.log_softmax 内部集成的 Log-Sum-Exp 技巧防止溢出
log_probs = F.log_softmax(logits, dim=-1) 

# 2. 使用 gather 从全量分布中取出实际生成的 Token 概率
# 维度变化：[batch, seq, vocab] -> [batch, seq]
current_token_log_prob = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

# 3. 在 Log 空间相减，确保 ratio 始终为正且计算鲁棒
# ratio 代表：相对于采样时的“旧我”，“新我”进步了多少
ratio = torch.exp(current_token_log_prob - old_log_probs)
