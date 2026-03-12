# 证明：在 On-Policy Distillation 下，分布差异过大会导致训练失败

### 1. 核心定义与假设
设教师模型分布为 $P_T(x)$，学生模型分布为 $P_S(x|\theta)$。
On-Policy Distillation (OPD) 的目标是最小化学生分布相对于教师分布的 **逆向 KL 散度 (Reverse KL Divergence)**：

$$J(\theta) = D_{KL}(P_S || P_T) = \mathbb{E}_{x \sim P_S(x|\theta)} \left[ \log P_S(x|\theta) - \log P_T(x) \right]$$

**假设（分布差异过大）：**
假设 $P_S$ 与 $P_T$ 的支撑集（Support）几乎不重叠，即存在两个互斥区域 $ \mathcal{X}_{\text{bad}} $ 和 $ \mathcal{X}_{\text{good}} $：
* **学生采样区 ($\mathcal{X}_{bad}$):** $\forall x \in \mathcal{X}_{bad}, P_S(x) > 0$ 但 $P_T(x) \approx \epsilon$ (极小值)。
* **教师目标区 ($\mathcal{X}_{good}$):** $\forall x \in \mathcal{X}_{good}, P_T(x) > 0$ 但 $P_S(x) \approx 0$。

---

### 2. 梯度推导
根据对数导数技巧（Log-Derivative Trick），目标函数 $J(\theta)$ 的梯度 $\nabla_\theta J(\theta)$ 为：

$$\nabla_\theta J(\theta) = \mathbb{E}_{x \sim P_S} \left[ \nabla_\theta \log P_S(x|\theta) \cdot (\log P_S(x|\theta) - \log P_T(x) + 1) \right]$$

在 OPD 的语境下，我们定义 **Advantage (优势函数)** 为 $$A(x) = \log P_T(x) - \log P_S(x)$$。则梯度可简化表示为：

$$\nabla_\theta J \propto \mathbb{E}_{x \sim P_S} \left[ \nabla_\theta \log P_S(x|\theta) \cdot (-A(x)) \right]$$

---

### 3. 分区域失败分析

#### A. 正向信号缺失（The Blindness Problem）
在教师认为“正确”的区域 $\mathcal{X}_{good}$，虽然 $A(x)$ 的值可能很大（正向反馈强），但由于是 **On-Policy 采样**：

$$\text{Grad}_{good} = \int_{\mathcal{X}_{good}} P_S(x|\theta) \cdot \nabla_\theta \log P_S(x|\theta) \cdot (-A(x)) \, dx$$

由于在 $\mathcal{X}_{good}$ 中 $P_S(x) \approx 0$，导致该积分项趋近于 **0**。
**证明结论 1：** 学生模型永远无法采样到高分样本，因此无法获得向正确分布靠拢的动力。

#### B. 盲目惩罚与梯度爆炸（The Punishment Problem）
在学生目前采样的区域 $\mathcal{X}_{bad}$，由于 $P_T(x) \approx \epsilon$：

$$A(x) = \log(\epsilon) - \log(P_S(x)) \to -\infty$$

此时梯度项：

$$\text{Grad}_{bad} \approx \mathbb{E}_{x \in \mathcal{X}_{bad}} [ \nabla_\theta \log P_S(x) \cdot (+\infty) ]$$

**证明结论 2：** 模型收到的所有信号都是极大的负反馈。梯度仅能告诉模型“不要生成当前的样本”，但由于结论 1，模型不知道该转而生成什么。

---

### 4. 结论：策略塌缩与不收敛
1.  **模式搜索失效：** Reverse KL 具有“Mode-seeking”特性，前提是学生必须能“摸到”老师的一个模式（Mode）。在分布差异过大时，学生处于零概率区间，无法发现老师的模式。
2.  **不稳定性：** 持续的负 Advantage 会导致 $P_S$ 在当前区域迅速塌缩，概率质量被随机挤压到未探索的区域，导致模型输出变得随机（Entropy 增加）或陷入死循环。

**最终证明：** 在没有重叠分布的前提下，OPD 无法建立有效的优化路径，训练必将失败。
