[https://verl.readthedocs.io/en/latest/advance/fully_async.html](https://verl.readthedocs.io/en/latest/advance/fully_async.html)

## 重点解析：全异步策略训练 (Fully Async Policy Trainer)

你提供的文档主要介绍的是 verl v0.7 引入的“大招”——**全异步策略训练**。这是目前业内提升强化学习训练效率的前沿技术。

### 1. 核心原理

传统的同步训练（Colocate Sync）是“串行”的：生成完数据 -> 训练 -> 训练完 -> 再生成。此时如果有一条数据生成的特别慢（长尾问题），GPU 就会空转等待。

**全异步模式**将资源彻底解耦：

* **训练器 (Trainer)** 和 **生成器 (Rollouter)** 使用物理隔离的 GPU 资源。
* **并行化**：Trainer 在训练旧数据时，Rollouter 已经在利用新参数生成下一批数据了。
* **数据流**：通过 `MessageQueue` 实现流式传输，Trainer 不再需要等一大批数据全部生成完才开始工作。

### 2. 四种工作模式

文档中提到的模式演进，反映了效率的逐步提升：

* **Mode 1: On-policy Pipeline** (最稳)：同步流式，减少了长尾等待，但仍有部分气泡。
* **Mode 2: Stream Off-policy**：通过增加本地更新次数减少同步频率。
* **Mode 3: Async Stream (Stale Samples)**：允许使用“过期”参数生成的数据，进一步填满 GPU 气泡。
* **Mode 4: Async Stream + Partial Rollout** (最快)：引入“中断机制”，当需要同步参数时，不必等当前的生成任务全部做完，直接中断并保存状态，同步后再继续。

没问题。`verl` 的全异步策略训练（Fully Async Policy Trainer）涉及多个维度的资源调度和算法平衡。我们可以把这些参数分为 **资源分配**、**数据流控制**、**同步策略** 和 **工程优化** 四个类别来理解。

## 参数介绍

### 1. 资源分配参数 (Resource Allocation)

这部分定义了训练器（Trainer）和生成器（Rollouter）各占多少“地盘”。

* **`trainer.nnodes` / `trainer.n_gpus_per_node**`:
* 定义训练节点数和单节点 GPU 数。这部分显卡专注于梯度计算和参数更新。


* **`rollout.nnodes` / `rollout.n_gpus_per_node**`:
* 定义生成节点数和单节点 GPU 数。这部分显卡专门跑推理（通常对接 vLLM），负责产出经验数据。
* **关键点**：在异步模式下，这两部分的 GPU 是物理隔离的。



---

### 2. 数据与任务量参数 (Data & Tasks)

控制任务的总规模和每一批次的大小。

* **`rollout.total_rollout_steps`**:
* **含义**：整个训练过程计划生成的总样本数。
* **公式**：在同步模式下它等于 `训练步数 * train_batch_size`。在异步模式下，它是 Rollouter 的“总工作量”。


* **`data.train_batch_size`**:
* **特别注意**：在全异步策略中，这个值通常设为 **0**（无效），因为训练不再按固定的 Batch 步进，而是由 `require_batches` 驱动。


* **`data.gen_batch_size`**:
* **含义**：Rollouter 每次向推理引擎发送的 Prompt 数量。通常设为 1，以配合流式（Streaming）产出。


* **`actor_rollout_ref.actor.ppo_mini_batch_size`**:
* **含义**：训练时的最小批次大小。这是模型执行一次反向传播（BP）所用的样本数。



---

### 3. 异步与新鲜度控制 (Async & Staleness)

这是全异步模式的核心，决定了训练的“新鲜度”和“效率”。

* **`async_training.require_batches`**:
* **含义**：Trainer 攒够多少个 `ppo_mini_batch_size` 才开始一次训练。
* **作用**：为了稳定训练，Trainer 不会出一个数据就练一次，而是等凑够一定数量（例如 4 个 batch）再开始更新，以保证梯度的统计稳定性。


* **`async_training.trigger_parameter_sync_step`**:
* **含义**：Trainer 每进行多少次局部更新（Local Updates），就向 Rollouter 推送一次新参数。
* **作用**：控制同步频率。值越大，训练越快（同步开销小），但数据的 Off-policy 程度越高（数据越“旧”）。


* **`async_training.staleness_threshold`**:
* **含义**：过期样本阈值（新鲜度控制）。
* **作用**：如果设为 0.5，表示允许训练数据中最多有 50% 是由“旧版本参数”生成的。如果 Rollouter 产生太快，超过这个限制，它会停下来等 Trainer 更新。


* **`async_training.partial_rollout`**:
* **含义**：是否开启“部分生成”。
* **作用**：开启后，如果 Trainer 要同步参数，Rollouter 会立刻中断当前的生成（哪怕只生成了一半），更新完参数再接着剩下的生成。这极大减少了 Trainer 等待生成器的时间。



---

### 4. 算法与工程优化 (Algorithm & Performance)

确保算法正确性和系统运行效率。

* **`async_training.use_rollout_log_probs`**:
* **含义**：直接使用 Rollouter 生成时计算的 Logits。
* **作用**：在 PPO/GRPO 中计算重要性采样比率（Importance Sampling）需要旧参数的概率。直接用 Rollout 产出的值可以省去 Trainer 重新计算的开销。


* **`async_training.checkpoint_engine.enable`**:
* **含义**：开启高效参数同步引擎。
* **作用**：利用 NCCL 算子在显存间直接广播参数，比通过磁盘或常规 PyTorch 序列化快得多。


* **`async_training.use_trainer_do_validate`**:
* **含义**：是否让 Trainer 节点执行验证（Validation）。
* **作用**：通常生成器很忙，让相对空闲的训练节点在更新间隙跑验证，能进一步压缩总时间。

## 模式支持

### 1. On-policy Pipeline (基础同步流水线)

这是最保守的模式，严格遵守 On-policy（同策略）原则。

* **配置**：同步步数为 1，不容忍任何过期数据（`staleness_threshold=0`）。
* **逻辑**：Rollouter 生成一批数据  Trainer 练完这批数据  同步参数  进入下一次生成。
* **优缺点**：
* **优点**：算法最稳，数据新鲜度 100%。
* **缺点**：**浪费严重**。如果生成器中有个别样本生成特别慢（长尾问题），Trainer 只能干等着，反之亦然。



---

### 2. Stream Off-policy Pipeline (流式异步，但无过期数据)

引入了“流式”概念，通过增加每次同步前处理的数据量来填补空隙。

* **配置**：同步步数 > 1，但不允许过期数据。
* **逻辑**：Rollouter 连续生成多批数据。Trainer 拿到第一批就开始练，不用等全部生成完。但在**所有**预定批次练完并同步参数前，Rollouter 不会开始下一轮大规模生成。
* **优缺点**：
* **优点**：利用流式传输，Trainer 的等待时间减少了。
* **缺点**：在第一轮开始（Trainer 等 Rollouter）和每一轮结束（Rollouter 等 Trainer 同步）时，依然存在明显的**流水线气泡（Idle）**。



---

### 3. Async Stream with Stale Samples (允许过期样本的异步流)

真正实现了 Trainer 和 Rollouter 的并行化，允许“一边练，一边跑”。

* **配置**：允许过期数据（`staleness_threshold > 0`），但不中断任务。
* **逻辑**：即使 Trainer 还没练完，Rollouter 也可以继续用旧参数产生下一波数据（Stale Samples）。这样当 Trainer 练完当前批次时，队列里已经有现成的数据可以练了。
* **优缺点**：
* **优点**：消除了 Trainer 等待第一批数据的尴尬，效率大幅提升。
* **缺点**：如果在同步参数时 Rollouter 还有正在跑的任务，它必须等这些任务**彻底跑完**才能更新参数。这依然会产生一段等待时间。



---

### 4. Async Stream with Partial Rollout (带“断点续传”的极致异步)

这是目前最先进的模式（对应你之前提到的 PipelineRL-k 思想）。

* **配置**：允许过期数据 + 开启 `partial_rollout=True`。
* **逻辑**：当 Trainer 算完并准备同步参数时，如果 Rollouter 还在生成 Token，**立刻叫停（Interrupt）**。Rollouter 更新参数后，在刚才中断的地方接着生成。
* **优缺点**：
* **优点**：**几乎消除了所有气泡**。Rollouter 不再需要为了同步参数而强行跑完长尾样本。
* **效果**：实验证明这种模式提速最明显（2.3x 以上），且通过中断机制保证了新生成的 Token 尽可能使用了最新的参数。


## 指标

### 一、 核心指标 (Key Metrics)：系统的“体检表”

这些指标可以帮你判断流水线哪里堵塞了，以及数据是否“太旧”。

#### 1. 效率类指标（判断资源是否浪费）

* **`trainer/idle_ratio` (训练器空闲率)**：
* 如果这个值高，说明训练器在干等数据。原因可能是 Rollouter 太慢，或者数据传输太堵。


* **`rollouter/idle_ratio` (生成器空闲率)**：
* 如果这个值高，说明生成器在等训练器更新参数，或者在等同步。



#### 2. 数据新鲜度类指标（判断算法风险）

* **`stale_samples_processed`**：
* 训练中使用了多少“旧参数”生成的样本。这个数如果非常大，可能会导致模型跑偏（Off-policy 偏差过大）。


* **`partial_ratio` & `max_partial_span**`：
* **Ratio**：在一轮训练中，有多少比例的样本是经过“中断再恢复（Partial）”生成的。
* **Span**：这些中断样本横跨了多少个参数版本。Span 越大，说明同一个回复中 Token 之间的参数差异越大，理论上风险越高。



---

### 二、 调优建议 (Tuning Recommendations)：你的“调节旋钮”

#### 1. 资源分配的“天平”原则

这是最重要的工程建议：**让 Rollout 时间和 Train 时间尽可能接近。**

* **动作**：观察上面的两个 `idle_ratio`。
* **法则**：
* `rollout/idle` 高  Rollouter 资源过剩  **拨几台显卡给 Trainer**。
* `trainer/idle` 高  Trainer 资源过剩  **拨几台显卡给 Rollouter**。



#### 2. 关键旋钮的调节效果

| 参数 | 往大调的效果 | 往小调的效果 |
| --- | --- | --- |
| **`staleness_threshold`** | **效率更高**，队列永远是满的。但模型可能因为数据太旧而不收敛。 | **模型更稳**，更接近 On-policy。但可能产生流水线气泡。 |
| **`require_batches`** | **训练更稳**，梯度统计更准。但启动训练的等待时间变长。 | **更接近实时流**，速度极快。但 batch 太小可能导致训练抖动。 |
| **`trigger_parameter_sync_step`** | **吞吐量极大**，减少同步开销。但模型参数更新滞后，风险大。 | **数据最新**，算法最安全。但频繁同步参数会导致严重的通信开销。 |

---

### 三、 场景化模式选择策略

文档最后给出了三类任务的“套路”建议：

1. **打比赛/做研究（追求稳定）**：
* 选择 **Mode 1**。
* **策略**：严格同步，不容忍过期。虽然慢点，但不容易出玄学问题。


2. **吞吐量优先，但不敢尝试 Off-policy（中庸方案）**：
* 选择 **Mode 2**。
* **策略**：用流式传输减少等待，但不允许使用旧参数（`staleness=0`）。


3. **工业级大规模生产（追求极致性能）**：
* 选择 **Mode 4**（Partial Rollout）。
* **策略**：开启 `partial_rollout` 和 `staleness_threshold > 0`。这是 Qwen2.5/DeepSeek 等大规模任务实现 2 倍以上加速的关键。



---

### 总结建议

如果你现在开始跑实验，建议先看 **`trainer/idle_ratio`**。

* 如果你发现它大于 20%，说明你的生成器（Rollout）是瓶颈，你应该增加 Rollouter 的 GPU 数量，或者适当调大 `staleness_threshold` 允许训练器先练着旧数据。

**你想针对你目前的模型规模（比如 7B 还是 70B）和显卡数量，让我帮你预估一个初始的资源分配比例吗？**

