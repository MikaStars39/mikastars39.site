在分布式推理系统（如 SGLang）中，Router 的核心任务是**请求分发（Request Dispatching）**。其性能不仅取决于负载均衡（Load Balancing），更取决于如何通过维护 **KV Cache 复用**来降低首字延迟（TTFT）。

以下是对这些调度策略及最佳实践的深度解析：

---

### 1. 调度策略（Routing Policies）对比

| 策略 | 核心逻辑 | 适用场景 |
| --- | --- | --- |
| **`cache_aware`** | **缓存感知**。优先将请求发送到已缓存该请求前缀（Prefix）的 Worker。 | **推荐生产环境使用**。尤其是长对话、多轮 Chat 或 RAG 场景。 |
| **`round_robin`** | **轮询**。按顺序循环分发请求到每个 Worker。 | 负载非常均匀且请求之间没有任何前缀重合的情况。 |
| **`random`** | **随机**。从可用列表中随机抽取一个 Worker。 | 性能基准测试（Baseline）或极简部署。 |
| **`power_of_two`** | **双选最优**。随机采样两个 Worker，对比它们的负载（队列长度），选择较轻的一个。 | 高并发、低延迟要求，且不考虑缓存复用的场景。 |

---

### 2. Cache-Aware（缓存感知）策略深度拆解

这是 SGLang Router 的灵魂，其目标是最大化 **Radix Tree（基数树）** 的命中率。

#### 核心机制：

1. **近似基数树（Approximate Radix Tree）**：Router 在内部为每个 Worker 维护一个轻量级的树结构，记录该 Worker 处理过的 Token 前缀。
2. **前缀匹配**：当新请求到达时，Router 计算其前缀与哪个 Worker 的缓存重合度最高。
3. **负载回退（Load Fallback）**：如果最优缓存节点的负载超过了阈值（由 `balance-rel-threshold` 等参数控制），Router 会放弃缓存一致性，转而选择负载更低的节点，防止单个节点被“热点请求”打死。

#### 关键参数解析：

* **`--cache-threshold 0.5`**：只有当缓存匹配度超过 50% 时，才触发缓存导向的路由，否则退化为普通负载均衡。
* **`--balance-abs-threshold 32`**：当两个 Worker 之间的待处理请求数差距超过 32 时，触发强制负载均衡。
* **`--max-tree-size 67108864`**：控制 Router 内存中缓存元数据的大小。对于超长上下文场景，需要调大此值以维持精确度。

---

### 3. 最佳实践（Best Practices）逻辑

#### 使用 `launch_server` (SMG) 而非 `sglang.launch_server`

* **SMG (SGLang Multi-GPU)**：这是一个更高级的封装，集成了健康检查、Prometheus 监控以及多模型支持。在生产环境中，它能提供更好的**可观测性**和**自愈能力**。

#### 启用健康检查（Health Checks）

* **`--router-health-check-interval-secs 30`**：
分布式系统中，Worker 可能会因为显存溢出（OOM）或硬件错误掉线。Router 必须具备主动探测能力，在 30 秒内发现异常并自动摘除故障节点，避免请求“掉进黑洞”。

#### 监控与度量（Prometheus）

* **`--router-prometheus-port 10001`**：
这是运维的关键。通过暴露的指标，你可以实时观察到：
* **Cache Hit Rate**：缓存感知策略的效果。
* **Throughput/Latency**：整体吞吐量和 P99 延迟。
* **Load Distribution**：各个 GPU 节点的负载是否均衡。



---

### 4. 推荐生产命令示例

```bash
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4 \  # 启动 4 个数据并行副本（通常对应 4 张或 4 组卡）
    --router-policy cache_aware \
    --router-health-check-interval-secs 30 \
    --router-prometheus-port 10001 \
    --host 0.0.0.0 \
    --port 30000

```

**系统架构逻辑总结：**
该配置通过 `dp-size 4` 构建了一个 4 节点的计算网格（Mesh），前端通过 `cache_aware` 路由实现“流量粘性”，确保长文本请求能够精准命中 KV 缓存，从而在不增加硬件开销的情况下，显著降低延迟并提升系统整体的吞吐上限。

需要我为您详细推演在 `dp-size` 较大时，`cache_aware` 策略如何在高并发压力下进行动态重平衡吗？
