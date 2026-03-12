深入浅出理解slime sft的实现：如何在一个RL训练框架中实现sft？

[https://thudm.github.io/slime/zh/examples/qwen3-4b-base-openhermes.html](https://thudm.github.io/slime/zh/examples/qwen3-4b-base-openhermes.html)

首先注意到slime在sft启动的时候使用的是train_async

slime使用ray来管理资源，同时通过ray来实现推理和训练的交错（interleave）进行。
这里第一部分首先创建资源组（1gpu 1cpu绑定并且通过global rank重排，保证分配顺序合理），Rollout Manager，以及训练的模型（Actor）。
并且在最开始同步推理引擎的权重，通过compare操作保证最开始的推训一致性。

rollout的部分建议看[https://zhuanlan.zhihu.com/p/1946402397409740613](https://zhuanlan.zhihu.com/p/1946402397409740613)，非常精彩的解释。

```python
import ray
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import should_run_periodic_action

def train(args):
    # 强制要求非共置模式，意味着训练和推理（Rollout）必须在不同的物理资源上进行，这是异步的前提。
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    
    # 1. 分配 GPU 资源组（Placement Groups）
    pgs = create_placement_groups(args)
    init_tracking(args)

    # 2. 创建推理管理器（Rollout Manager），内部包含 sglang 推理引擎
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # 3. 创建 Actor 和 Critic 模型（训练模型）
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # 4. 初始权重同步：确保推理引擎拥有最新的模型权重
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))
```

slime的异步实现并没有使用 Python 标准库的 asyncio，而是使用了 Ray 框架的异步机制。训练大致可以分为两个步骤：

- 动作 A（推理/生成数据）: rollout_manager.generate.remote(...)

- 动作 B（训练模型）: actor_model.async_train(...)

在下面的代码中，标出了这两个步骤。

```python
# 5. 预启动：在进入循环前，先触发第一个 batch 的数据生成
    # 注意这里使用了 .remote()，它是非阻塞的 【动作A】
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # 6. 获取当前需要的数据
        if rollout_data_next_future is not None:
            # ray.get 是阻塞操作，如果数据还没生成完，主线程会在这里等待
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # 7. 提前触发下一次数据生成 (流水线/Prefetching)
        if rollout_id + 1 < args.num_rollout:
            # 关键点：当前循环在训练 rollout_id 的数据时，
            # 系统已经在后台开始生成 rollout_id + 1 的数据了  【动作A】
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        # 8. 异步执行训练
        if args.use_critic:
            # 启动 Critic 训练任务，返回句柄，不阻塞  【动作B】
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            
            if rollout_id >= args.num_critic_only_steps:
                # 启动 Actor 训练任务，并等待结果  【动作B】
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
            
            # 等待 Critic 训练完成
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
```

这里没有使用类似于async def， wait等异步操作的指令，为什么能够实现异步的推训呢，答案就在ray这里。我们首先需要了解ray的几个基本操作：

[https://zhuanlan.zhihu.com/p/23824881486](https://zhuanlan.zhihu.com/p/23824881486)

.remote(): 这是一个远程调用。主程序不会在这里卡住等待结果，而是立刻拿到一个 future，代表“将来会有数据”。
在slime代码的情境中，程序继续往下跑，去处理当前的训练，而 rollout_manager 在另一台机器/进程上开始干活了。
ray.get(object_ref)`：Ray 是异步的，调用函数后它会立刻返回一个 ID（类似快递单号），代码继续往下跑。当你真正需要结果时，调用 `ray.get(单号)`，程序会暂停等待，直到结果被送回来。


* **场景举例**：
1. **并行数据处理**：如果你有 10000 张图片要预处理，用 `for` 循环很慢。用 Ray 可以瞬间启动 100 个 Worker 并行处理，最后用 `ray.get` 收集结果。
2. **超参数搜索**：同时跑 50 个不同配置的模型训练任务。
3. **在线服务**：部署一个模型作为 Actor，接收 HTTP 请求并推理。


* **用法示范**：

```python
import ray
import time

# 1. 初始化 Ray
ray.init()

# 2. 定义一个耗时任务
@ray.remote
def heavy_computation(x):
    time.sleep(1) # 模拟耗时操作
    return x * x

# 3. 异步调用 (此时不耗时)
# futures 就像是一堆快递单号
futures = [heavy_computation.remote(i) for i in range(5)]

# 4. 同步获取结果 (这里会等待所有任务完成)
# 这一步会大约花费1秒，而不是5秒，因为是并行的
results = ray.get(futures)
print(results) # 输出: [0, 1, 4, 9, 16]

```


## PlacementGroup


```python
import logging
import socket
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from .actor_group import RayTrainGroup
from .rollout import RolloutManager

logger = logging.getLogger(__name__)

@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        # 返回当前节点IP和当前进程可见的GPU ID列表
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]

def sort_key(x):
    index, node_identifier, gpu_id = x
    # 尝试将节点标识符解析为 IP 地址部分，用于排序
    try:
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        try:
            # 尝试将主机名解析为 IP
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # 兜底方案：将字符串转为 ASCII 码数值列表
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)

```

这一部分主要是实现了bundle group的重新排序
* **逻辑**：这是一个使用 `@ray.remote` 装饰的类，意味着它是一个 **Actor**（运行在 Ray 集群中的独立进程）。
* **目的**：它的唯一作用是“占位”和“侦查”。
* `num_gpus=1`：告诉 Ray，启动这个 Actor 需要独占 1 个 GPU。
* `get_ip_and_gpu_id`：当 Actor 启动后，它会报告自己所在的服务器 IP 和被分配的 GPU 编号。


```python
def _create_placement_group(num_gpus):
    """创建一个指定 GPU 数量的放置组 (Placement Group)。"""
    # 1. 定义资源包：每个包需要 1 GPU 和 1 CPU
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    # 2. 创建放置组，策略为 PACK (尽可能紧凑地安排在同一个节点)
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    # 3. 等待放置组资源就绪
    ray.get(pg.ready())

    # 4. 启动临时 Actor 来获取真实的物理 GPU ID
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i, # 指定占这个坑位
                )
            ).remote()
        )
    
    # 5. 并行获取所有 Actor 的位置信息
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    
    # 6. 用完即焚：销毁这些临时 Actor，释放资源给后续真正的训练任务
    for actor in info_actors:
        ray.kill(actor)

    # 7. 对资源进行排序，确保分布式训练时的顺序一致性
    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    
    # 提取排序后的索引和 GPU ID
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    # 打印日志确认分配情况
    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids

```

*(注：后续的 `create_placement_groups`, `create_training_models` 等函数主要是在调用上述基础逻辑进行业务分配，将在解析部分详细说明。)*

---

### 第二步：逐行/逐块 细粒度解析

#### 1. `InfoActor` 类



#### 2. `sort_key(x)` 函数

* **背景**：在多机分布式训练中，不同机器启动的顺序是不确定的。为了保证每次训练或者不同进程间对 GPU 的编号认知一致（比如 Rank 0 永远在 IP 最小的机器上），我们需要排序。
* **解析**：
* 首先尝试把节点标识符（`node_identifier`）直接当作 IP (`192.168.1.1`) 解析并按数字排序。
* 如果失败，尝试当作主机名 (`node-01`) 去 DNS 查询 IP。
* 如果还失败（比如是在特殊的集群环境中），就简单粗暴地把字符串转成 ASCII 码列表进行字典序排序。
* **最终返回**：`(IP地址数字列表, GPU ID)` 的元组，确保先按机器排，再按机器内的 GPU 号排。



#### 3. `_create_placement_group` (核心难点)

这是一个非常经典的 Ray 资源预占模式，我们一步步看：

* **定义 Bundles**：`bundles = [{"GPU": 1, "CPU": 1} ...]`。这里定义了每一个“坑位”需要什么资源。
* **`placement_group(..., strategy="PACK")`**：向 Ray 集群申请一组资源。`PACK` 策略非常有讲究，它要求 Ray 尽可能把这些资源分配在**同一个节点**或**尽可能少**的节点上，以减少网络通信延迟（这对训练速度至关重要）。
* **`ray.get(pg.ready())`**：这是一个阻塞操作。程序会停在这里，直到 Ray 成功在集群里圈到了这块地。如果资源不够，程序就会一直等。
* **“探针”操作 (InfoActor)**：
* Placement Group 只是圈了一块逻辑上的资源，但我们不知道具体的 GPU ID 是多少（比如是卡0还是卡7？）。
* 所以代码启动了 `InfoActor`，并利用 `PlacementGroupSchedulingStrategy` 强制把这个 Actor 塞到刚才圈好的第 `i` 个坑位里。


* **`ray.kill(actor)`**：一旦获取到了 IP 和 GPU ID，这些 `InfoActor` 就没用了。必须把它们杀掉，否则它们会一直占着 GPU，导致后面真正的训练代码（Actor/Critic 模型）无法启动。

#### 4. `create_placement_groups` 函数

* **逻辑**：根据传入的参数 `args`（通常来自命令行），计算总共需要多少 GPU。
* **条件分支**：
* `debug_train_only` / `debug_rollout_only`：调试模式下只申请部分资源。
* `colocate`：如果开启共置，意味着 Actor（训练者）和 Rollout（采样者）可能复用资源或在同一节点。
* `else`（默认）：分别计算 Actor 节点、Critic 节点（如果有）和 Rollout 节点需要的 GPU 总和。


* **切片 (Slicing)**：调用 `_create_placement_group` 拿到一大堆排序好的 GPU 资源后，像切蛋糕一样分给不同的角色：
* 前 `N` 个给 Actor。
* 中间 `M` 个给 Rollout。
* 最后 `K` 个给 Critic。



#### 5. `create_training_models` 函数

* **目的**：真正启动用于训练的类 `RayTrainGroup`。
* **逻辑**：
* 利用刚才分配好的 `pgs["actor"]` 和 `pgs["critic"]` 来初始化训练组。
* **`async_init`**：异步初始化。因为分布式环境初始化很慢（加载模型、同步权重），异步可以避免卡死主线程。
* **`actor_model.connect(critic_model)`**：如果是 Actor-Critic 架构，Actor 需要知道 Critic 的存在（通常是为了计算 Value Loss 或进行参数同步），这一步建立了它们之间的通信链路。



