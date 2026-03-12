Pytorch官方文档：[https://docs.pytorch.org/docs/stable/distributed.tensor.html#pytorch-dtensor-distributed-tensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html#pytorch-dtensor-distributed-tensor)

### 什么是 DTensor (Distributed Tensor)？

DTensor（分布式张量）是 PyTorch 针对大规模分布式训练引入的一个核心原生抽象（位于 `torch.distributed.tensor` 库中）。它的出现主要是为了解决编写复杂的张量并行（Tensor Parallelism）和数据并行（Data Parallelism）代码时，开发者需要手动管理底层通信算子（如 `all_reduce`、`all_gather`）的痛点。

简单来说，**DTensor 让你能够像操作单机上的普通 Tensor 一样，去操作一个实际上分布在多台机器/多块显卡上的“全局大张量”**。系统会在后台自动为你计算并插入所需的跨设备通信。

#### DTensor 的三大核心概念：

1. **DeviceMesh（设备网格）**
它是集群物理计算资源（GPU/TPU 等）的逻辑拓扑抽象。你可以将 N 个 GPU 组成一个 1 维网格（纯数据并行），或者组成一个 2 维网格（例如行表示数据并行，列表示张量并行）。在复杂的训练框架中（如模型有 4-D 混合并行需求），DeviceMesh 是分配资源的基石。
2. **Placement（分布策略/放置方式）**
它定义了这个全局张量是如何映射和存放到 DeviceMesh 的具体设备上的。主要有三种策略：
* **`Shard(dim)`**：分片。将张量在指定的维度 `dim` 上切开，平分给网格中的不同设备。
* **`Replicate()`**：复制。让网格中的每个设备都持有该张量的完整副本。
* **`Partial()`**：部分值。多卡各自持有一部分中间结果（通常是梯度的累加或求和操作的中间态），在被真正使用前，系统会隐式地触发归约（Reduction）通信把它聚合完整。


3. **Local Tensor（局部张量）**
这是每块 GPU 显存里实实在在存储的底层数据。对于一个全局的 DTensor，它在单卡上的视角就是一个 Local Tensor。

#### 开发与排错中的一个关键准则（非常重要）：

使用 DTensor 时，它遵循 SPMD（单程序多数据）编程范式。**在进行数学运算（如加法、乘法等）时，参与运算的张量必须保持类型一致并处于同一个分布式上下文中**。
如果一个操作数是已经被 `Shard` 切分过的 DTensor，而另一个操作数（例如从外部传入的缓存变量或位置编码变量）还是普通的 `torch.Tensor`，底层的算子分发系统（以及像 Dynamo 这样的计算图编译器）就会因为不知道如何对齐这两个张量而抛出类型混合的错误。正确的做法是在运算前，将那个普通的 Tensor 也注册到相同的 DeviceMesh 上，或者转换为与之兼容的 DTensor。

### 测试题目

3个基本题目：
[https://gemini.google.com/share/4194d263079f](https://gemini.google.com/share/4194d263079f)

10个代码题目
[https://gemini.google.com/share/5f72621970cc](https://gemini.google.com/share/5f72621970cc)
