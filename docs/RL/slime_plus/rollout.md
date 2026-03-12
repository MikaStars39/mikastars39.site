从上到下，函数的职责从“业务入口”逐渐深入到“单一请求”：

- **入口层 (generate_rollout)**:
这是最外层的 API。它是一个同步函数，负责根据 evaluation 参数判断是走“训练采样”还是“评估采样”。它使用 run() 将内部的异步逻辑包装起来。

- **编排层 (generate_rollout_async / eval_rollout)**:
负责控制整个 Batch 的进度。它会不断向下一层提交任务，并监控“好样本”是否拿够了。

- **组合层 (generate_and_rm_group)**:
处理一组样本（Group）。在 GRPO 等算法中，一个 Prompt 对应多个输出。这个函数会把一组请求并发发出去，并处理GRPO的reward（Group RM）。

- **任务层 (generate_and_rm)**:
单一请求的完整闭环。它负责：1. 采样 (Generate) -> 2. 算分 (Reward Model)。

- **原子层 (generate)**:
最底层的 HTTP 通信。它负责把 Prompt 发送给 SGLang Router，并拿回生成的 Token 和 Logprobs。

那么，如果我们想要设计一个离线推理引擎，哪些部分是不适合的呢？

我们首先从数据吞吐量的不同切入。对于RL的rollout，它要做的事情，举例子来说就是“得到1024个examples, 每个roll8次，算分，然后返回“。显然，这个任务有两个特点：

1. **数据量固定且吞吐量不大**：在例子中，就是1024*8个需要生成的样本，我们很难想象一个batch会roll100万个样本。在一个不大的固定数据量下，我们其实不需要对数据生产的吞吐有很多的优化（例如使用什么装箱算法重新排序）。
2. **必须把样本和reward都算好**：当然！否则无法训练。

但是，在离线推理中，这两条都是不成立的。

1. 离线推理可能需要一次性处理大量的样本，所以如何把这些样本高效率，以一个最优的方式喂给推理引擎是最重要的。
2. 离线推理完全可以先roll完所有样本，再算每个是否正确（比如评测），甚至可以完全不算reward（比如生产数据）。

因此，我们返回来思考，slime这里的rollout设计，是否完全适用于离线推理呢？让我们分析一下：

1. **入口层 (generate_rollout)**：他非常简洁，我们暂且就当他【**可以保留**】。
2. **编排层 (generate_rollout_async / eval_rollout)**：显然是【**无法保留**】，
原因也很简单：等一个batch处理完再处理下一个也太蠢了。当我们遇到长尾的样本的时候，大家都要停下来等他roll完，这个时候所有的资源都在闲置。
我们需要做的事情是持续的向router喂请求让他吃饱，而不是一定要维持一个固定的batch size。

> p.s., slime其实有一个fully_async的实现，其中用了一个基于生产者-消费者模型的流式Rollout worker来实现rollout和训练的解耦，这个其实已经非常接近我们离线推理的设计。
> 但是鉴于这个里面加了一些abort和不太优雅的超参数（比如写死了queue的上限是1000），而且只有一个文件，所以我们还是准备自己实现。

3. **组合层 (generate_and_rm_group)**:显然是【**无法保留**】,因为离线推理完全可以先roll完所有样本，再算每个是否正确（比如评测），甚至可以完全不算reward（比如生产数据）
4. **任务层 (generate_and_rm)**：显然是【**无法保留**】，理由同上

## 数据读取设计(Data Buffer)

slime的数据设计实现非常简洁，主要结构在`slime/ray/rollout_data_source.py`，通过两个核心类`RolloutDataSource`和`RolloutDataSourceWithBuffer`来管理数据。主要实现的功能包括：读取数据，根据`max_token_per_gpu`打包micro batch，`apply_chat_template`，根据key处理数据。

> 这里我不明白的一点是为什么slime不用hf默认的datasets来处理，AReal就是这样用的：
> 
> AReaL 通过 get_custom_dataset 加载训练数据集，支持 HuggingFace datasets，并在控制器上加载后由调度器分发到工作节点；SFT 与 GRPO 的数据格式与处理流程不同，可通过 TrainDatasetConfig/ValidDatasetConfig 配置路径、分片、批大小与最大长度等参数
> 
> --- from deepwiki
> 
所以这里我还是准备基于datasets自己实现一个类似的逻辑。
我们分析data source主要需要实现的功能：

```
class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Length of the data source. May change when samples are added/fetched.
        """
```
