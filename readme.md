***一个单结点多进程推理的框架***

1. inference.py: 推理框架的实现
2. example.py: 一个简单的使用例子



**多进程推理框架**
- 在 **Inferencer** 类中实现
- 在 **Inferencer** 的构造方法中需要给出下面五个参数：
    - 模型 model
    - 数据 data (list | Tensor | ndarray)
    - 批次大小 batch_size
    - 推理方法 forward_fn (给定 model 和 batch_data, 计算想要的推理结果)
    - 进程数 nproc (每个进程使用一个 GPU)
- 执行 *inference* 方法后会自动启动 nproc 个进程进行推理
- *inference* 的过程:
    - 将 data 平均分割为 nproc 段
    - 启动 nproc 个子进程, 每个进程上推理对应的数据段
    - 主进程收集所有子进程的推理结果并进行汇总

- 设置 **Forward** 类是为了提供更灵活的推理, 通过实现 *forward* 方法, 可以定义自己的推理方法。 例如当 batch_data 是一批句子时, 需要进行截断和填充, 此时可以定义如下 **Forward** 类
```
class MyForward(Forward):
    def forward(self, model: nn.Module, batch_data: list[int]) -> Tensor:
        inputs = collate_fn(batch_data)  # pad to max_length and convert to Tensor
        inputs = inputs.to(self.device)
        return model.forward(inputs).cpu()
        
forward_fn = MyForward()
```
- 一个推理的使用示例:
```
batch_outputs = Inferencer(
    model=model,
    data=data,
    batch_size=batch_size,
    forward_fn=forward_fn,
    nproc=nproc,
).inference()
```

- 输出的结果是一个 *list*, 它的每个元素是每个 batch 的结果而不是每个样本的推理结果, 之所以这么设置是因为当每个 batch 的输出都是 Tensor 或 ndarray 时, 通过 *torch.cat()* 或 *np.concatenate()* 合并比使用 *list.extend()* 更加高效，让使用者自己决定如何合并
