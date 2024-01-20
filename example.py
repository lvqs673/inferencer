from typing import Any
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.modules import Module
from inference import *

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


"""
分布式推理框架使用示例:
    模型是一个三层的MLP
    数据是随机生成的

注意点:
    在自己实现的Forward类中, 可以自定义更复杂的推理方法
    但需要注意, Tensor必须放到device上, 推理的结果最好放到cpu上避免显存溢出

使用方法:
    给出下面五个参数即可：
    1. 模型 model
    2. 数据 data
    3. 批次大小 batch_size
    4. 推理方法 forward_fn (给定model和batch_data,计算想要的推理结果)
    5. 进程数 nproc (每个进程使用一个gpu)

    
执行下面代码即可得到每个批次的输出:
batch_outputs = Inferencer(
    model=model,
    data=data,
    batch_size=batch_size,
    forward_fn=forward_fn,
    nproc=nproc,
).inference()

"""

dim1 = 1000
dim2 = 1000
nproc = torch.cuda.device_count()

data_size = int(1e6)  # 数据大小
batch_size = 2000


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.Sigmoid(),
            nn.Linear(dim2, dim1),
            nn.ReLU(),
            nn.Linear(dim1, dim2),
            nn.Dropout(),
        )

    def forward(self, x):
        return self.mlp(x)


class MyForward(Forward):
    def forward(self, model: nn.Module, batch_data: torch.Tensor) -> torch.Tensor:
        inputs = batch_data.to(self.device)
        return model(inputs).cpu()


def test_ddp_inference():
    torch.manual_seed(0)

    model = Model()
    data = torch.randn(data_size, dim1)
    forward_fn = MyForward()

    batch_outputs: list[Tensor] = Inferencer(
        model=model,
        data=data,
        batch_size=batch_size,
        forward_fn=forward_fn,
        nproc=nproc,
    ).inference()

    # 如果推理的每个batch都是一个Tensor的话，最后用torch.cat合并一次更高效
    outputs = torch.cat(batch_outputs)

    print(f"len(outputs) = {len(outputs)}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    test_ddp_inference()
