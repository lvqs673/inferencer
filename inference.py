import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from numpy import ndarray
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Any


# 给的模型和输入数据的情况下进行推理
class Forward(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.device = None  #  device为rank所对应的显卡

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    @abstractmethod
    def forward(self, model: nn.Module, batch_data: Any) -> Any:
        pass


"""
一个例子：
class MyForward(Forward):    
    def forward(self, model: nn.Module, batch_data: list[int]) -> Tensor:
        inputs = collate_fn(batch_data)  # pad to max_length and convert to Tensor
        inputs = inputs.to(self.device)
        return model.forward(inputs).cpu()
"""


# 单结点多进程推理
class Inferencer:
    def __init__(
        self,
        model: nn.Module,
        data: list | ndarray | Tensor,
        batch_size: int,
        forward_fn: Forward,
        nproc: int,
    ):
        self.world_size = nproc
        self.forward_fn = forward_fn
        self.model = model
        self.data = data
        self.batch_size = batch_size

    # 把 data 平均分成 world_size 段
    def split_data(self):
        c, r = divmod(len(self.data), self.world_size)
        beg = 0
        for i in range(self.world_size):
            end = beg + c + (i < r)
            yield self.data[beg:end]
            beg = end

    def inference(self) -> list[Any]:
        results = [None] * self.world_size
        with mp.Pool(processes=self.world_size) as pool:
            results_futures = [
                pool.apply_async(
                    func=inference_worker,
                    args=(rank, self.model, sub_data, self.batch_size, self.forward_fn),
                )
                for rank, sub_data in enumerate(self.split_data())
            ]

            for f in results_futures:
                rank, result = f.get()
                results[rank] = result

        # 返回每个batch的结果的列表
        return [output for result in results for output in result]


# 用来推理的进程
def inference_worker(
    rank: int,
    model: nn.Module,
    data: list | ndarray | Tensor,
    batch_size: int,
    forward_fn: Forward,
):
    device = torch.device(f"cuda:{rank}")
    forward_fn.device = device
    model.to(device)
    model.eval()

    n = len(data)
    n_batch = n // batch_size + (n % batch_size != 0)

    results = []
    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, n, batch_size), 1):
            j = min(n, i + batch_size)
            outputs = forward_fn(model, data[i:j])
            results.append(outputs)

    return rank, results
