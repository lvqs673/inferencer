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
            yield (beg, end, self.data[beg:end])
            beg = end

    def inference(self) -> list[Any]:
        input_queue = mp.Queue(maxsize=self.world_size)
        result_queue = mp.Queue(maxsize=self.world_size)
        done_event = mp.Event()

        for input_args in self.split_data():
            input_queue.put(input_args)

        procs = []
        for rank in range(self.world_size):
            inference_args = {
                "local_rank": rank,
                "model": self.model,
                "forward_fn": self.forward_fn,
                "batch_size": self.batch_size,
            }
            proc = mp.Process(
                target=inference_worker,
                args=(inference_args, input_queue, result_queue, done_event),
            )
            proc.start()
            procs.append(proc)

        # results[i]第i个进程计算的每个批次的结果的列表
        results: list[tuple[int, int, list[Any]]] = [
            result_queue.get() for _ in range(self.world_size)
        ]
        done_event.set()
        for proc in procs:
            proc.join()

        results.sort(key=lambda x: x[0])

        # 返回每个batch的结果的列表
        return [output for beg, end, result in results for output in result]


# 用来推理的进程
def inference_worker(
    inference_args: dict[str, Any],
    input_queue: mp.Queue,
    result_queue: mp.Queue,
    done_event: mp.Event,
):
    local_rank = inference_args["local_rank"]
    model = inference_args["model"]
    forward_fn = inference_args["forward_fn"]
    batch_size = inference_args["batch_size"]

    device = torch.device(f"cuda:{local_rank}")
    forward_fn.device = device
    model.to(device)
    model.eval()

    (beg, end, data) = input_queue.get()
    n = len(data)
    n_batch = n // batch_size + (n % batch_size != 0)

    results = []
    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, n, batch_size), 1):
            j = min(n, i + batch_size)
            outputs = forward_fn(model, data[i:j])
            results.append(outputs)

    result_queue.put((beg, end, results))

    while not done_event.is_set():
        done_event.wait()
