"""上游 ReplicatedLinear 带量化和权重加载逻辑并牵出 sglang.srt；AttnRes 只读它的 weight，副本只保留这个参数。"""

import torch


class ReplicatedLinear(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(output_size, input_size))
