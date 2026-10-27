"""上游 RMSNorm 带多种 kernel 后端并牵出 sglang.srt；副本用纯 PyTorch 版本，只保留 weight、variance_epsilon 和 forward。"""

import torch


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight.float() * x).to(dtype)
