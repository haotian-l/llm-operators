"""上游 flashinfer/utils.py 会牵出 flashinfer 的 JIT 和库加载；KDA kernel 只用到 get_compute_capability，语义照抄上游。"""

from typing import Tuple

import torch


def get_compute_capability(device: torch.device) -> Tuple[int, int]:
    if device.type != "cuda":
        raise ValueError("device must be a cuda device")
    properties = torch.cuda.get_device_properties(device)
    return properties.major, properties.minor
