"""上游 sglang/srt/utils/__init__.py 会牵出整个 sglang.srt；这里只保留 kernel 文件用到的平台探测函数，语义照抄上游。

上游 get_device_sm 会优先走 NVML 以避免过早初始化 CUDA context，这里直接用 torch.cuda，结果相同。
"""

from functools import lru_cache

import torch


def is_hip() -> bool:
    return torch.version.hip is not None


@lru_cache(maxsize=1)
def is_cuda():
    return torch.cuda.is_available() and torch.version.cuda is not None


def is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def get_device_sm():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    return 0


def is_gfx95_supported():
    if torch.version.hip:
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx95"])
    return False


@lru_cache(maxsize=1)
def is_sm90_supported():
    if not is_cuda():
        return False
    return get_device_sm() // 10 == 9 and tuple(map(int, torch.version.cuda.split(".")[:2])) >= (12, 3)
