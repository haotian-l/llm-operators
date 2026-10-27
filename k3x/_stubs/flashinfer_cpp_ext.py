"""上游 flashinfer/jit/cpp_ext.py 是 nvcc/ninja 构建工具；KDA kernel 只用到 is_cuda_version_at_least，这里按 torch 编译时的 CUDA 版本判断。"""

import torch
from packaging.version import Version


def is_cuda_version_at_least(version_str: str) -> bool:
    return torch.version.cuda is not None and Version(torch.version.cuda) >= Version(version_str)
