"""上游 sglang.kernels.jit.utils 是 tvm-ffi JIT 编译框架；这里只需要 is_arch_support_pdl（非 HIP 且 arch major >= 9）。"""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def is_arch_support_pdl() -> bool:
    if torch.version.hip is not None or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 9
