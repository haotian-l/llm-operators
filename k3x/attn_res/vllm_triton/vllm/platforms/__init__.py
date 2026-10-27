"""上游 vllm.platforms 会牵出整个 vllm；副本只保留 attn_res.py 用到的两个查询，语义照抄上游 CUDA 平台实现。"""

import torch


class _CudaPlatform:
    def is_device_capability_family(self, capability: int) -> bool:
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        return (major * 10 + minor) // 10 == capability // 10

    def is_arch_support_pdl(self) -> bool:
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


current_platform = _CudaPlatform()
