"""上游 vllm.triton_utils 会在没有 Triton 时提供占位对象并牵出 vllm 主包；副本直接用 triton。"""

import triton
import triton.language as tl

__all__ = ["triton", "tl"]
