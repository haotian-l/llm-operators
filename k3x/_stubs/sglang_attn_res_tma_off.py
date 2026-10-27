"""上游这里是 TMA 快速路径的 tvm-ffi JIT 封装（会牵出 JIT 加载器）；CUDA 源码单独抽取在 attn_res/sglang_cuda。

研读副本在 SM10x/SM11x 上调用 _aggregate 会走到这里并报错；要看 Triton 路径，直接调用 _aggregate_fused。
"""


def attn_res_fused_tma(*args, **kwargs):
    raise NotImplementedError("研读副本不含 SGLang 的 JIT 加载器；TMA 快速路径见 k3x/attn_res/sglang_cuda")
