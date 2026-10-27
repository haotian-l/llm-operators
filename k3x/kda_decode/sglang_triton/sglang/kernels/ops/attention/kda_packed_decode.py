"""上游这里会转去 CUDA 的 kda_packed_decode 并牵出 JIT 加载器；CUDA 版本单独抽取在 sglang_cuda，副本始终走 Triton。"""


def covered(*args, **kwargs):
    return False
