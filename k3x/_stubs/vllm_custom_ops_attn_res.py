"""上游 vllm._custom_ops 会加载 vllm 的 _C 扩展并牵出整个 vllm；AttnRes 的 CUDA 版单独抽取在 attn_res/vllm_cuda。"""


def kimi_k3_attn_res(*args, **kwargs):
    raise NotImplementedError("研读副本不含 vllm._C；CUDA 版本见 k3x/attn_res/vllm_cuda")
