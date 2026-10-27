"""AMD ROCm 的 AttnRes 路径不在研读范围内。"""


def supports_attn_res_hip(*args, **kwargs):
    return False


def attn_res_hip(*args, **kwargs):
    raise NotImplementedError("研读副本不含 ROCm 路径")
