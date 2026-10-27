"""序列并行下 AttnRes 的 all-gather / reduce-scatter 需要多卡通信组，研读副本只保留单卡路径。"""


def attn_res_all_gather(*args, **kwargs):
    raise NotImplementedError("研读副本不含多卡通信")


def reduce_scatter_attn_res(*args, **kwargs):
    raise NotImplementedError("研读副本不含多卡通信")
