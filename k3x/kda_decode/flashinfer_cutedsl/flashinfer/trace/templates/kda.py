"""上游是 KDA 的 FlashInfer-Bench TraceTemplate（op_type="kda"）；副本里只作占位，原文见 third_party/flashinfer/flashinfer/trace/templates/kda.py。"""


def __getattr__(name: str):
    return None
