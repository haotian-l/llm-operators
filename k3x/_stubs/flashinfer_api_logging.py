"""上游 flashinfer_api 负责 API 日志和 FlashInfer-Bench trace 采集；副本里是恒等装饰器。"""


def flashinfer_api(func=None, **kwargs):
    if func is None:
        return lambda f: f
    return func
