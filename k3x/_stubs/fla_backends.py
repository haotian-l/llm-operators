"""上游的运行时后端分发会按字符串动态 import fla.ops.<op>.backends；研读副本只保留默认实现，dispatch 是恒等。

上游 fla/backends.py 允许 FlashKDA、TileLang、Gluon、Ascend 等后端通过 @dispatch / @register 在运行时替换函数。
研读副本里 register 也是空操作：备选后端（例如 attnres 的 Gluon 版）需要直接调用其函数。
"""


class BaseBackend:
    backend_type = "default"


def dispatch(entry_point):
    if isinstance(entry_point, str):
        return lambda fn: fn
    return entry_point


def register(entry_point, *, backend=None, verifier=None, env_var=None):
    return lambda implementation: implementation
