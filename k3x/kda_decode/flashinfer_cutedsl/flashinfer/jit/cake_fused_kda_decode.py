"""Cake / FlashKDA 后端是 csrc/kda 下 400 多个生成的 CUDA 变体（约 76 MB），经 flashinfer.jit 构建；副本只保留 CuTe DSL 路径。

这些名字在上游是变体枚举和模块加载函数；副本里变体表为空、加载函数直接报错。走到 cake / flash-kda 分支时请改用已安装的 flashinfer。
"""

from typing import Any


def _unavailable(*args, **kwargs):
    raise NotImplementedError("研读副本不含 Cake/FlashKDA CUDA 后端，请用已安装的 flashinfer 或选择 backend='cute-dsl'")


def __getattr__(name: str) -> Any:
    if name.isupper():
        return ()
    if name[:1].isupper():
        return Any
    return _unavailable
