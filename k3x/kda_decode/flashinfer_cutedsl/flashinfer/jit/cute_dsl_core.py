"""上游 build_and_load_cute_dsl_kernel 带磁盘缓存并牵出 flashinfer.jit 框架；副本只在进程内缓存，直接调用 compile_fn。

compile_fn 内部就是 cute.compile(..., options="--enable-tvm-ffi")，返回的可调用对象和上游一致；区别只是每个进程都要重新编译。
"""

from typing import Any, Callable, Sequence

_cache = {}


def build_and_load_cute_dsl_kernel(
    module_name: str,
    kernel_name: str,
    compile_fn: Callable[[], Any],
    extra_key_files: Sequence[str] = (),
) -> Any:
    key = (module_name, kernel_name)
    if key not in _cache:
        _cache[key] = compile_fn()
    return _cache[key]
