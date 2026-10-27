"""检查当前 Python 环境能否跑 K3 算子 benchmark。

    python scripts/check_env.py [--engine vllm|sglang|auto] [--ncu]

打印关键包版本和 GPU 信息，逐个 import K3 相关的 kernel 入口。--ncu 会实际跑一次 ncu，
检查性能计数器权限（ERR_NVGPUCTRPERM）。选中引擎的必需项失败时返回非零。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import platform
import shutil
import subprocess
import sys

PACKAGES = [
    "torch",
    "triton",
    "flashinfer-python",
    "nvidia-cutlass-dsl",
    "flash-linear-attention",
    "flashinfer-bench",
    "nsight-python",
    "vllm",
    "sglang",
    "sglang-kernel",
    "sgl-deep-gemm",
    "flash_kda",
]

# (说明, 模块, 需要存在的属性；None 表示只检查能否 import)
COMMON_CHECKS = [
    ("FLA KDA", "fla.ops.kda", ["chunk_kda", "fused_recurrent_kda"]),
    ("FLA ShortConvolution", "fla.modules", ["ShortConvolution", "FusedRMSNormGated"]),
    ("flashinfer KDA", "flashinfer", ["fused_kda_decode", "packed_kda_decode", "recurrent_kda"]),
    ("flashinfer MoE", "flashinfer.fused_moe", ["trtllm_fp4_block_scale_moe", "cutlass_fused_moe"]),
    ("flashinfer MLA", "flashinfer", ["BatchMLAPagedAttentionWrapper"]),
    ("flashinfer-bench", "flashinfer_bench", ["Benchmark", "TraceSet"]),
    ("Nsight Python", "nsight", None),
]
ENGINE_CHECKS = {
    "vllm": [
        ("vLLM K3 model", "vllm.models.kimi_k3", None),
        ("vLLM K3 AttnRes", "vllm.models.kimi_k3.nvidia.ops.attn_res", None),
    ],
    "sglang": [
        ("SGLang BaseFusedOp", "sglang.kernels.fused_op", ["BaseFusedOp"]),
        ("SGLang K3 model", "sglang.srt.models.kimi_k3", None),
        ("SGLang AttnRes", "sglang.kernels.ops.attention.attn_res", None),
        ("SGLang KDA fused decode", "sglang.kernels.ops.attention.kda_fused_decode", None),
    ],
}


def version(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "-"


def check_import(module: str, attrs: list[str] | None) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module)
    except Exception as e:  # noqa: BLE001 - 任何 import 错误都要报告
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''}"
    missing = [a for a in attrs or [] if not hasattr(mod, a)]
    if missing:
        return False, "缺少 " + ", ".join(missing)
    return True, ""


def detect_engine() -> str | None:
    for engine in ("vllm", "sglang"):
        if version(engine) != "-":
            return engine
    return None


def gpu_info() -> bool:
    try:
        import torch
    except ImportError:
        print("torch 未安装")
        return False
    if not torch.cuda.is_available():
        print("CUDA 不可用（在 Mac 上是正常的）")
        return False
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  sm_{props.major}{props.minor}  SMs={props.multi_processor_count}  "
          f"mem={props.total_memory / 2**30:.0f} GiB  torch.cuda={torch.version.cuda}")
    return True


def check_ncu() -> bool:
    ncu = shutil.which("ncu")
    if ncu is None:
        print("ncu: 不在 PATH 中")
        return False
    probe = "import torch; x = torch.ones(1, device='cuda'); x.add_(1); torch.cuda.synchronize()"
    cmd = [ncu, "--metrics", "gpu__time_duration.sum", sys.executable, "-c", probe]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    text = out.stdout + out.stderr
    if "ERR_NVGPUCTRPERM" in text:
        print("ncu: 没有性能计数器权限（ERR_NVGPUCTRPERM），需要管理员开放或容器加 --cap-add SYS_ADMIN")
        return False
    if "gpu__time_duration.sum" not in text:
        print(f"ncu: 运行异常（exit {out.returncode}）\n{text[-800:]}")
        return False
    print("ncu: 计数器权限正常")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", choices=["vllm", "sglang", "auto"], default="auto")
    parser.add_argument("--ncu", action="store_true", help="实际跑一次 ncu 检查计数器权限")
    args = parser.parse_args()

    engine = detect_engine() if args.engine == "auto" else args.engine
    print(f"Python {platform.python_version()} on {platform.platform()}  engine={engine or '-'}")
    for dist in PACKAGES:
        print(f"  {dist:<24} {version(dist)}")
    for tool in ("ncu", "nsys"):
        print(f"  {tool:<24} {shutil.which(tool) or '-'}")

    has_gpu = gpu_info()
    checks = COMMON_CHECKS + ENGINE_CHECKS.get(engine or "", [])
    failed = 0
    for name, module, attrs in checks:
        ok, msg = check_import(module, attrs)
        failed += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name:<26} {module}  {msg}")

    if args.ncu:
        failed += not (has_gpu and check_ncu())

    # 没有 GPU 时只做报告，不判失败
    return 1 if (failed and has_gpu) else 0


if __name__ == "__main__":
    sys.exit(main())
