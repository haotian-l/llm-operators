"""抽取清单：每个 (算子, 实现) 从哪个子模块、哪些入口文件抽取。由 scripts/extract.py 读取。

字段：
  source        SUBMODULES 里的键
  lang          pytorch / triton / gluon / cuda / cute-dsl / helion
  mode          python：沿 root 包内 import 补齐闭包，改写 import 前缀
                cpp：沿 #include 在 include_dirs 里补齐头文件闭包，逐字复制
  src_root      python 模式：root 包所在目录（相对子模块）
  root          python 模式：顶层包名
  entries       入口文件（相对子模块）
  stubs         python 模式：{相对 src_root 的模块路径: k3x/_stubs/ 下的替身文件}
                替身的 docstring 第一行写明为什么要替换
  base          cpp 模式：输出路径相对的上游目录
  include_dirs  cpp 模式：解析 <...> 头文件的目录
  copy          cpp 模式：额外逐字复制的文件（比如 Python 侧的 JIT 封装，只供阅读）
  notes         写进 SOURCE.json 的说明
"""

SUBMODULES = {
    "fla": "third_party/flash-linear-attention",
    "vllm": "third_party/vllm",
    "sglang": "third_party/sglang",
    "flashinfer": "third_party/flashinfer",
}

_FLA_STUBS = {
    "fla/backends.py": "fla_backends.py",
    "fla/__init__.py": "fla_init.py",
}

_SGLANG_SRT_UTILS_STUBS = {
    "sglang/srt/utils/__init__.py": "sglang_srt_utils.py",
    "sglang/srt/utils/common.py": "sglang_srt_utils_common.py",
}

OPS = {
    "kda_decode": {
        # ---- 参考实现
        "torch_naive": dict(
            source="fla", lang="pytorch", mode="python", src_root=".", root="fla",
            entries=["fla/ops/kda/naive.py"],
            description="FLA 的纯 PyTorch 逐 token 递推（naive_recurrent_kda），可在 CPU 上跑，作为数值参考",
        ),
        # ---- 只做递推的 kernel（输入已经过 conv、split 成 q/k/v）
        "fla_triton": dict(
            source="fla", lang="triton", mode="python", src_root=".", root="fla",
            entries=["fla/ops/kda/fused_recurrent.py"],
            stubs=_FLA_STUBS,
            description="FLA fused_recurrent_kda（Triton）：HF modeling 代码在 decode 时调用的就是它；"
                        "vLLM 的 triton 回退路径是它的 vendored 副本",
            notes=[
                "fla/utils/decorators.py 里的 sys.modules.get('fla.utils') 只用来读 tensor_cache 的开关，"
                "decode 路径不经过，保持原样",
            ],
        ),
        "sglang_triton": dict(
            source="sglang", lang="triton", mode="python", src_root="python", root="sglang",
            entries=[
                "python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py",
                "python/sglang/kernels/ops/attention/fla/fused_recurrent.py",
            ],
            stubs={
                **_SGLANG_SRT_UTILS_STUBS,
                "sglang/kernels/jit/utils/__init__.py": "sglang_jit_utils.py",
                "sglang/kernels/ops/attention/kda_packed_decode.py": "sglang_kda_packed_decode_off.py",
            },
            description="SGLang 的 Triton KDA decode kernel。K3 有 gate lower bound，Triton backend 不支持带 lower bound 的 "
                        "packed decode，所以 K3 实际走 fused_sigmoid_gating_recurrent.py 的 "
                        "fused_sigmoid_gating_delta_rule_update；fused_recurrent.py 里的 packed 变体给无 lower bound 的模型用",
        ),
        "sglang_cutedsl": dict(
            source="sglang", lang="cute-dsl", mode="python", src_root="python", root="sglang",
            entries=["python/sglang/kernels/ops/attention/cutedsl_kda.py"],
            description="SGLang 的 CuTe DSL KDA kernel，--linear-attn-decode-backend cutedsl 时使用",
        ),
        "sglang_helion": dict(
            source="sglang", lang="helion", mode="python", src_root="python", root="sglang",
            entries=["python/sglang/kernels/ops/attention/helion/kda_decode.py"],
            description="SGLang 的 Helion KDA decode kernel，--linear-attn-decode-backend helion 时使用",
        ),
        "flashinfer_cutedsl": dict(
            source="flashinfer", lang="cute-dsl", mode="python", src_root=".", root="flashinfer",
            entries=["flashinfer/kda_decode.py"],
            stubs={
                "flashinfer/utils.py": "flashinfer_utils.py",
                "flashinfer/api_logging.py": "flashinfer_api_logging.py",
                "flashinfer/trace/templates/kda.py": "flashinfer_trace_kda.py",
                "flashinfer/jit/cpp_ext.py": "flashinfer_cpp_ext.py",
                "flashinfer/jit/cute_dsl_core.py": "flashinfer_cute_dsl_core.py",
                "flashinfer/jit/cake_fused_kda_decode.py": "flashinfer_cake_off.py",
                "flashinfer/jit/cake_kda_decode.py": "flashinfer_cake_off.py",
                "flashinfer/jit/cake_kda_packed_t1.py": "flashinfer_cake_off.py",
                "flashinfer/jit/cake_flash_kda_packed_t1.py": "flashinfer_cake_off.py",
                "flashinfer/jit/flash_kda_decode.py": "flashinfer_cake_off.py",
            },
            description="flashinfer 的 KDA decode（CuTe DSL）：fused_kda_decode（vLLM 在 SM100/103 的第二选择，"
                        "conv+递推+norm 全融合）、recurrent_kda（SGLang flashinfer backend）、packed_kda_decode；"
                        "入口 kda_decode.py 是公共 API 和 cute-dsl / cake 后端选择",
            notes=[
                "Cake / FlashKDA 后端是 third_party/flashinfer/csrc/kda 下 400 多个生成的 CUDA 变体（约 76 MB），没有抽取；"
                "副本里这些分支会报错，fused_kda_decode 的默认 backend 本来就是 cute-dsl",
                "CuTe DSL kernel 原本经 flashinfer.jit 做磁盘缓存，副本改为进程内缓存",
            ],
        ),
        # ---- 融合 conv1d 更新 + 递推 + gated RMSNorm 的 kernel（输入是投影 GEMM 的原始输出）
        "vllm_cuda": dict(
            source="vllm", lang="cuda", mode="cpp", base="csrc",
            entries=["csrc/kimi_k3/fused_kda_decode_kernel.cu"],
            include_dirs=["csrc"],
            copy=["csrc/ops.h"],
            description="vLLM 原生 fused_kda_decode（CUDA）：conv1d 更新 + 递推 + 门控 + 输出 norm 一个 kernel，"
                        "K3 在 SM90/SM10x/SM12x 上 decode 的默认路径",
            notes=[
                "op schema 在 third_party/vllm/csrc/torch_bindings.cpp（搜 fused_kda_decode），"
                "Python 封装在 third_party/vllm/vllm/_custom_ops.py 的 fused_kda_decode",
                "上游编译：CUDA>=13，arch 9.0a;10.0f;12.0f，--use_fast_math，torch stable ABI（STABLE_TORCH_LIBRARY）",
                "ops.h 只为了看函数声明；独立编译的胶水代码尚未编写",
            ],
        ),
        "sglang_cuda": dict(
            source="sglang", lang="cuda", mode="cpp", base="python/sglang/kernels",
            entries=[
                "python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh",
                "python/sglang/kernels/jit/csrc/attention/kda_packed_decode.cuh",
            ],
            include_dirs=["python/sglang/kernels/jit/include"],
            copy=[
                "python/sglang/kernels/ops/attention/kda_fused_decode.py",
                "python/sglang/kernels/ops/attention/kda_packed_decode.py",
            ],
            description="SGLang 的两个 CUDA KDA decode kernel：kda_fused_decode（conv+递推+gated RMSNorm 全融合，"
                        "来自 NVIDIA x Moonshot K3 优化包，H∈{3,6,12}）和 kda_packed_decode（只做递推）",
            notes=[
                "两个 .py 是上游的 tvm-ffi JIT 封装，只作阅读参考；它们依赖 sglang.kernels.jit 的加载器，没有抽取",
                "独立编译的胶水代码（tvm_ffi 直接加载 .cuh）尚未编写",
            ],
        ),
    },
    "attn_res": {
        # ---- 参考实现
        "torch_naive": dict(
            source="fla", lang="pytorch", mode="python", src_root=".", root="fla",
            entries=["fla/ops/attnres/naive.py"],
            description="FLA 的 AttnRes 参考实现（naive_attnres）：每个残差源 RMSNorm 后和 pseudo-query 打分，"
                        "沿深度做 softmax 再加权求和，可选再接一个输出 RMSNorm",
        ),
        "fla_triton": dict(
            source="fla", lang="triton", mode="python", src_root=".", root="fla",
            entries=["fla/ops/attnres/fused.py"],
            stubs=_FLA_STUBS,
            description="FLA fused_attnres（Triton）：前向和反向 kernel，残差源以指针表传入（不需要先 stack）",
        ),
        "fla_gluon": dict(
            source="fla", lang="gluon", mode="python", src_root=".", root="fla",
            entries=["fla/ops/attnres/backends/gluon.py"],
            stubs=_FLA_STUBS,
            description="FLA 的 AttnRes Gluon 后端（Triton 更底层的方言，显式控制 warp 划分和 cp.async）",
            notes=["上游通过 fla.backends.register 把它注册成 fused_attnres 的替换；副本里 register 是空操作，直接调用其函数"],
        ),
        "vllm_triton": dict(
            source="vllm", lang="triton", mode="python", src_root=".", root="vllm",
            entries=["vllm/models/kimi_k3/nvidia/ops/attn_res.py"],
            stubs={
                "vllm/_custom_ops.py": "vllm_custom_ops_attn_res.py",
                "vllm/platforms/__init__.py": "vllm_platforms.py",
                "vllm/triton_utils/__init__.py": "vllm_triton_utils.py",
            },
            description="vLLM 的 attn_res()：残差加法 + 写 block 快照 + AttnRes 混合 + 输出 RMSNorm 融合成一个 Triton kernel；"
                        "SM10x 且 hidden=7168 时改走 CUDA 版（见 vllm_cuda）",
        ),
        "sglang_triton": dict(
            source="sglang", lang="triton", mode="python", src_root="python", root="sglang",
            entries=["python/sglang/srt/layers/attn_residual.py"],
            stubs={
                **_SGLANG_SRT_UTILS_STUBS,
                "sglang/srt/layers/layernorm.py": "sglang_layernorm.py",
                "sglang/srt/layers/linear.py": "sglang_linear.py",
                "sglang/srt/layers/communication/k3_sp_collective.py": "sglang_k3_sp_collective_off.py",
                "sglang/kernels/ops/attention/attn_res.py": "sglang_attn_res_tma_off.py",
                "sglang/kernels/ops/attention/attn_res_hip.py": "sglang_attn_res_hip_off.py",
            },
            description="SGLang 的 AttnRes 层：Triton 两段式（_score_kernel 打分 + _combine_kernel 混合）和分发逻辑；"
                        "SM10x/SM11x 且 H=7168 时改走 TMA CUDA 快速路径（见 sglang_cuda）",
        ),
        "vllm_cuda": dict(
            source="vllm", lang="cuda", mode="cpp", base="csrc",
            entries=["csrc/kimi_k3/attn_res_kernel.cu"],
            include_dirs=["csrc"],
            copy=["csrc/ops.h"],
            description="vLLM 的 kimi_k3_attn_res（CUDA）：SM10x、hidden=7168、num_blocks<=8 时替代 Triton 版",
            notes=[
                "op schema 在 third_party/vllm/csrc/torch_bindings.cpp（搜 kimi_k3_attn_res）",
                "ops.h 只为了看函数声明；独立编译的胶水代码尚未编写",
            ],
        ),
        "sglang_cuda": dict(
            source="sglang", lang="cuda", mode="cpp", base="python/sglang/kernels",
            entries=["python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh"],
            include_dirs=["python/sglang/kernels/jit/include"],
            copy=["python/sglang/kernels/ops/attention/attn_res.py"],
            description="SGLang 的 AttnRes TMA 快速路径（CUDA，SM10x/SM11x，H=7168），"
                        "以及和 reduce-scatter / all-gather 融合的多卡变体",
            notes=["attn_res.py 是上游的 tvm-ffi JIT 封装，只作阅读参考"],
        ),
    },
}
