# KDA decode 研读指南

本目录下每个子目录是一份从上游抽出的实现。抽取由 `scripts/extract.py` 按 `k3x/manifest.py` 完成，来源、commit、sha256 和所有改动都记在各自的 `SOURCE.json` 里。上游版本：vLLM v0.32.0、SGLang v0.5.22、FLA v0.6.0、flashinfer v0.7.1。

> 副本是只读的，`python scripts/extract.py --check` 会发现手改。要做实验，就把文件复制到 `kernels/` 下再改；读书笔记写在这个 README 里。

---

## 1. "KDA decode"其实有两种算子边界

**核心递推**：输入是已经过 conv、并拆成 q/k/v 的张量，只做 delta rule 递推。K3 的设定是 K=V=128，`gate_lower_bound=-5`，在 kernel 内做 l2norm 和 β 的 sigmoid，状态是 fp32、V-first 布局 `[slots, HV, V, K]`。

```
q̂ = q / sqrt(Σq² + 1e-6) · K^-1/2          k̂ = k / sqrt(Σk² + 1e-6)
gₖ = lower_bound · σ(exp(A_log[h]) · (g + dt_bias[h,:]))   # 每个 K 通道一个对数衰减，取值在 (-5, 0)
β  = σ(raw_β)
S  ← S · diag(exp(gₖ))                     # S: [V, K]
v' = β · (v − S k̂)
S  ← S + v' k̂ᵀ
o  = S q̂
```

**全融合 decode step**：在核心递推的前后各融进一段，输入直接是投影 GEMM 的原始输出。
- 前面加 causal conv1d 更新：4 个 tap，SiLU，`conv_state` 里缓存 3 个 token。
- 后面加 gated RMSNorm：`o ← RMSNorm(o) · σ(g_out)`，对应 HF 代码里的 `FusedRMSNormGated(sigmoid)`。

生产环境里，两个引擎在 NVIDIA 上的首选都是全融合 kernel，因为三个 kernel 合成一个能省掉两次中间结果读写 HBM。

> FLA 的 `naive_recurrent_kda` 只覆盖递推的数学部分：它接收的 `g` 已经是对数空间、`β` 已经过 sigmoid，也不做 l2norm，状态布局是 K-first `[B,HV,K,V]`。所以要拿它当完整参考实现，得自己补上前处理和转置。这一步放在 P2 的 `k3ops/ops/kda.py` 里做。

**访存量**：每条序列每层的状态是 96×128×128×4 B = 6.29 MB（TP8 下每卡 12 头，是 0.79 MB），每一步都要读一遍再写一遍。算术强度大约 1 FLOP/B，是纯访存受限的算子。

---

## 2. 生产路径：两个引擎怎么选 kernel

### vLLM v0.32.0

入口在 `third_party/vllm/vllm/models/kimi_k3/nvidia/kda.py`。`resolve_kda_decode_backend`（L286）在 `auto` 模式下按顺序尝试：

1. **原生 CUDA** `ops.fused_kda_decode`（调用点 L1012），对应 → `vllm_cuda/`
   - 条件见 `is_fused_kda_decode_supported`（L161）：每卡头数 ∈ {12, 24, 48, 96}，D=128，conv=4，不做投机解码，bf16 输入，fp32 状态，SM90、SM10x 或 SM12x。
2. **flashinfer** `fused_kda_decode`（L995），对应 → `flashinfer_cutedsl/`
   - 条件见 L191：SM100/103。
   - 默认后端是 CuTe DSL。
3. **Triton**（L1136），是 vendored 的 FLA `fused_recurrent_kda`，对应 → `fla_triton/`

投机解码（1~7 个 token）另走 flashinfer 的 `packed_fused_kda_decode`（L964）。

### SGLang v0.5.22

入口在 `third_party/sglang/python/sglang/srt/layers/attention/linear/kda_backend.py` 的 `forward_decode`，按顺序：

1. **AMD gfx950** 的 HIP 融合路径（L779 起），不在本次范围内。
2. **NVIDIA 全融合** `kda_fused_decode.kda_fused_decode`（L869–918），对应 → `sglang_cuda/jit/csrc/attention/kda_fused_decode.cuh`
   - 启用条件：模型把输出门交给了 attention backend（见 `kimi_k3.py` 里的 attempt-and-verify 机制），并且 `covered()` 接受当前 shape。
   - 只为每卡 H ∈ {3, 6, 12} 编译了实例，对应 TP32、TP16、TP8。
3. **未融合链**：先用 Triton 做 `causal_conv1d_update`（L935），再交给 `KDAKernelDispatcher`（L99），按 `--linear-attn-decode-backend` 选递推 kernel：
   - **packed decode**（L945）需要 backend 支持带 lower bound 的 packed 路径。**K3 有 lower bound，只有 Helion 满足**，对应 → `sglang_helion/`
   - 其余 backend 走非 packed 的 `decode`（L981）：
     - `triton`：`fused_sigmoid_gating_delta_rule_update`，对应 → `sglang_triton/.../fla/fused_sigmoid_gating_recurrent.py`
     - `cutedsl`：`cutedsl_fused_sigmoid_gating_kda_update`，对应 → `sglang_cutedsl/`
     - `flashinfer`：`recurrent_kda`，SM100，对应 → `flashinfer_cutedsl/`

还有一个隐藏分支：`fused_recurrent.py` 里的 Triton packed decode 在 `covered()` 时会转去调 CUDA 的 `kda_packed_decode`（文件内搜 `kda_decode_cuda`）。上游注释写的理由是：row-streaming 的 CUDA 版状态读写能到约 9.6 TB/s，而 Triton 每个 warp 在寄存器里持有 `[BV, K]` 的状态块，最多约 5 TB/s。这两个数字是上游注释里的说法，我们还没实测。

---

## 3. 抽出来的实现

| 目录 | 语言 | 边界 | 主文件 | 行数 |
|---|---|---|---|---|
| `torch_naive/` | PyTorch | 递推数学（不含门控激活和 l2norm） | `fla/ops/kda/naive.py` → `naive_recurrent_kda` | 166 |
| `fla_triton/` | Triton | 核心递推 | `fla/ops/kda/fused_recurrent.py` → `fused_recurrent_kda_fwd_kernel` | 490 |
| `sglang_triton/` | Triton | 核心递推 | `.../fla/fused_sigmoid_gating_recurrent.py` → `fused_sigmoid_gating_delta_rule_update_kernel`（K3 实际走这个）；`.../fla/fused_recurrent.py` → `fused_recurrent_kda_packed_decode_kernel` | 750 + 1277 |
| `sglang_cutedsl/` | CuTe DSL | 核心递推 | `.../cutedsl_kda.py` → `cutedsl_fused_sigmoid_gating_kda_update` | 1562 |
| `sglang_helion/` | Helion | 核心递推（packed 输入） | `.../helion/kda_decode.py` → `helion_fused_recurrent_kda_packed_decode` | 393 |
| `sglang_cuda/` | CUDA | `kda_packed_decode.cuh` 只做递推；`kda_fused_decode.cuh` 是全融合 | `jit/csrc/attention/*.cuh`（加上 `jit/include/sgl_kernel/` 头文件，`ops/attention/*.py` 是上游的 JIT 封装，只供阅读） | 228 + 1064 |
| `vllm_cuda/` | CUDA | 全融合 | `kimi_k3/fused_kda_decode_kernel.cu`（`ops.h` 只为看函数声明） | 1167 |
| `flashinfer_cutedsl/` | CuTe DSL | `fused_kda_decode` 是全融合；`recurrent_kda` 是核心递推 | 公共 API 在 `flashinfer/kda_decode.py`；kernel 在 `flashinfer/kda_kernels/fused_kda_decode.py`、`recurrent_kda.py`、`packed_kda_decode_cute.py`。闭包里还带进了投机解码（MTP）用的 `fused_kda_decode_multitoken.py`、`kda_decode_wy_output_only.py`，以及 `sm120_prefill/` | 1026 + 2583 + 2613 |

**为了能独立运行而用的替身**：完整清单和原因见各目录的 `SOURCE.json` → `stubs`。
- FLA：替换了运行时后端分发器 `fla/backends.py`，以及包的 `fla/__init__.py`。
- flashinfer：替换了 JIT 磁盘缓存（改为进程内缓存）、API 日志、trace 模板、几个工具函数，以及 Cake / FlashKDA 的 CUDA 加载器（改成直接报错）。
- SGLang：替换了几个平台探测函数和 JIT 工具函数；并把 Triton 文件里转去 CUDA 的入口替换成恒返回 `False`，CUDA 版本在 `sglang_cuda/` 里单独看。

---

## 4. 建议的阅读顺序和要回答的问题

**1. `torch_naive`：先把数学弄清楚。**
- 状态 S 的形状是什么？
- 每一步读写多少字节？
- gate、β、l2norm 分别在哪一步起作用？

**2. `fla_triton` 的 `fused_recurrent_kda_fwd_kernel`：最直接的 GPU 写法。**
- 并行划分：grid = (V/32) · N · HV，每个 program 用 4 个 warp，在寄存器里持有一块 `[32, 128]` 的 fp32 状态（16 KB）。
- 为什么沿 V 切而不沿 K 切？提示：`S k̂` 和 `S q̂` 都是沿 K 归约。
- `STATE_V_FIRST` 对访存合并有什么影响？

**3. `sglang_triton` 的 `fused_sigmoid_gating_delta_rule_update_kernel`：K3 在 SGLang Triton backend 上实际跑的 kernel。**
- 和 FLA 版逐段对比：launch 配置怎么按 batch 选（`_select_recurrent_launch_config`）？
- 怎么用 PDL？
- `_cutedsl_exp` 这类近似指数函数是干什么用的？

**4. `sglang_cuda/.../kda_packed_decode.cuh`：row-streaming 的 CUDA 版递推。**
- 对照第 2 步，体会怎样把状态读写带宽推到接近 HBM 峰值。
- 拿 §6 的 ncu 指标验证上游注释里 5 TB/s 和 9.6 TB/s 的说法。

**5. 全融合：`vllm_cuda/kimi_k3/fused_kda_decode_kernel.cu` 和 `sglang_cuda/.../kda_fused_decode.cuh`。**

两者都源自 NVIDIA 的 `kda_decode_fusion_many_heads_kernel`，最好 diff 着读。

| | vLLM | SGLang |
|---|---|---|
| launch | grid = (B, H)，256 线程，`__launch_bounds__(256, 2)` | 同样的 kernel 骨架 |
| 状态搬运 | V 切成 4 段、每段 32；用 cp.async 把状态块搬进 shared memory | 1D TMA bulk 载入，3~4 级流水 |
| 头数实例 | 12 / 24 / 48 / 96 | 3 / 6 / 12 |
| 其他改动 | — | 显式 row stride（省掉 `.contiguous()`）、in-place 的状态池 slot stride（K3 上每个 slot 跨所有层，pitch 是 56,171,520 B）、PDL、padding slot 处理。全部列在 `kda_fused_decode.cuh` 开头的注释里 |

> SGLang 的 `ops/attention/kda_fused_decode.py` docstring 里写着 "no lower bound"，但 `.cuh` 已经支持 `kUseLowerBound`（L385、L568，注释里明确写了 K3 为 -5.0）。以 `.cuh` 为准。

**6. `flashinfer_cutedsl/.../fused_kda_decode.py`：同样是全融合，换成 CuTe DSL 来写。**
- 它把 4 tap 的 conv、SiLU、逐 K 通道门控的 delta rule 和 gated RMSNorm 放进一次 launch，支持 8、12、24、32、48、96 头。
- 和第 5 步的两份 CUDA 对照：用 DSL 写同样的融合，哪些东西更容易表达，哪些更难？
- `recurrent_kda.py` 是 SGLang flashinfer backend 用的核心递推，还包含投机解码的多 token 路径。
- **后端选择**：`kda_decode.py` 里 `fused_kda_decode` 和 `recurrent_kda` 默认都走 `backend="cute-dsl"`；`packed_kda_decode`（H=12）只有 Cake 一个后端。

**7. `sglang_cutedsl`、`sglang_helion`：同一个算法用别的 DSL 怎么写。**
- CuTe DSL 版在 shared memory 里按逻辑上的 `(K, V)` 计算，全局内存保持 VK 布局，载入时做转置。
- Helion 版实现的是 packed decode 的接口约定。

---

## 5. 还没做的部分

- **CUDA 副本的独立编译胶水。**
  - vLLM 用 torch stable ABI（`STABLE_TORCH_LIBRARY`），编译条件是 CUDA ≥ 13、arch `9.0a;10.0f;12.0f`、`--use_fast_math`。op schema 见 `third_party/vllm/csrc/torch_bindings.cpp`。
  - SGLang 用 tvm-ffi JIT（`load_jit` 加上 `KdaFusedDecodeKernel<...>::run` 包装）。
  - 这两样都要在 GPU 机器上写好并验证。
- **flashinfer 的 Cake / FlashKDA CUDA 后端**：它们是 `third_party/flashinfer/csrc/kda/` 下 400 多个由 agent 生成的变体（约 76 MB），没有抽取，副本里走到这些分支会报错。想看的话，直接去子模块里读那个目录。
- flashinfer 自带 KDA 的 FIB trace 模板 `third_party/flashinfer/flashinfer/trace/templates/kda.py`，写 Definition 时可以参考。副本里它被换成了占位。
- **vLLM vendored 的 FLA Triton**（`vllm/models/kimi_k3/nvidia/ops/third_party/kda/`）。它和 `fla_triton` 基本相同，没有单独抽取，需要时 diff 两者即可。

---

## 6. 怎么用

Python 副本（torch_naive、fla_triton、sglang_*、flashinfer_cutedsl）在 `pip install -e .` 之后可以直接 import，不会和已安装的 fla、sglang 冲突：

```python
from k3x.kda_decode.fla_triton.fla.ops.kda.fused_recurrent import fused_recurrent_kda
```

Triton、CuTe DSL、Helion 需要 GPU 环境。`torch_naive` 在 CPU 上就能跑，但要 Python ≥ 3.10，因为 FLA 用了 `float | None` 这类注解。

维护：

```bash
python scripts/extract.py kda_decode
```

```bash
python scripts/extract.py --check
```

用 ncu 看访存是否吃满带宽、occupancy 怎么样，命令和指标见 `docs/kimi-k3-operator-bench-plan.md` §6。
