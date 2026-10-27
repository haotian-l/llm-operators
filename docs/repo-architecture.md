# 代码架构与三方依赖

> 版本核对于 2026-10-26。全部锁定到**已发布的 release**：vLLM **v0.32.0**、SGLang **v0.5.22**、FLA **v0.6.0**。flashinfer-bench 用 main 分支的 `15618a6`，Kimi-K3 HF 仓库用 revision `f831ab6`。具体锁定见 [`requirements/`](../requirements) 和 [`third_party/README.md`](../third_party/README.md)。
> 方法论（roofline、ncu、路线图）见 [kimi-k3-operator-bench-plan.md](kimi-k3-operator-bench-plan.md)。本文只讲代码怎么组织、三方库怎么获取。

## 0. 几个关键决定

- **vLLM 和 SGLang 两个引擎同等对待，一起跑。** 两边的 torch 版本冲突（2.13.0 vs 2.14.1），所以各用一个环境。它们共用同一份数据集，包括 Definition、Workload 和参考实现；各自跑自己能 import 的那部分实现，最后把 trace 合并成一张表。
- **不需要 clone FlashInfer-Bench**，用 pip 从 git 安装并锁定 commit。PyPI 上的 `flashinfer-bench 0.1.2` 是 2026-02 的旧版，缺很多后来加的功能。
- **flashinfer-trace 数据集也不整库 clone**，只用脚本拉下十来个模板 JSON，见 §3.5。
- **要研读的 kernel 从引擎里抽出来，放进 `k3x/`。** 按 `k3x/manifest.py` 用 `scripts/extract.py` 从 pin 住的子模块抽取：
  - Python 实现会自动补齐依赖闭包，并把 import 前缀改写成 `k3x.<op>.<impl>.`，所以不会和已安装的包冲突；
  - CUDA 实现会补齐头文件闭包；
  - 除了 import 行之外文件逐字不变，来源、sha256 和替身都记在 `SOURCE.json` 里。

  详见 §6。

---

## 1. 分层架构

```
k3ops/model/                 models/kimi_k3/hf/config.json → ModelSpec；并行场景 → per-rank 维度
        │
        ▼
k3ops/ops/<op>.py            ← 唯一的事实来源（每个算子一个模块）
  ├─ axes(spec, scenario)        shape（const / var 轴）
  ├─ reference(**inputs)         PyTorch 参考实现（= FIB Definition.reference）
  ├─ make_inputs(axes, dist)     random / zipf / captured 三种输入
  └─ cost(axes)                  FLOPs / Bytes（给 roofline 用）
        │
        ▼
k3ops/impls/<lib>/<op>.py    适配器：把三方 kernel 包成统一的 run(**inputs) 签名
  共用：torch_ref │ fla │ flashinfer │ ours
  vLLM 环境专属：vllm │ deepgemm（vLLM 内置版）
  SGLang 环境专属：sglang（经 BaseFusedOp 逐个 backend 调用）
        │                                     │
        ▼                                     ▼
k3ops/fib/  生成 FIB 数据集               k3ops/perf/
  dataset/{definitions,solutions,           ├─ hw.py / roofline.py
           workloads}                       ├─ ncu.py / sweep.py（Nsight Python）
        │                                   └─ nsys.py
        ▼
┌──────────── envs/vllm ─────────────┐  ┌──────────── envs/sglang ───────────┐
│ flashinfer-bench run               │  │ flashinfer-bench run               │
│   --solutions 共用 + vllm_*        │  │   --solutions 共用 + sglang_*      │
└──────────────┬─────────────────────┘  └──────────────┬─────────────────────┘
               └──── flashinfer-bench report merge ────┘
                              ▼
                 k3ops/report/ → results/<gpu>/<date>/
```

**五条约定：**

1. **事实来源是 Python，FIB 的 JSON 是生成出来的。**
   - 同一个算子要按并行场景派生出很多 shape，比如 KDA decode 在 TP8 下每卡 12 头、在 DP-attn 下 96 头。手写 JSON 维护不过来。
   - 生成器用 `inspect.getsource` 把 `reference` 内联进 Definition，把 `impls/` 下的适配器打包成 Solution。
2. **Solution 名字带上来源前缀**：`torch_ref_*`、`fla_*`、`flashinfer_*`、`vllm_*`、`sglang_<backend>_*`、`ours_*`。每个环境用 `--solutions` 只跑自己能 import 的那部分。
   - 共用实现（torch_ref、fla、flashinfer）在**两个环境里都跑**。两边的 flashinfer 版本不同（0.7.0.post1 vs 0.7.1），这样顺带能看出版本差异。
3. **`impls/` 只做适配，不拷贝三方 kernel 源码。** layout 转换和权重预处理放在 setup 阶段，不计入时间。
   - 三方 kernel 可以从已安装的包 import，也可以从 `k3x/` 的研读副本 import。
   - 两者同时注册成 Solution，还能顺便验证抽出来的副本和原版行为一致。
4. **perf 层直接调用 `impls/`，不经过 FIB。** 签名是同一个，可以用 NVTX 精确包住一次调用，给 ncu 和 nsys 过滤用。
5. **自己写的 kernel 放 `kernels/`，在 `impls/ours/` 里注册**，和三方实现在同一张表上比。

---

## 2. 目录结构

```
Operators/
├── pyproject.toml                 # 本仓库的包 k3ops（只依赖轻量包；GPU 依赖由 setup_env.sh 管）
├── requirements/
│   ├── base.txt                   # Mac 开发用：生成 shape/cost 表、跑小 shape 的 reference
│   ├── vllm.txt                   # 引擎锁定：vllm==0.32.0
│   ├── sglang.txt                 # 引擎锁定：sglang==0.5.22
│   └── tools.txt                  # 两个环境共用：flashinfer-bench(git)、nsight-python；FLA 单独 --no-deps 装
├── envs/                          # .gitignore：envs/vllm、envs/sglang 两个 venv
├── third_party/                   # git submodule，只读，pin 到 release tag（§3.4）
│   ├── vllm/                      # v0.32.0
│   ├── sglang/                    # v0.5.22
│   ├── flash-linear-attention/    # v0.6.0
│   └── README.md
├── models/kimi_k3/                # 只放数据，不是 Python 包
│   ├── hf/                        # HF 快照（revision 锁定）：config.json + modeling/configuration .py + LICENSE
│   └── scenarios.yaml             # decode B∈{1..512}，prefill chunk∈{1K..16K}，ctx∈{4K..1M}
├── k3ops/
│   ├── model/                     # P1：kimi_k3.py（config → ModelSpec）、parallel.py（TP / DP-attn / EP → per-rank 维度）
│   ├── ops/                       # attn_res、norm、gemm、router、moe、act、conv、kda、mla、quant、sampling
│   ├── impls/{torch_ref,fla,flashinfer,vllm,sglang,deepgemm,ours}/
│   ├── fib/                       # gen_definitions / gen_solutions / gen_workloads
│   ├── perf/
│   └── report/
├── k3x/                           # 从上游抽出的 kernel 研读副本（只读，脚本管理，§6）
│   ├── manifest.py                # 抽取清单
│   └── <op>/                      # README.md 是手写的阅读指南；<impl>/ 由脚本生成，内含 SOURCE.json
├── dataset/                       # 生成出来的 FIB 数据集
│   └── definitions/ solutions/ workloads/ traces/
├── templates/flashinfer-trace/    # 从 flashinfer-trace 拉下来的模板 Definition（Apache-2.0）
├── kernels/                       # 自写 kernel（Triton / CuTe DSL / CUDA）
├── scripts/
│   ├── setup_env.sh               # setup_env.sh vllm|sglang|all
│   ├── check_env.py               # 打印版本、GPU 架构；检查 K3 相关 import；可选检查 ncu 权限
│   ├── extract.py                 # 按 k3x/manifest.py 抽取 kernel；--check 校验副本和上游
│   ├── fetch_hf.sh                # 下载 HF 快照（不下权重）
│   └── fetch_fib_templates.sh     # 下载 flashinfer-trace 里的模板 JSON
├── tests/
├── results/<gpu>/<date>/          # csv、png 进 git；.ncu-rep、.nsys-rep 被忽略
└── docs/
```

---

## 3. 三方依赖

### 3.1 两个环境的版本锁定

| | `envs/vllm` | `envs/sglang` |
|---|---|---|
| 引擎 | **vllm==0.32.0**（tag commit `5df63a7`） | **sglang==0.5.22**（tag commit `baa9bef`） |
| torch | 2.13.0（cu13） | 2.14.1（cu13） |
| flashinfer-python | 0.7.0.post1 | 0.7.1 |
| nvidia-cutlass-dsl | 4.8.0 | 4.8.0 |
| DeepGEMM | 内置在 vLLM wheel 里（vllm-project fork） | `sgl-deep-gemm==0.2.2` |
| 其他 kernel 包 | FlashKDA（内置）、FlashMLA、triton_kernels | `sglang-kernel==0.4.9`、`sgl-deep-ep`、`flash-attn-4` |
| 共用工具 | flash-linear-attention 0.6.0（`--no-deps`）、flashinfer-bench（git `15618a6`）、nsight-python 1.1.0 | 同左 |

- **两个 wheel 都是 cu13 版本，驱动要支持 CUDA 13。**
- **安装顺序**：先装引擎，然后 `pip freeze` 成 constraints 文件，再在这个约束下装共用工具。这样共用工具不会把引擎锁定的 torch、flashinfer 偷偷升级掉；真有冲突会直接报错。`scripts/setup_env.sh` 就是这么做的。
- **FLA 用 `--no-deps` 安装**。它唯一的硬依赖是 `transformers>=5.6.2`，而 kernel 用不到 transformers，没必要为它动引擎的 pin。
- **vLLM main 比 0.32.0 多了一整套 `vllm/models/kimi_k3/nvidia/specialized_decode/`**（CuTe DSL 的 decode 栈）。以后想测它，把 `requirements/vllm.txt` 换成 nightly，再把 `third_party/vllm` 挪到同一个 commit。
- **两个引擎的官方 docker 镜像也可以直接当底座**，在镜像里再跑 `setup_env.sh` 的"共用工具"那一步即可。容器里跑 ncu 要加 `--cap-add SYS_ADMIN`。

### 3.2 vLLM 侧提供的 K3 实现（`impls/vllm/`）

| 算子 | 实现 |
|---|---|
| AttnRes（+RMSNorm） | `vllm.models.kimi_k3.nvidia.ops.attn_res`（Triton）、CUDA `csrc/kimi_k3/attn_res_kernel.cu` |
| KDA decode | CUDA `csrc/kimi_k3/fused_kda_decode_kernel.cu`（融合 conv、recurrent、gate、norm） |
| KDA prefill | 内置 FlashKDA；vendored FLA（`ops/third_party/kda`） |
| KDA 投影 | `nvidia/low_latency_gemm.py`；`ops/cute_dsl/kda_skinny_gemm.py` |
| latent MoE | `nvidia/latent_moe_runner.py`；`ops/latent_moe_tail.py`（tail 融合 RMSNorm、up 投影和通信） |
| MLA | `nvidia/mla.py`（gate 融合、key concat 写 cache） |
| 现成微基准 | `third_party/vllm/benchmarks/kernels/benchmark_kimi_k3_{attn_res,kda_decode,kda_projection,latent_moe_tail,sp_collectives}.py` |

### 3.3 SGLang 侧提供的 K3 实现（`impls/sglang/`）

SGLang 的实现选择**最丰富**，而且接口对我们最友好：

- **`sglang.kernels` 有统一的多 backend 契约 `BaseFusedOp`。**
  - 用 `forward(..., backend=KernelBackend.X)` 可以显式指定某个 backend。
  - 设 `SGLANG_FORCE_FUSED_OP_BACKEND=native` 可以全局切回纯 torch 实现。
  - 因此 `impls/sglang/` 的适配器可以**直接枚举同一个算子的所有 backend**，不用为每个 backend 单独拼调用路径。
- **KDA**：linear-attention 后端有 `triton`、`cutedsl`、`flashinfer`、`flashkda`、`nvidia_kda`、`ptx_kda`、`helion`，入口在 `srt/layers/attention/linear/kda_backend.py`。kernel 在 `kernels/ops/attention/{kda_fused_decode,cutedsl_kda,kda_decode_mtp}.py`、`kernels/ops/attention/fla/` 和 `kernels/kda_kernels/`。*各 backend 的硬件要求还没逐个核对。*
- **AttnRes**：`kernels/ops/attention/attn_res.py` 和 `srt/layers/attn_residual`。
- **MLA**：`flashinfer_mla`、`trtllm_mla`、`cutedsl_mla`、`fa3`、`fa4`、`triton`。`flashmla` 后端在 K3 上不适用，原因见 §3.4。
- **MoE runner**：`deep_gemm`、`triton`、`triton_kernel`、`flashinfer_trtllm(_routed)`、`flashinfer_cutlass`、`flashinfer_mxfp4`、`flashinfer_cutedsl`、`flashinfer_megamoe`、`cutlass`、`marlin`。
- **现成微基准**：
  - `third_party/sglang/test/registered/kernels/benchmark/attention/bench_kimi_k3_kda_decode.py`
  - `.../benchmark/gemm/bench_kda_fp8_skinny_gemm.py`
  - `third_party/sglang/benchmark/bench_linear_attention/`，里面有 KDA decode/prefill 各实现的对比

### 3.4 `third_party/` 子模块（只读，不安装）

| 子模块 | pin | 为什么需要源码 | 批次 |
|---|---|---|---|
| **vllm** | v0.32.0 | K3 kernel 的 csrc（wheel 里没有 `.cu`）、`benchmarks/kernels/benchmark_kimi_k3_*.py`、模型实现 | **第一批** |
| **sglang** | v0.5.22 | `srt/models/kimi_k3.py`、`kernels/ops/*`、`BaseFusedOp`、K3 微基准和 `bench_linear_attention/` | **第一批** |
| **flash-linear-attention** | v0.6.0 | KDA 的 Triton 源码；`fla/ops/kda/naive.py` 里的 `naive_recurrent_kda` 和 `naive_chunk_kda` 是参考实现的起点 | **第一批** |
| **flashinfer** | v0.7.1 | KDA decode 的 CuTe DSL kernel（已抽取）；以后要抽的 MLA、MoE kernel 也在这里。约 475 MB | **已加入** |
| FlashKDA | vllm-project/FlashKDA @`17a037d`（vLLM 的 pin） | CUTLASS KDA 的设计文档和 H20 基准 | 按需 |
| DeepGEMM | vLLM fork 或 sgl-deep-gemm 对应的版本 | FP8×FP4 grouped GEMM、Mega MoE | 按需 |
| DeepEP、cutlass | — | 多卡通信、自己写 kernel | 后期 |

- **不需要 clone 的**：flashinfer-bench（pip 从 git 装）、flashinfer-trace（脚本拉模板）、tritonbench（只参考）、Kimi K3 权重。
- **FlashMLA 不需要。** 它主分支的 decode 只支持稀疏注意力 + FP8 KV cache（代码里有 `assert is_fp8_kvcache`），不适用于 K3 的稠密 BF16 MLA。
- **新机器上同步子模块**：`git submodule update --init`。`.gitmodules` 里已经写了 `shallow = true`。

### 3.5 flashinfer-trace 模板

`scripts/fetch_fib_templates.sh` 会把下面这些拉到 `templates/flashinfer-trace/`，生成器以它们为蓝本改成 K3 的 shape：

| 模板 | 改成 K3 的什么 |
|---|---|
| `gdn/gdn_{decode,prefill}_*` | KDA。gate 要从每头一个标量改成逐通道 `[...,H,128]`，另外加 l2norm、`lower_bound` |
| `mla_paged/mla_paged_decode_h16_ckv512_kpe64_ps64` | MLA decode h96 / h12（K3 是 NoPE） |
| `mla_ragged/mla_ragged_prefill_causal_h16_qk192_vo128` | MLA prefill |
| `moe/moe_fp8_block_scale_ds_routing_topk8_ng1_kg1_e384_h7168_i2048`（Kimi K2） | router：e896、top16 |
| `moe/trtllm_fp4_block_scale_moe_*`、`gemm/grouped_gemm_mxfp8_mxfp4_*` | latent MoE：h3584、i3072、SiTU |
| `rmsnorm/{rmsnorm,fused_add_rmsnorm}_h7168`、`rmsnorm_h1536/h512` | 直接复用，另外补一个 h3584 |
| `sampling/*_v129280` | vocab 改成 v163840 |

---

## 4. 环境与运行方式

| | Mac（开发机） | GPU 机器 |
|---|---|---|
| 环境 | `requirements/base.txt`（Python ≥3.10，建议用 uv 装 3.12） | `scripts/setup_env.sh all` → `envs/vllm` 和 `envs/sglang` |
| 能做的事 | P1 的 shape/cost 表；生成 Definition 和 Workload；用 GUI 打开 `.ncu-rep` / `.nsys-rep` | 所有计时、ncu、nsys |

**一次完整跑分**（后面由 `scripts/run_bench.sh` 封装）：

```bash
envs/vllm/bin/flashinfer-bench run --local dataset --solutions <共用 + vllm_* 的列表>
```

```bash
envs/sglang/bin/flashinfer-bench run --local dataset --solutions <共用 + sglang_* 的列表>
```

```bash
envs/vllm/bin/flashinfer-bench report merge --local <vllm 跑出的 trace> --local <sglang 跑出的 trace> --output results/<gpu>/<date>/traces
```

**每次跑分都要冻结环境**：把两个环境的 `pip freeze`、`nvidia-smi -q` 和 GPU 时钟写进 `results/<gpu>/<date>/env/`。

---

## 5. 落地顺序

1. **（Mac）搭骨架**：目录、三个子模块、HF 快照、模板。✅ 已完成。
2. **（Mac）写 P1**：`k3ops/model/{kimi_k3,parallel}.py`、各 `k3ops/ops/*` 的 `axes` 和 `cost`，产出"算子 × 场景"的 roofline 大表。
3. **（GPU）建环境**：`scripts/setup_env.sh all`，然后 `check_env.py --ncu`，再跑 `perf/hw.py` 测峰值。
4. **（GPU）用 KDA decode 竖切一刀，两个引擎一起打通全链路。** 可以对比的实现大约有 10 个：
   - 共用：torch reference、FLA `fused_recurrent_kda`、flashinfer `fused_kda_decode` 和 `packed_kda_decode`（h12）
   - vLLM：CUDA `fused_kda_decode`
   - SGLang：`triton`、`cutedsl`、`flashinfer`、`flashkda`、`nvidia_kda`、`ptx_kda` 各个 backend

   全链路是：生成数据集 → 两个环境各跑一遍 → merge → roofline → ncu。跑通之后再按 plan 的 P2 横向铺开到其他算子。

---

## 6. kernel 抽取（`k3x/`）

**目的**：研读。把散落在引擎深处的 kernel 和它的最小依赖闭包放到一处，按算子组织，可以对照、diff，也能在 GPU 环境里单独 import 运行。

**规则**：

| | Python 实现（Triton / CuTe DSL / Helion / PyTorch） | CUDA 实现 |
|---|---|---|
| 闭包 | 从入口文件出发，沿 `root` 包内的 import（包括函数内的惰性 import）补齐 | 沿 `#include` 在 `include_dirs` 里补齐；torch、tvm-ffi、CUDA 等外部头文件只记录不复制 |
| 改动 | 只改写 import 前缀 `root.` → `k3x.<op>.<impl>.root.`；路径上的空包生成 marker `__init__.py` | 逐字复制，保持上游相对路径，include 不用改 |
| 切断 | 会牵出整个引擎的模块（运行时分发器、平台探测、JIT 加载器）用替身代替，替身照抄上游语义，原因写进 `SOURCE.json` | 编译胶水另写（待做） |

**用法**：
- 新增一个算子或实现：在 `k3x/manifest.py` 里加一项，然后运行 `python scripts/extract.py <op>`。
- 检查：`--check` 会报告三类问题：
  - `[EDITED]`：副本被手改过；
  - `[STALE]`：子模块升级了，需要重新抽取；
  - `[IMPORT]`：import 在副本内解析不到。
- 抽取脚本只用标准库，Mac 上的系统 Python 3.9 就能跑。

**替身**：放在 `k3x/_stubs/`，多个实现共用；每个文件 docstring 的第一行写明为什么要替换，抽取时会记进 `SOURCE.json`。

**已抽取**：

| 算子 | 实现 | 阅读指南 |
|---|---|---|
| `kda_decode` | 8 个：torch、FLA Triton、SGLang Triton / CuTe DSL / Helion / CUDA、vLLM CUDA、flashinfer CuTe DSL | [k3x/kda_decode/README.md](../k3x/kda_decode/README.md) |
| `attn_res` | 7 个：torch、FLA Triton / Gluon、vLLM Triton / CUDA、SGLang Triton / CUDA | [k3x/attn_res/README.md](../k3x/attn_res/README.md) |
