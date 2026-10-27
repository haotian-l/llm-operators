# Kimi K3 算子梳理 · Benchmark · Profiling 方案

> 调研日期：2026-10-26。模型信息来自 HF `moonshotai/Kimi-K3` 的 `config.json`、`modeling_kimi_linear.py` 和 `model.safetensors.index.json`。工具信息来自 flashinfer-bench / tritonbench 源码和 Nsight Python 文档。

## 0. 结论速览

1. **不用从零写框架。** 用 **FlashInfer Trace schema**（Definition / Solution / Workload / Trace）描述每个算子的输入、输出和 PyTorch reference，用 **flashinfer-bench** 跑多实现的正确性校验和计时。它的设计和"每个算子有输入、输出、多种实现，输入可以先用 random"几乎一一对应，而且已经有 MLA(ckv512/kpe64)、MXFP4 GEMM、Kimi K2 MoE routing、GDN、RMSNorm h7168 等定义，可以直接复用或改 shape。
2. **profiling 这层自己补一个薄层。** flashinfer-bench 只有面向 agent 的 ncu 文本封装，没有 roofline，也不做 sweep。建议自己加三样：
   - 解析式 cost model：从 Definition 的 axes 算 FLOPs 和 Bytes，再得出 roofline 下界和 SOL%。
   - ncu：用 NVTX 过滤目标 kernel，结果导出成 CSV。
   - Nsight Python：做 batch size × 实现的 sweep。
3. **TritonBench（Meta）作参考和补充。** 它的 metrics 很全，包括 latency/tflops/gbps/hw_roofline/ncu_rep/nsys_rep/arithmetic_intensity，也已经有 `mxfp4_gemm`、`gdn_fwd_h` 等 op。但它安装很重（要 PyTorch nightly 和一堆 submodule），算子写在 Python 代码里，没有可移植的 schema，不建议当主框架。
4. **顺序：先建 shape/roofline 表，再挑热点深挖。** shape/roofline 表在 Mac 上就能跑，不需要 GPU。它会告诉你哪些算子占时间、离理论上限多远，然后只对排在前面的算子做 ncu 分析。

---

## 1. Kimi K3 结构要点

| 项 | 值 |
|---|---|
| 层数 | 93。第 4,8,…,92 层和第 93 层（1-indexed）是 **Gated MLA（24 层）**，其余是 **KDA（69 层）** |
| FFN | layer 0 是 dense MLP（intermediate 33792），其余 92 层是 **Latent MoE** |
| hidden / heads | 7168 / 96 |
| KDA | head_dim 128，96 头（投影宽 12288），short conv k=4 + SiLU，forget gate 低秩（7168→128→12288），**full-rank 输出门** + `FusedRMSNormGated(sigmoid)`，`gate_lower_bound=-5`，状态 `[B,96,128,128]` fp32 |
| MLA | q_lora 1536，kv_lora 512，qk_nope 128，rope 64，v 128。**NoPE**：代码里 `rotary_emb=None`，rope 那 64 维仍然在 latent cache 里，但不做旋转。带**输出门** `sigmoid(g_proj(x))` |
| MoE | 896 个 routed experts，top-16，2 个 shared experts。Router 是 fp32 GEMM，`sigmoid` + `e_score_correction_bias`（noaux_tc），n_group=1，renormalize |
| Latent MoE | 先 7168→3584 下投影，在 3584 维上跑 experts（gate_up 3584→2×3072，down 3072→3584），再 RMSNorm(3584)，最后 3584→7168 上投影 |
| 激活 | **SiTU-GLU**：`4·tanh(g/4)·σ(g) · 25·tanh(u/25)`（beta=4，linear_beta=25） |
| AttnRes | `attn_res_block_size=12`。每层在 attention 前、MLP 前各做一次"深度方向 softmax 加权"（block_residual 最多约 8 个 block） |
| 量化 | **只有 routed experts 的 w1/w2/w3 是 MXFP4**（group 32，e8m0 scale）。按 model card，激活是 MXFP8。attention、shared experts、latent 投影、dense MLP、lm_head 都是 **BF16**，router 是 fp32 |
| vocab / ctx | 163840 / 1M。没有 MTP 层（`num_nextn_predict_layers=0`） |

**校验：** 按下面的清单累加，每 token 激活参数约 **104.2B**，和官方的 104B 一致，说明清单没有漏项。

**用 config 粗算的几个关键量**（TP=1 全局视角，是估算，不是实测）：

| 量 | 数值 | 含义 |
|---|---|---|
| BF16 非 expert 权重 | ~56B 参数，约 112 GB。其中 attention 投影占 36B | attention 没有量化。小 batch decode 时，这部分 GEMV 每步都要读一遍权重，是一笔固定的访存成本 |
| MXFP4 routed experts | 2.72T 参数，约 1.45 TB | 单个 expert 有 33.0M 参数 |
| KDA 状态 | 每条序列每层 6.29 MB（fp32），69 层合计 **434 MB/序列** | decode 每步都要读写一次 |
| MLA latent cache | 1152 B/token/层，24 层合计 27.6 KB/token | |
| KDA 和 MLA 的交叉点 | 上下文约 **31K tokens** | 短于这个长度时，decode 的"attention 访存"主要花在 KDA 状态上，而不是 MLA cache 上 |
| MoE 的稀疏程度 | 全局 decode batch=64 时，每层会激活约 613/896 个 expert，平均每个 expert 只有 1.7 个 token | 极度访存受限。要让 W4A8 grouped GEMM 接近 B200 的 ridge（约 150 token/expert），全局 batch 需要约 8K token |

---

## 2. 算子清单

记号：T 是本批 token 数（prefill 是各序列长度之和，decode 时 T=B），B 是序列数。shape 按 TP=1 写。实际 per-rank shape 由并行策略决定，见 §7。

| # | 算子 | 输入 → 输出 | 每层次数 | 可对比的现成实现（作为 Solution） | FlashInfer-Bench 现有定义 | 预期瓶颈 |
|---|---|---|---|---|---|---|
| 1 | Embedding | ids[T] → [T,7168]，表 [163840,7168] bf16 | 1（全模型） | torch | — | mem |
| 2 | **AttnRes 混合** | prefix[T,7168]，blocks[T,nb≤8,7168]，w[7168] → [T,7168] | 2 | torch ref，vLLM（Triton、CUDA `attn_res_kernel.cu`、CuTe DSL，融合了 RMSNorm），SGLang `attn_residual` | **新增** | mem |
| 3 | RMSNorm / fused add RMSNorm | [T,7168]，另有 [T,1536]、[T,512]、[T,3584] | 2~4 | flashinfer，sgl-kernel，vLLM，Liger/Triton | `rmsnorm_h7168/h1536/h512` ✅，h3584 需新增 | mem |
| 4 | Dense GEMM（BF16） | KDA q/k/v/g (N=12288,K=7168)×4 和 o (7168,12288)，f_a (128,7168)，f_b (12288,128)，b (96,7168)；MLA q_a (1536,7168)，q_b (18432,1536)，kv_a (576,7168)，kv_b (24576,512)，g (12288,7168)，o (7168,12288)；latent down (3584,7168) 和 up (7168,3584)；shared gate_up (12288,7168) 和 down (7168,6144)；dense0 gate_up (67584,7168) 和 down (7168,33792)；lm_head (163840,7168) | 多次 | cuBLAS(Lt)，DeepGEMM，CUTLASS，Triton | `gemm_n{N}_k{K}` 模式，补 K3 shape 即可 | decode 时 mem，prefill 时 compute |
| 5 | Router GEMM | [T,7168] × [896,7168]ᵀ → [T,896] fp32 | 1 | torch fp32，各引擎的 router GEMM 专用 kernel | 类似 `gemm_n256_k7168` | mem |
| 6 | **Router top-k** | logits[T,896] + bias → idx[T,16] i32，w[T,16] f32 | 1 | torch ref，flashinfer/TRTLLM routing，sglang `biased_grouped_topk`，vLLM grouped_topk | MoE 定义里的 `ds_routing_..._ng1_kg1`（Kimi K2 有 e384） | latency |
| 7 | **Routed experts（Latent MoE）** | x_lat[T,3584] (MXFP8)，W13 [896,6144,3584] MXFP4，W2 [896,3584,3072] MXFP4，topk → y[T,3584] | 1 | flashinfer `trtllm_fp4_block_scale_moe`（TRTLLM-Gen，已支持 `ActivationType.Situ`），flashinfer `cutlass_fused_moe`，DeepGEMM FP8×FP4，vLLM latent MoE runner，Triton `matmul_ogs`，Marlin（Hopper 上 W4A16） | `trtllm_fp4_block_scale_moe_*`、`grouped_gemm_mxfp8_mxfp4_*`（其他 shape） | decode 时 mem，大 batch 时 compute |
| 7a | └ 拆开测的子算子 | MXFP8 量化 → permute → GEMM1+**SiTU** → GEMM2 → unpermute+加权 reduce；latent tail（RMSNorm + up 投影 + 通信） | | flashinfer TRTLLM-Gen 已带 SiTU，其他 kernel 多数只有 SwiGLU；latent tail 可以和 vLLM 的 `latent_moe_tail` 对比 | | |
| 8 | SiTU-GLU（独立） | [T,2I] → [T,I]，I=6144（shared）或 33792（dense） | 1 | torch，Triton，torch.compile | **新增** | mem |
| 9 | ShortConv1d（causal，k=4，SiLU） | x[T,12288]×3（可合并为 36864），conv_state[B,12288,3] | 3 | FLA `ShortConvolution`，causal-conv1d，vLLM 融合版 | **新增** | mem |
| 10 | **KDA chunk（prefill）** | q,k,v,g [T,96,128] bf16，beta[T,96]，A_log[96]，dt_bias[12288]，h0[B,96,128,128] f32，cu_seqlens → o[T,96,128]，hT | 1 | FLA `chunk_kda`（Triton），**FlashKDA**（CUTLASS，SM90+，FLA 会自动调用它），flashinfer `kda_prefill`，SGLang CuTe DSL | 参照 `gdn_prefill_*` 改（gate 从标量改成逐通道） | compute（受 chunk size 影响） |
| 11 | **KDA recurrent（decode）** | 同上，T=B，状态原地更新 | 1 | FLA `fused_recurrent_kda`，flashinfer `fused_kda_decode` / `packed_kda_decode`（H=12，SM100a/103a），vLLM CUDA `fused_kda_decode`（融合 conv+recurrent+gate+norm） | 参照 `gdn_decode_*` 改 | **mem（状态 6.29 MB/seq，AI≈1）** |
| 12 | FusedRMSNormGated | o[T,96,128]，g[T,96,128] → [T,96,128] | 1 | FLA，自写 Triton | **新增** | mem |
| 13 | MLA prefill（ragged） | q[T,96,192]，k[T,96,192]，v[T,96,128]，causal varlen → o[T,96,128] | 1 | FA3/FA4，flashinfer（cutlass/trtllm），cuDNN | `mla_ragged_prefill_causal_h16_qk192_vo128`（改头数） | compute |
| 14 | **MLA decode（paged，absorbed）** | q_nope'[B,96,512]，q_pe[B,96,64]，ckv[pages,ps,512]，kpe[pages,ps,64]，indptr/indices → o[B,96,512] | 1 | flashinfer（`trtllm_batch_decode_with_kv_cache_mla`、`cute_dsl_mla_decode`、xqa、`BatchMLAPagedAttentionWrapper`），`tokenspeed-mla`，vLLM CuTe DSL，Triton。注意：FlashMLA 主分支的 decode 现在只支持稀疏注意力 + FP8 KV，用不上 | `mla_paged_decode_h16_ckv512_kpe64_ps1/ps64`（改成 h96 或 h12） | 96 头时 AI≈181（接近 BF16 ridge），12 头时 AI≈23（mem） |
| 14a | └ absorb BMM | q_nope × W_UK[96,128,512]，o_lat × W_UV[96,512,128] | 2 | torch.bmm，DeepGEMM | **新增** | mem |
| 15 | MXFP8 激活量化 | [T,3584] bf16 → fp8 + e8m0 scale（每 32 个元素一组） | 1 | flashinfer `mxfp8_quantize`，DeepGEMM，Triton | **新增** | mem |
| 16 | EP dispatch/combine | [T,3584]（latent 维，载荷是 hidden 的一半）+ topk | 2 | DeepEP，NCCL all2all，flashinfer MNNVL a2a | — | 通信延迟 |
| 17 | LM head + sampling | [B,7168] → logits[B,163840] → top-k/top-p | 1 | flashinfer sampling，torch | `*_sampling_from_probs_v129280`（改成 v163840） | mem，与数据分布相关 |

---

## 3. 业界怎么做

成熟的项目基本收敛到三层：

1. **算子契约**：shape（区分常量轴和可变轴）+ dtype + 一份清晰的 PyTorch reference + 从真实模型或流量抓来的 workload。
2. **多实现计时和正确性**：同一组输入跑 N 个 backend，和 reference 比数值，用 CUPTI 或 CUDA graph 去掉 launch 噪声。
3. **硬件计数器分析**：nsys 看时间线，ncu 看 SOL、roofline、occupancy、stall，再和解析式 roofline 对照。

| 工具 | 定位 | 输入/输出契约 | 多实现 | 计时 | ncu / roofline | K3 覆盖 | 建议 |
|---|---|---|---|---|---|---|---|
| **[FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench)** | LLM 算子 benchmark 数据集 + 流水线（MLSys'26） | ✅ JSON Definition（const/var 轴，inputs/outputs，PyTorch `run()` reference） | ✅ Solution 支持 python/triton/cuda/tilelang | `bench_gpu_time_with_cupti`，**冷 L2**，取中位数；按 rtol/atol 判 PASSED | 只有 agent 用的 `flashinfer_bench_run_ncu(set, sections, page)` 文本输出；**没有 roofline** | MLA、GDN、MXFP4 GEMM、K2 MoE routing、RMSNorm h7168 | **主框架** |
| **[TritonBench](https://github.com/meta-pytorch/tritonbench)** | PyTorch/Triton 算子性能套件 | Python `Operator.get_input_iter()` | ✅ `@register_benchmark(baseline=…)` | do_bench，支持 `--cudagraph` | ✅ `--metrics latency,tflops,gbps,speedup,accuracy,hw_roofline,arithmetic_intensity,memory_traffic,ncu_rep,nsys_rep,kineto_trace` | `mxfp4_gemm`、`nvfp4_gemm`、`grouped_gemm`、`gdn_fwd_h`（FLA vs TileLang）、`rms_norm` | 参考它的 metric 设计，个别 op 可以直接跑 |
| **[Nsight Python](https://docs.nvidia.com/nsight-python/analyze.html)** | 在 Python 里驱动 ncu | — | 通过 `configs` 做 sweep | ncu `gpu__time_duration` | ✅ 任意 ncu metric，输出 DataFrame/CSV/图；支持 `clock_control`、`cache_control`、`thermal_mode` | — | **sweep 和 occupancy 分析用它** |
| 计时原语 | `triton.testing.do_bench(_cudagraph)`、`flashinfer.testing.bench_gpu_time(_with_cupti)`、CUDA Event | | | | | | 自己写脚本时用 |
| 引擎自带 bench | vLLM `benchmarks/kernels/benchmark_kimi_k3_*.py`（AttnRes、KDA decode、KDA 投影、latent MoE tail、SP 通信），SGLang `benchmark/bench_linear_attention/`，FLA `benchmarks/` | 各自一套 | | | | **已有 K3 专用脚本** | 当作 **Solution 来源**和交叉验证 |
| 模型级解析 roofline | [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer)、GenZ 一类 | 按层累加 FLOPs/Bytes | | | 解析式 | 不支持 KDA/AttnRes | 借鉴思路，自己实现 §5.1 |
| KernelBench、GPU MODE | LLM 生成 kernel 的评测 | | | | | | 现阶段用不上 |

---

## 4. 仓库结构和三方依赖

目录结构、分层，以及哪些库 pip 安装、哪些 clone，都单独写在 [repo-architecture.md](repo-architecture.md)。要点：

- 事实来源是 Python 的 `k3ops/ops/*`（shape、reference、输入生成、cost model 都在这里）。FIB 的 JSON 由生成器产出到 `dataset/`。
- 不需要 clone FlashInfer-Bench，pip 安装即可。GPU 环境以 vLLM nightly 为底座，它已经内置了 FlashKDA、DeepGEMM 和 K3 专用 kernel。`third_party/` 里只放只读的源码参考。

**闭环流程：**

```
Definition（契约+reference）
  → Workload（random 或真实抓取）
  → Solutions（N 种实现）
  → flashinfer-bench run（正确性 + CUPTI 延迟）
  → traces → cost_model → SOL% 表 / roofline 图
  → 按"时间占比 × (1-SOL%)"排序，挑出头部算子
  → ncu / Nsight Python 深挖 → 优化 → 作为新 Solution 加回来
```

---

## 5. Roofline

### 5.1 解析式（每个 Definition 配一对 `flops(axes)` 和 `bytes(axes)`）

- 理论时间 `T_roof = max(FLOPs / P_peak, Bytes / BW_peak)`，`SOL% = T_roof / T_measured`。这是给所有算子排序时**最有用的单一指标**。
- 常用公式：
  - GEMM `[M,K]×[N,K]ᵀ`：FLOPs = 2MNK；Bytes = M·K·a + N·K·w + M·N·o（+ scale）。
  - MXFP4 权重每个元素按 0.5 B + 1/32 B（e8m0 scale）≈ **0.531 B** 计。小 m 时 AI ≈ 2m/0.531 ≈ **3.76·m**。
  - MoE grouped GEMM：FLOPs = 2·Σmₑ·(3584·6144 + 3072·3584)；Bytes ≈ 被激活的 expert 数 × 33.03M × 0.531 B + 激活值。被激活的 expert 数 ≈ 896·(1-(1-16/896)^T)。
  - KDA recurrent decode：Bytes ≈ B·96·128²·4·2（状态读写）+ q/k/v/g/o；FLOPs ≈ B·96·~8·128²，所以 **AI≈1**。
  - MLA decode（absorbed）：每个 cache token 需要 FLOPs = 2·H·(576+512)，Bytes = 1152，AI ≈ 1.89·H（H=96 时约 181，H=12 时约 23）。
- 硬件峰值**以实测为准**：用大 GEMM 测各精度的 FLOPS，用 stream/memcpy 测带宽，写进 `perf/hw.yaml`。规格表上的约值（dense）：

| GPU | BF16 | FP8 | FP4 | HBM | BF16 ridge | FP8 ridge |
|---|---|---|---|---|---|---|
| H100 SXM | ~989 TF | ~1979 TF | — | 3.35 TB/s | ~295 | ~590 |
| H200 | ~989 TF | ~1979 TF | — | 4.8 TB/s | ~206 | ~412 |
| B200 | ~2.25 PF | ~4.5 PF | ~9 PF | ~8 TB/s | ~281 | ~562 |

> MXFP4 权重 × MXFP8 激活（W4A8）的混合精度 MMA，一般按 FP8 那一档的吞吐算，请用实测确认。Hopper 没有原生 FP4，只能在 kernel 内反量化（W4A16 或 W4A8），ridge 要另算。

### 5.2 实测

- ncu 的 `SpeedOfLight_RooflineChart` 和 `SpeedOfLight_Hierarchical*RooflineChart` 这几个 section 会直接画出 roofline（用 `ncu --list-sections` 确认当前版本的 section 名）。
- **用实测 DRAM 字节数 `dram__bytes.sum` 对照解析式的最小字节数。** 实测明显偏大，说明数据复用差、L2 miss 多，或者有中间结果写回了 HBM（没融合）。这是判断"该不该融合"的直接证据。

---

## 6. Nsight：SM 利用率和 warp occupancy

```bash
# ① 时间线：找热点和 launch 间隙；CUDA graph 按节点展开
nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=node -o results/k3_layer python perf/run_layer.py
```

```bash
nsys stats --report cuda_gpu_kern_sum results/k3_layer.nsys-rep
```

```bash
# ② 单个 kernel 深挖：只抓 NVTX range 里的目标 kernel，跳过 warmup
ncu --nvtx --nvtx-include "kda_decode/" \
    --section SpeedOfLight --section SpeedOfLight_RooflineChart \
    --section LaunchStats --section Occupancy \
    --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis \
    --launch-skip 5 --launch-count 1 \
    -o results/kda_decode_b64 python perf/bench_one.py --op kda_decode --B 64
```

```bash
ncu --import results/kda_decode_b64.ncu-rep --csv --page raw > results/kda_decode_b64.csv
```

**主要看这些指标：**

| 关注点 | metric / section | 怎么读 |
|---|---|---|
| 计算 SOL | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 和下一行比，看是算力受限还是访存受限 |
| 访存 SOL | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`、`dram__throughput.avg.pct_of_peak_sustained_elapsed` | 访存受限的算子目标是 >80% |
| SM 是否都在干活 | `launch__grid_size` 和 SM 数比较（H100 是 132，B200 是 148）、`launch__waves_per_multiprocessor`、`sm__cycles_active.avg` | decode 的小 kernel 常常 grid 小于 SM 数，只有部分 SM 在工作。需要 split-K 或 split-KV、persistent kernel |
| 理论 occupancy 和限制因素 | `sm__maximum_warps_per_active_cycle_pct`、`launch__occupancy_limit_{registers,shared_mem,warps,blocks}`、`launch__registers_per_thread` | 看是寄存器还是 smem 在卡 occupancy |
| 实际 occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | 比理论值低很多，通常是 tail effect 或负载不均 |
| 发射效率和停顿 | `smsp__issue_active.avg.pct_of_peak_sustained_active`，WarpStateStats 里的 stall reasons | `long_scoreboard` 表示在等访存，另外看 `barrier`、`mio_throttle`、`wait` |
| Tensor Core | 用 `ncu --query-metrics \| grep -i tensor` 查当前架构对应的 pipe 指标 | Hopper（wgmma）和 Blackwell（tcgen05）的指标名不一样 |

**几个容易踩的坑：**

- **低 occupancy 不一定是问题。** FlashMLA、DeepGEMM、TRTLLM-Gen、FA3/FA4 这类 Hopper/Blackwell 上的 warp-specialized kernel 通常每个 SM 只跑 1 个 CTA，occupancy 低是有意设计的。判断时以 SOL% 为主，occupancy 只作参考。
- **ncu 测出的 duration 不等于真实延迟。** ncu 默认 `--clock-control base`（锁基频）和 `--cache-control all`（清 cache），还会多次 replay。真实延迟以 CUPTI 或 CUDA event 的结果为准。
- **需要性能计数器权限。** 云上和容器里常见 `ERR_NVGPUCTRPERM` 错误，需要管理员打开权限，或者容器加 `--cap-add SYS_ADMIN`。
- **报告可以拿回 Mac 看。** Nsight Compute 和 Nsight Systems 的 GUI 都支持 macOS，`.ncu-rep` 和 `.nsys-rep` 下载回来就能打开。

**用 Nsight Python 做 sweep 的示意**（具体 API 以[官方文档](https://docs.nvidia.com/nsight-python/analyze.html)为准）：

```python
import nsight

@nsight.analyze.kernel(
    configs=[(impl, B) for impl in ("fla", "flashkda") for B in (1, 8, 32, 128, 512)],
    runs=5,
    metrics=[
        "gpu__time_duration.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ],
    output_csv=True,
)
def sweep_kda_decode(impl, B):
    args = make_inputs(B)                 # 输入在 annotate 外面准备，不计入 profile
    with nsight.annotate("kda_decode"):
        IMPLS[impl](**args)
```

---

## 7. Benchmark 方法论要点

- **计时**：用 CUPTI（FIB 默认）或 CUDA graph 去掉 launch 开销。报中位数和 p10/p90。warmup 要充分。锁频（`nvidia-smi -lgc`），或者至少把频率记录下来。测试时独占 GPU。
- **L2 冷热**：真实 decode 时权重和 KDA 状态在 L2 里是"冷"的，因为 93 层轮转，单层数据量远大于 L2。所以默认测冷 L2（FIB 默认 `cold_l2_cache=True`），热 L2 的结果只作对照。
- **random 输入什么时候不行**：
  - **MoE 路由**：random logits 会让 expert 负载接近均匀，而真实路由是偏斜的，这会影响 grouped GEMM 的负载不均和 EP 的慢节点。workload 至少要准备 uniform、Zipf、真实抓取三种。
  - **Sampling**：top-k/top-p 的 rejection sampling 迭代次数取决于 logits 的分布。
  - **变长**：prefill 的序列长度分布、MLA decode 的上下文长度分布（cu_seqlens、kv_indptr）都会影响性能。
  - **FP4 权重**：先生成 random BF16，再用正式的量化器量化。不要直接填随机字节：e8m0 的 0xFF 表示 NaN。
  - **KDA 的 gate、A_log、dt_bias**：数值不影响性能，但会影响 reference 的数值稳定性。按模型初始化的分布来生成（A_log 取 log(U(1,16))）。
  - **需要验证真实数值时**：只下载一个 safetensors shard，取一层的真实权重。
- **shape 要按 per-rank 定**：同一个算子在 attention TP8（12 头）和 DP-attn（96 头）下 shape 不同，MoE 在 EP16 下是每卡 56 个 expert。`shapes.py` 根据并行场景生成 axes。Definition 的 const 轴把头数写进名字（参照 FIB 现有定义的 `h16`/`tp:1` 标签）。
- **正确性**：BF16 用 atol/rtol 约 1e-2。FP8/FP4 和反量化后的 reference 比，用 cosine 或 SNR。KDA 的状态要单独比 fp32 的终态。

---

## 8. 路线图

| 阶段 | 内容 | 产出 | 需要 GPU |
|---|---|---|---|
| **P0 环境**（0.5 天） | GPU 节点（Hopper 或 Blackwell，驱动要支持 CUDA 13）；`scripts/setup_env.sh all` 建好 vLLM 0.32.0 和 SGLang 0.5.22 两个环境；ncu/nsys 版本和驱动匹配；`check_env.py --ncu` 确认计数器权限；用 microbench 测出实际峰值 | `perf/hw.yaml` | ✅ |
| **P1 shape 和解析 roofline**（1~2 天） | `k3ops/model/{kimi_k3,parallel}.py`：config + 并行场景 → 每个算子的 per-rank axes；`k3ops/ops/*` 的 `cost()`：FLOPs/Bytes；按场景（decode B=1..512，prefill 4K/16K）汇总出"每步每层的理论时间分解" | 算子 × 场景的 FLOPs/Bytes/AI/T_roof 大表，以及模型级的理论时间分解 | ❌（Mac 可跑） |
| **P2 Definition 和 Solution** | 先做 FIB 已有、只需改 shape 的：rmsnorm、gemm、mla_paged decode、MoE routing、mxfp4 grouped GEMM。再做新增的：KDA prefill/decode（照 GDN 定义改）、AttnRes(+RMSNorm)、SiTU、ShortConv、Latent MoE。每个至少 torch_ref 加 2 个实现 | `definitions/`、`solutions/`、random `workloads/` | 写 ❌，验证 ✅ |
| **P3 跑分** | `flashinfer-bench run --local .`，再用 traces join cost_model | SOL% 表和 roofline 图 | ✅ |
| **P4 深挖** | 按"时间占比 × (1-SOL%)"挑 Top-N，用 nsys 看时间线，用 ncu 和 Nsight Python 做 sweep | 每个热点算子一页分析（瓶颈、证据、优化方向） | ✅ |
| **P5 真实 workload** | 用 vLLM/SGLang 跑 K3，`FIB_ENABLE_TRACING=1` 抓真实 shape、路由和 seqlen 分布。没有足够的卡时，可以用同架构的小模型 Kimi-Linear 抓分布（shape 要换成 K3 的） | `workloads/` 真实数据 | ✅（多卡） |

**decode 热点的预估排序**（P1 的表会给出实际排序）：

1. MXFP4 routed experts 的 grouped GEMM 和 EP 通信
2. BF16 attention 投影的 GEMV
3. KDA recurrent 状态读写
4. MLA decode
5. AttnRes + RMSNorm
6. router top-k
7. sampling

prefill 主要看 KDA chunk、MLA ragged prefill 和 MoE GEMM。

---

## 参考

- Kimi K3 模型卡与 config：<https://huggingface.co/moonshotai/Kimi-K3>
- vLLM Kimi K3 预览（kernel 路径：FlashKDA/FLA、TRTLLM-Gen/DeepGEMM MXFP4、AttnRes 融合）：<https://vllm.ai/blog/2026-07-22-kimi-k3-preview>
- FlashKDA：<https://github.com/MoonshotAI/FlashKDA>
- FlashInfer-Bench 论文：<https://arxiv.org/abs/2601.00227>
- FlashInfer-Bench 代码：<https://github.com/flashinfer-ai/flashinfer-bench>
- FlashInfer-Bench 数据集：<https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace>
- TritonBench：<https://github.com/meta-pytorch/tritonbench>
- Nsight Python：<https://docs.nvidia.com/nsight-python/analyze.html>
