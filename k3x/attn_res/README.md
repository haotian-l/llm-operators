# AttnRes（Attention Residuals）研读指南

本目录下每个子目录是一份从上游抽出的实现。抽取由 `scripts/extract.py` 按 `k3x/manifest.py` 完成，来源、commit、sha256 和替身都记在各自的 `SOURCE.json` 里。上游版本：vLLM v0.32.0、SGLang v0.5.22、FLA v0.6.0。flashinfer 没有 AttnRes 实现。

> 副本是只读的；要做实验，就把文件复制到 `kernels/` 下再改。

---

## 1. 它在算什么

K3 不用普通的残差连接 `h = h + f(h)`。每层的输入是**沿深度方向做注意力**得到的：拿若干个"历史快照"加上当前的累积残差当作源，每个源先做 RMSNorm，再和一个可学习向量打分，沿深度 softmax 后加权求和。参考代码是 `models/kimi_k3/hf/modeling_kimi_linear.py` 的 `_apply_attn_res`（L1075）和 `KimiDecoderLayer._forward_attn_residual`（L973）。

```
sources = [block_0, …, block_{nb-1}, prefix]      # 每个形状都是 [T, 7168]，nb ≤ 8
kᵢ      = sourceᵢ / sqrt(mean(sourceᵢ²) + eps)     # 不带权重的 RMSNorm
w       = norm.weight ⊙ proj.weight                # [7168]，两个权重可以预先乘好
pᵢ      = softmax_i( kᵢ · w )                      # 每个 token 在 nb+1 个源上做 softmax
h       = Σᵢ pᵢ · sourceᵢ
```

**每层的数据流：**

1. **attention 之前混合一次**：`h = AttnRes(prefix, blocks)`，然后做 `input_layernorm`，再进 attention。第 0 层没有 block，这一步跳过。
2. **写快照**：当 `layer_idx % 12 == 0`（`attn_res_block_size=12`）时，把当前 `prefix` 追加进 `blocks`，并把 `prefix` 清零重新累积。
3. **attention 之后**：`prefix += attn_out`。
4. **MLP 之前再混合一次**：`h = AttnRes(prefix, blocks)`，然后做 `post_attention_layernorm`，再进 MoE。
5. **MLP 之后**：`prefix += mlp_out`。
6. **最后一层之后**：再做一次输出混合，然后接最终的 norm。

**规模**：93 层一共要调用 **186 次**。快照最多 8 个，所以每次调用最多有 9 个源。

**访存量**（按上面的流程逐层累加，TP=1，bf16）：
- 每个 token 一次前向要读 14.4 MB、写 2.7 MB，合计约 **17 MB**。
- 算术强度约 2.5 FLOP/B，是纯访存受限的算子。
- prefill 8K token 时总访存约 140 GB，按 8 TB/s 算约 17 ms，**不能忽视**。

所以生产实现的方向是：
- 把残差加法、写快照、混合、紧跟着的 RMSNorm 融成一个 kernel，避免中间结果反复读写 HBM；
- 在序列并行下把这部分访存分摊到各张卡上（vLLM 博客提到了这一点，SGLang 有和 reduce-scatter / all-gather 融合的变体）。

---

## 2. 生产路径

### vLLM v0.32.0

入口是 `third_party/vllm/vllm/models/kimi_k3/nvidia/ops/attn_res.py` 的 `attn_res(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, num_blocks, block_write_idx, eps, output_norm_eps)`，调用点在 `nvidia/model.py` 的 L1024、L1057、L1270、L1383。

一个调用里融合了四件事：
- 残差加法：`prefix + delta`；
- 写快照：`block_write_idx ≥ 0` 时把结果写进 `blocks`；
- AttnRes 混合；
- 输出 RMSNorm：也就是后面的 `input_layernorm` 或 `post_attention_layernorm`。

选哪个实现：
1. **SM10x**，并且 hidden=7168、0 ≤ num_blocks ≤ 8、按 token 连续存储 → **CUDA** `kimi_k3_attn_res`，对应 → `vllm_cuda/`
2. **其他情况** → **Triton** `_attn_res_kernel`，对应 → `vllm_triton/`
   - grid = (num_tokens,)，`BLOCK_D` = 8192，覆盖整个 hidden。
   - num_tokens ≥ 256 或 num_blocks ≤ 1（最多 2 个源）时，每次处理 1 个源、4 个 warp；否则每次处理 4 个源、8 个 warp。注释说这组参数是在 GB300 上调出来的：按源分块对 decode 有利，每次一个源对 prefill 扩展性更好。

### SGLang v0.5.22

入口是 `third_party/sglang/python/sglang/srt/layers/attn_residual.py` 的 `_aggregate`（L400），只在混合这一步分发：
1. **SM10x/SM11x**（不含 SM12x）并且 H=7168 → **TMA CUDA 快速路径** `attn_res_fused_tma`，对应 → `sglang_cuda/`
2. **AMD** → HIP 路径（不在本次范围内）
3. **其他情况** → **Triton 两段式** `_aggregate_fused`：先用 `_score_kernel` 打分，再用 `_combine_kernel` 混合，最后做 RMSNorm。对应 → `sglang_triton/`

`AttnResidual` 类（L443）负责管理快照库（bank）。序列并行时还会用 `k3_sp_collective` 做 all-gather / reduce-scatter，`sglang_cuda/` 里的多卡融合变体就是为这个准备的。

---

## 3. 抽出来的实现

| 目录 | 语言 | 融合范围 | 主文件 | 行数 |
|---|---|---|---|---|
| `torch_naive/` | PyTorch | 混合，可选再接输出 RMSNorm | `fla/ops/attnres/naive.py` → `naive_attnres` | 83 |
| `fla_triton/` | Triton | 混合（前向和反向） | `fla/ops/attnres/fused.py` → `attnres_fwd_kernel` | 572 |
| `fla_gluon/` | Gluon | 同上 | `fla/ops/attnres/backends/gluon.py` | 772 |
| `vllm_triton/` | Triton | 残差加法 + 写快照 + 混合 + 输出 RMSNorm | `vllm/models/kimi_k3/nvidia/ops/attn_res.py` → `_attn_res_kernel` | 249 |
| `sglang_triton/` | Triton | 打分 / 混合两段 + RMSNorm，外加分发逻辑和快照库管理 | `sglang/srt/layers/attn_residual.py` | 620 |
| `vllm_cuda/` | CUDA（SM100） | 和 vLLM Triton 版相同 | `kimi_k3/attn_res_kernel.cu` | 1033 |
| `sglang_cuda/` | CUDA（SM10x/11x） | 混合 + 写快照，还有和 NVLS reduce-scatter / all-gather 融合的多卡版 | `jit/csrc/kimi_k3/attn_res/fused_tma.cuh` | 911 |

**和 K3 的对应关系**：FLA 的 `naive_attnres(query, residuals, rms_weight, output_rms_weight)` 是论文里的通用形式（arXiv 2603.15031）。对应到 K3：
- `query` = `proj.weight`
- `rms_weight` = `norm.weight`
- `residuals` = 快照加上当前的 prefix
- `output_rms_weight` = 紧跟着的那个 layernorm

**为了能独立运行而用的替身**：详见各目录的 `SOURCE.json` → `stubs`，替身源码在 `k3x/_stubs/`。
- vLLM：替换了 `_custom_ops`、`platforms`、`triton_utils`。
- SGLang：`RMSNorm` 和 `ReplicatedLinear` 换成了纯 PyTorch 的最小版本；TMA 快速路径、HIP 路径和多卡通信改成直接报错。
  - **在 SM10x 上运行副本的 `_aggregate` 会走到 TMA 的替身并报错。** 想看 Triton 路径，直接调用 `_aggregate_fused`。

---

## 4. 建议的阅读顺序和要回答的问题

**1. `torch_naive`，对照 HF 的 `_apply_attn_res`：先弄清楚数学。**
- 为什么 `norm.weight` 和 `proj.weight` 可以预先乘在一起？
- 每个 token 在每次调用里读多少字节？

**2. `vllm_triton` 的 `_attn_res_kernel`：一个 token 对应一个 program，整个 hidden 放在一个 tile 里。**
- `BLOCK_L` 按源分块是怎么做的？
- online softmax 是否必要？
- 融合残差加法、写快照和输出 RMSNorm 一共省掉了几次 HBM 读写？

**3. `sglang_triton` 的 `_score_kernel` 和 `_combine_kernel`：同一个算法拆成两段。**
- 拆开的代价是什么？分数要写回去再读一次。
- 好处是什么？`_BLOCK_H=1024` 把 7168 切成 7 段，提高并行度。

**4. `fla_triton` 和 `fla_gluon`：同一个前向，分别用 Triton 和更底层的 Gluon 写。**
- Gluon 版显式指定了 warp 划分和 `cp.async`，看看相比 Triton 多控制了什么。

**5. `vllm_cuda/kimi_k3/attn_res_kernel.cu`：Blackwell 上的 warp-specialized 实现。**
- 1 个 producer warp 用 `cp.async.bulk` 把行搬进 shared memory，8 个 consumer warp 做归约和输出。
- `Q = res_weight * rms_weight` 一直留在寄存器里，跨 persistent token 复用。
- **源向量第一遍读进来后转成 FP32 存进 TMEM**，第二遍直接从 TMEM 取。TMEM 是 Blackwell 新增的 Tensor Memory，这里被当成大容量暂存用。

**6. `sglang_cuda/.../fused_tma.cuh`：TMA + mbarrier + tcgen05/TMEM 的单卡版本，以及两个多卡变体。**
- L572 起：NVLS pull reduce-scatter 和本地 AttnRes 融合。
- L627 起：本地 AttnRes 和直接 all-gather 的 epilogue 融合。
- 和第 5 步对照着读，看两家在同一硬件上做出了哪些不同的选择。

---

## 5. 还没做的部分

- **CUDA 副本的独立编译胶水**：vLLM 用 torch stable ABI，op 名是 `kimi_k3_attn_res`；SGLang 用 tvm-ffi JIT，见 `sglang_cuda/ops/attention/attn_res.py` 里的 `load_jit` 调用。
- **vLLM main 上更新的 CuTe DSL 版本**：`vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/attn_res/`，0.32.0 里还没有。
- **反向传播**：FLA 的反向 kernel 已经在副本里，但推理用不到。

维护和使用方式同 `k3x/kda_decode/README.md` §6。
