# Operators：Kimi K3 算子 benchmark 和 profiling

把 Kimi K3 拆成独立的算子。每个算子有输入、输出和 PyTorch 参考实现，用多种实现对比，覆盖 vLLM、SGLang、flashinfer、FLA 和自写 kernel。做正确性校验、计时、roofline 分析和 Nsight（ncu/nsys）分析。

- 方案和方法论：[docs/kimi-k3-operator-bench-plan.md](docs/kimi-k3-operator-bench-plan.md)
- 代码架构和三方依赖：[docs/repo-architecture.md](docs/repo-architecture.md)

## 目录

| 路径 | 内容 |
|---|---|
| `k3ops/` | 本仓库的 Python 包：`model/`（config 解析和并行场景）、`ops/`（算子定义）、`impls/`（各实现的适配器）、`fib/`（生成 FlashInfer-Bench 数据集）、`perf/`（roofline、ncu、nsys）、`report/` |
| `k3x/` | 从 vLLM、SGLang、FLA、flashinfer 抽出的 kernel 研读副本，按算子组织，每个算子有一份阅读指南（`k3x/<op>/README.md`）。已有 `kda_decode`（8 个实现）和 `attn_res`（7 个实现）。由 `scripts/extract.py` 按 `k3x/manifest.py` 生成，只读 |
| `models/kimi_k3/hf/` | HF 快照（config 和 modeling 代码，revision 锁定，不含权重） |
| `templates/flashinfer-trace/` | flashinfer-trace 的模板 Definition（Apache-2.0） |
| `dataset/` | 生成出来的 FIB 数据集，用 `flashinfer-bench run --local dataset` 跑 |
| `third_party/` | 只读源码参考：vllm v0.32.0、sglang v0.5.22、flash-linear-attention v0.6.0、flashinfer v0.7.1 |
| `requirements/`、`scripts/` | 环境锁定和安装、检查、下载脚本 |
| `kernels/` | 自写 kernel |
| `results/` | 跑分结果（`.ncu-rep`、`.nsys-rep` 不进 git） |

## 运行流程与产出

下面是规划好的完整流程。✅ 表示已经有了，其余还没实现。

| 步骤 | 在哪跑 | 做什么 | 产出 |
|---|---|---|---|
| 0. 研读 ✅ | 任意 | `scripts/extract.py` 抽取 kernel，配合 `k3x/<op>/README.md` 读 | 按算子组织的源码副本和阅读指南；抽出的副本也可以注册成 Solution，和已安装的版本对拍 |
| 1. 理论表（P1） | Mac | 用 config 和并行场景（TP、DP-attn、EP、batch、上下文长度）算出每个算子在每张卡上的 shape、FLOPs、访存量 | 一张"算子 × 场景"的表：算术强度、理论下界 T_roof，以及 decode/prefill 一步里各算子的理论耗时占比 |
| 2. 生成数据集（P2） | Mac | `k3ops/ops` 生成 FIB 的 Definition、Workload（random / Zipf / 真实抓取）和 Solution | `dataset/` 下的 JSON |
| 3. 跑分（P3） | GPU | `envs/vllm` 和 `envs/sglang` 各跑一遍 `flashinfer-bench run`，再合并 trace | 每个 (算子, 实现, shape) 是否正确（和参考实现比数值），以及 CUPTI 测的延迟 p50/p10/p90 |
| 4. 汇总 | 任意 | trace 和第 1 步的理论值对齐 | `results/<gpu>/<date>/summary.csv` 和 roofline 图：`SOL% = T_roof / 实测`，按"时间占比 × (1-SOL%)"排出优化优先级 |
| 5. 深挖（P4） | GPU | 对排在前面的算子跑 ncu / Nsight Python 扫参数、nsys 看时间线 | 计算和访存的 SOL、实测 DRAM 字节与理论值的差、理论和实际 occupancy、warp stall 原因 |

**最终能回答的问题**，针对 K3 的每个算子：
- 在给定的 GPU 和 shape 下，vLLM、SGLang、flashinfer、FLA 和自写实现谁最快、谁算得对；
- 离硬件上限还有多远；
- 卡在哪里：访存、延迟、occupancy 还是 tail effect；
- 下一步该融合或优化什么。

`summary.csv` 的列大致是：

| op | scenario | impl | env | correct | p50 (µs) | T_roof (µs) | SOL% | bound | DRAM GB/s | occupancy |
|---|---|---|---|---|---|---|---|---|---|---|

## 快速开始

在 GPU 机器上，vLLM 和 SGLang 各建一个环境，两边地位相同：

```bash
git submodule update --init
```

```bash
scripts/setup_env.sh all
```

```bash
envs/sglang/bin/python scripts/check_env.py --ncu
```

在 Mac 上开发（生成 shape/cost 表、Definition），用 Python ≥3.10：

```bash
uv venv --python 3.12 .venv && uv pip install -r requirements/base.txt && uv pip install -e .
```

抽取或校验 kernel 研读副本（只用标准库，Mac 也能跑）：

```bash
python3 scripts/extract.py
```

```bash
python3 scripts/extract.py --check
```

更新 HF 快照或模板：

```bash
scripts/fetch_hf.sh
```

```bash
scripts/fetch_fib_templates.sh
```
