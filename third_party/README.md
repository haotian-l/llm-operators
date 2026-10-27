# third_party

只读的源码参考，**不安装**。运行时用到的包由 `scripts/setup_env.sh` 从 pip 安装，版本和这里一一对应；升级时 `requirements/*.txt` 和子模块要一起改。

| 子模块 | 版本 | commit | 对应的 pip 包 | 用途 |
|---|---|---|---|---|
| vllm | v0.32.0 | `5df63a7` | `vllm==0.32.0`（envs/vllm） | K3 kernel 源码 `csrc/kimi_k3/`、`vllm/models/kimi_k3/`；现成微基准 `benchmarks/kernels/benchmark_kimi_k3_*.py`（wheel 里没有） |
| sglang | v0.5.22 | `baa9bef` | `sglang==0.5.22`（envs/sglang） | `python/sglang/srt/models/kimi_k3.py`、`python/sglang/kernels/`（`BaseFusedOp` 多 backend）、`test/registered/kernels/benchmark/attention/bench_kimi_k3_kda_decode.py`、`benchmark/bench_linear_attention/` |
| flash-linear-attention | v0.6.0 | `413ad2c` | `flash-linear-attention==0.6.0`（两个环境都装，`--no-deps`） | KDA 的 Triton 源码；`fla/ops/kda/naive.py` 可以当参考实现的起点；AttnRes 的参考实现、Triton 版和 Gluon 版 |
| flashinfer | v0.7.1 | `0b46115` | envs/sglang 里的 `flashinfer-python==0.7.1`（envs/vllm 用的是 0.7.0.post1） | KDA decode 的 CuTe DSL kernel（`flashinfer/kda_kernels/`）；Cake 生成的 CUDA 变体在 `csrc/kda/`。子模块约 475 MB，大头在 `csrc/` |

新 clone 本仓库之后：

```bash
git submodule update --init
```

`.gitmodules` 里设置了 `shallow = true`，所以每个子模块只拉 pin 住的那个 commit。

按需再加（见 `docs/repo-architecture.md` §3.4）：FlashKDA（vllm-project fork `17a037d`）、DeepGEMM、DeepEP、cutlass。

这些子模块也是 `scripts/extract.py` 抽取 kernel 研读副本（`k3x/`）的来源。升级子模块之后，要跑一次 `python scripts/extract.py --check`：它会把过时的副本标成 `[STALE]`。
