#!/usr/bin/env bash
# 从 HF 数据集 flashinfer-ai/flashinfer-trace（Apache-2.0）拉取模板 Definition，
# 放到 templates/flashinfer-trace/<op_type>/。生成 K3 的 Definition 时以它们为蓝本。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="flashinfer-ai/flashinfer-trace"
REV="${REV:-da915083d4c7c5e61aa3005e3d17ae488e0fc71c}"
OUT="$ROOT/templates/flashinfer-trace"
FILES=(
  definition.md
  solution.md
  trace.md
  definitions/gdn/gdn_decode_qk16_v32_d128_k_last.json
  definitions/gdn/gdn_prefill_qk16_v32_d128_k_last.json
  definitions/mla_paged/mla_paged_decode_h16_ckv512_kpe64_ps64.json
  definitions/mla_paged/mla_paged_prefill_causal_h16_ckv512_kpe64_ps64.json
  definitions/moe/moe_fp8_block_scale_ds_routing_topk8_ng1_kg1_e384_h7168_i2048.json
  definitions/moe/trtllm_fp4_block_scale_moe_topk8_e128_h2048_i768.json
  definitions/gemm/grouped_gemm_mxfp8_mxfp4_g4_n2048_k2048.json
  definitions/gemm/gemm_mxfp4_n4096_k4096.json
  definitions/gemm/gemm_n256_k7168.json
  definitions/rmsnorm/rmsnorm_h7168.json
  definitions/rmsnorm/fused_add_rmsnorm_h7168.json
  definitions/rmsnorm/rmsnorm_h1536.json
  definitions/rmsnorm/rmsnorm_h512.json
)

for f in "${FILES[@]}"; do
  dst="$OUT/${f#definitions/}"
  mkdir -p "$(dirname "$dst")"
  curl -fsSL -m 60 "https://huggingface.co/datasets/$REPO/resolve/$REV/$f" -o "$dst"
  echo "  $f"
done

# mla_ragged 和 sampling 的文件名随数据集更新变化，按目录列出后匹配
fetch_matching() {
  local dir="$1" pattern="$2"
  curl -fsSL -m 60 "https://huggingface.co/api/datasets/$REPO/tree/$REV/definitions/$dir" \
    | grep -oE "\"path\":\"definitions/$dir/[^\"]*$pattern[^\"]*\"" \
    | sed -E 's/"path":"(.*)"/\1/' \
    | while read -r f; do
        dst="$OUT/${f#definitions/}"
        mkdir -p "$(dirname "$dst")"
        curl -fsSL -m 60 "https://huggingface.co/datasets/$REPO/resolve/$REV/$f" -o "$dst"
        echo "  $f"
      done
}
fetch_matching mla_ragged "qk192_vo128"
fetch_matching sampling "v129280"

echo "$REPO@$REV" > "$OUT/REVISION"
echo "==> $OUT ($REPO@$REV)"
