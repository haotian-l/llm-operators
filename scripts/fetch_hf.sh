#!/usr/bin/env bash
# 下载 Kimi K3 的 config 和 modeling 代码到 models/kimi_k3/hf/（不下载权重）。
# revision 锁定，升级时改 REV 并重新运行。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="moonshotai/Kimi-K3"
REV="${REV:-f831ab66814297da540d832a5235f8e904f29d06}"
OUT="$ROOT/models/kimi_k3/hf"
FILES=(
  config.json
  generation_config.json
  configuration_kimi_k3.py
  modeling_kimi_linear.py
  modeling_kimi_k3.py
  LICENSE
)

mkdir -p "$OUT"
for f in "${FILES[@]}"; do
  curl -fsSL -m 120 "https://huggingface.co/$REPO/resolve/$REV/$f" -o "$OUT/$f"
  echo "  $f"
done
echo "$REPO@$REV" > "$OUT/REVISION"
echo "==> $OUT ($REPO@$REV)"
