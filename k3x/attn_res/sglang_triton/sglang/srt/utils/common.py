"""上游 sglang/srt/utils/common.py 有几千行并牵出整个 sglang.srt；这里只需要 torch_release（逐字照抄上游定义）。"""

import torch
from packaging import version as pkg_version

torch_release = pkg_version.parse(torch.__version__).release
