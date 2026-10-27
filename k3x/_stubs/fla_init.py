"""上游 fla/__init__.py 会按字符串 import fla.layers / fla.models（改名后会误加载已安装的 fla）；副本只需要 __version__。"""

__version__ = "0.6.0"
