"""重导出按场景磁盘懒缓存接口。

模块: data/replay_cache/__init__.py
依赖: data.replay_cache.replay_cache
读取配置: model_data.replay_cache.*
对外接口:
    - ReplayDiskCache
"""

from data.replay_cache.replay_cache import ReplayDiskCache

__all__ = ["ReplayDiskCache"]

