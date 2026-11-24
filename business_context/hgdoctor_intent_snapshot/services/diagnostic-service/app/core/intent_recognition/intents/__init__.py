"""
各功能模块的意图注册入口

每个模块提供一个 register_xxx_intent(registry) 函数，
在应用启动时由 main.py lifespan 统一调用。
支持自动发现：目录下所有 *_intent.py 中的 register_*_intent 函数自动注册。
"""

import importlib
import logging
from pathlib import Path

from ..registry import IntentRegistry

logger = logging.getLogger(__name__)


def register_all_intents(registry: IntentRegistry) -> None:
    """自动发现并注册 intents/ 目录下所有意图。

    扫描当前目录下所有 *_intent.py 文件，
    找到名为 register_*_intent 的函数并调用。
    """
    intents_dir = Path(__file__).parent
    for file in sorted(intents_dir.glob("*_intent.py")):
        module_name = file.stem
        try:
            mod = importlib.import_module(f".{module_name}", package=__name__)
            # 找到 register_xxx_intent 函数
            for attr_name in dir(mod):
                if attr_name.startswith("register_") and attr_name.endswith("_intent"):
                    func = getattr(mod, attr_name)
                    if callable(func):
                        func(registry)
        except Exception as e:
            logger.error("加载意图模块 %s 失败: %s", module_name, e)


__all__ = ["register_all_intents"]
