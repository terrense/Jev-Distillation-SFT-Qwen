"""
意图注册表

管理所有已注册的意图定义，提供 prompt 片段生成能力。
各功能模块在应用启动时通过 register() 方法注入自己的 IntentDefinition。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .models import IntentDefinition

logger = logging.getLogger(__name__)


class IntentRegistry:
    """意图注册表

    收集各模块注册的 IntentDefinition，提供：
    - 注册/查询接口
    - 全局配置加载（idle_response、max_clarification_rounds、disabled_intents）
    - prompt 片段生成（将所有已启用 intent 拼接为 LLM 可用的文本）
    """

    def __init__(self):
        self._intents: Dict[str, IntentDefinition] = {}
        # 全局配置（默认值）
        # idle 意图（原 unknown）无法识别时的追问兜底文案
        self.idle_response: str = (
            "抱歉，没太听明白您的需求。请问您是想看病问诊、了解该挂什么科，还是想查询地点或就医流程？"
        )
        self.max_clarification_rounds: int = 3
        self.disabled_intents: List[str] = []

    def register(self, intent_def: IntentDefinition) -> None:
        """注册一个意图定义。

        如果同名 intent 已存在，会覆盖并记录警告日志。

        Args:
            intent_def: 意图定义实例
        """
        if intent_def.name in self._intents:
            logger.warning(
                "意图 '%s' 已注册，将被覆盖", intent_def.name
            )
        self._intents[intent_def.name] = intent_def
        logger.info("注册意图: %s (priority=%d)", intent_def.name, intent_def.priority)

    def get_intent(self, name: str) -> Optional[IntentDefinition]:
        """按名称查询意图定义。

        Args:
            name: 意图标识

        Returns:
            IntentDefinition 或 None（未找到时）
        """
        return self._intents.get(name)

    def get_enabled_intents(self) -> List[IntentDefinition]:
        """返回所有已启用的意图（按 priority 升序排列）。

        排除 disabled_intents 配置中列出的意图。
        """
        enabled = [
            intent
            for intent in self._intents.values()
            if intent.name not in self.disabled_intents
        ]
        enabled.sort(key=lambda x: x.priority)
        return enabled

    def get_intent_names(self) -> List[str]:
        """返回所有已启用意图的名称列表（按 priority 排序）。"""
        return [intent.name for intent in self.get_enabled_intents()]

    def build_prompt_fragment(self) -> str:
        """将所有已启用 intent 的信息组装为 LLM prompt 片段。

        生成格式：
        ```
        ## 可识别的意图列表

        ### 1. medical
        描述: ...
        示例用户输入:
        - "..."
        - "..."
        额外提取要求:
        ...

        ### 2. inquiry
        ...
        ```

        Returns:
            拼接好的 prompt 片段文本
        """
        intents = self.get_enabled_intents()
        if not intents:
            return ""

        lines = ["## 可识别的意图列表\n"]

        for idx, intent in enumerate(intents, 1):
            lines.append(f"### {idx}. {intent.name}")
            lines.append(f"描述: {intent.description}")

            if intent.examples:
                lines.append("示例用户输入:")
                for example in intent.examples:
                    lines.append(f'- "{example}"')

            if intent.extraction_prompt:
                lines.append(f"额外提取要求:\n{intent.extraction_prompt}")

            lines.append("")  # 空行分隔

        return "\n".join(lines)

    def load_global_config(self, yaml_path: str) -> None:
        """加载全局参数配置文件。

        配置文件不存在时使用代码内默认值，不阻塞启动。

        Args:
            yaml_path: YAML 配置文件路径（相对或绝对）
        """
        path = Path(yaml_path)
        if not path.exists():
            logger.warning(
                "意图识别全局配置文件不存在: %s，使用默认值", yaml_path
            )
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            # 兜底文案：优先读新键 idle_response，兼容历史键 unknown_response
            if "idle_response" in config:
                self.idle_response = str(config["idle_response"])
            elif "unknown_response" in config:
                self.idle_response = str(config["unknown_response"])
            if "max_clarification_rounds" in config:
                self.max_clarification_rounds = int(config["max_clarification_rounds"])
            if "disabled_intents" in config:
                self.disabled_intents = list(config["disabled_intents"] or [])

            logger.info(
                "意图识别全局配置加载成功: max_rounds=%d, disabled=%s",
                self.max_clarification_rounds,
                self.disabled_intents,
            )
        except Exception as e:
            logger.error("加载意图识别全局配置失败: %s，使用默认值", str(e))


# 模块级单例
_registry_instance: Optional[IntentRegistry] = None


def get_intent_registry() -> IntentRegistry:
    """获取 IntentRegistry 单例实例。"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = IntentRegistry()
    return _registry_instance
