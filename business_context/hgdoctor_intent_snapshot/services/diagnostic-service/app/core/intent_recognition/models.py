"""
意图定义数据模型
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class IntentDefinition:
    """单个意图的完整定义，由各功能模块注册时提供。

    Attributes:
        name: 意图标识（如 "triage"），全局唯一
        description: 能力描述，直接拼入 LLM prompt 供模型理解该意图覆盖的场景
        examples: 示例 utterances，作为 few-shot 拼入 prompt
        extraction_prompt: 模块特定的预处理提示词。
            当 LLM 判定为该意图时，需按此指令额外输出结构化数据到 extra_data。
            为空字符串时表示无需额外提取。
        priority: 优先级，数字越小越靠前（影响 prompt 中的排列顺序）
    """

    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    extraction_prompt: str = ""
    priority: int = 10
