"""
意图识别框架

提供统一的意图识别入口，各功能模块通过注册接口将自己的 intent schema 注入框架。
框架负责收集、拼接 prompt 并调用 LLM 识别用户意图。
"""

from .models import IntentDefinition
from .registry import IntentRegistry, get_intent_registry
from .service import IntentRecognitionResult, IntentRecognitionService

__all__ = [
    "IntentDefinition",
    "IntentRegistry",
    "get_intent_registry",
    "IntentRecognitionResult",
    "IntentRecognitionService",
]
