"""
问诊意图 - 意图注册

当用户已经挂号、或表达要做初步问诊（旧称预问诊）时，识别为 consultation 意图。
跳过导诊，直接进入问诊 FSM。

典型场景：
- 用户已经挂了号，想做初步问诊
- 用户通过导诊推荐了科室后表达要开始问诊
- 用户明确说"我要初步问诊""我要预问诊""我已经挂号了"
"""

from ..models import IntentDefinition
from ..registry import IntentRegistry


def register_consultation_intent(registry: IntentRegistry) -> None:
    """注册问诊意图到 IntentRegistry。

    Args:
        registry: 意图注册表实例
    """
    registry.register(
        IntentDefinition(
            name="initial_consultation",
            description=(
                "用户已经挂号或明确表达要做初步问诊（也可能说旧称'预问诊'），需要跳过导诊直接进入问诊流程。"
                "前提是用户已经有挂号信息或明确说了要初步问诊/预问诊。"
            ),
            examples=[
                "我已经挂好号了，想做初步问诊",
                "我挂了消化内科的号",
                "帮我初步问诊一下",
                "我要做初步问诊",
                "初步问诊",
                "帮我预问诊一下",
                "我要做预问诊",
                "已经挂号了，肚子疼",
                "我挂了号，最近头晕",
                "预问诊",
            ],
            extraction_prompt=(
                "如果识别为 consultation，请从用户输入中提取症状关键词（如果有的话）。\n"
                "输出 extra_data 中包含 extracted_symptoms 数组，每项包含 name 字段。\n"
                "如果用户没有描述症状，extracted_symptoms 可以为空数组。\n"
                '示例1: {"extracted_symptoms": [{"name": "肚子疼"}, {"name": "头晕"}]}\n'
                '示例2: {"extracted_symptoms": []}\n'
                "【重要】只有用户明确表达了'初步问诊'、'预问诊'、'已经挂号'等意愿时才判定为 consultation。\n"
                "如果用户只是描述症状但没提到挂号或初步问诊/预问诊，应判定为 triage。"
            ),
            priority=2,  # 在 triage 之后，inquiry 之前
        )
    )
