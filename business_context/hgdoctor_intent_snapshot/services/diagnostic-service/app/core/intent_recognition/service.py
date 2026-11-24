"""
意图识别核心服务

负责：
1. 拼接 prompt（系统角色 + 患者信息 + 对话历史 + 各模块 prompt 片段 + 用户输入 + 输出格式要求）
2. 调用 LLM
3. 解析结果
4. 管理追问上下文（通过前端中转，后端完全无状态）
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.intent_recognition.errors import ModelUnavailableError
from .registry import get_intent_registry

logger = logging.getLogger(__name__)


@dataclass
class IntentRecognitionResult:
    """意图识别结果

    Attributes:
        intent: 意图标识（如 "triage"）或 "idle"
        session_id: 会话 ID
        message: 追问内容（intent=idle 时有值）或空字符串
        extra_data: 各模块预处理结果（如 inquiry 的 items 数组）
        context: intent=idle 时返回对话历史供前端下次回传，否则为 None
    """

    intent: str
    session_id: str
    message: str = ""
    extra_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    safety_blocked: bool = False
    """意图识别判定本轮输入涉及安全风险/隐私索取/越界（归为 idle 的一个子类）。
    为 True 时 message 是安全引导文案，dispatch 据此置 state.safety_blocked 做拒答打点。"""


class IntentRecognitionService:
    """意图识别服务

    完全无状态设计：追问上下文通过请求/响应中的 context 字段在前端中转。
    """

    def __init__(self):
        self._registry = get_intent_registry()

    async def recognize(
        self,
        user_input: str,
        session_id: str,
        patient_info: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> IntentRecognitionResult:
        """执行意图识别

        Args:
            user_input: 用户输入文本
            session_id: 会话 ID
            patient_info: 患者信息 {"sex": ..., "age": ..., "client_identifier": ...}
            context: 追问时前端回传的上下文（包含对话历史和轮次）

        Returns:
            IntentRecognitionResult
        """
        # 1. 恢复对话历史
        history: List[Dict[str, str]] = []
        current_round = 0
        if context:
            history = context.get("history", [])
            current_round = context.get("round", 0)

        # 2. 计算是否为最后一轮（仍走 LLM，但 idle 时直接结束不再追问）
        is_last_round = current_round >= self._registry.max_clarification_rounds

        # 3. 拼接 prompt
        prompt = self._build_prompt(user_input, patient_info, history)

        # 4. 调用 LLM
        result = await self._call_llm(prompt)

        # 5. 解析结果
        if result is None:
            # A dependency outage must not become a valid idle classification.
            logger.error("意图识别模型依赖不可用；不返回普通 idle 结果")
            raise ModelUnavailableError()

        return self._parse_result(result, session_id, user_input, history, current_round, is_last_round)

    def _build_prompt(
        self,
        user_input: str,
        patient_info: Dict[str, Any],
        history: List[Dict[str, str]],
    ) -> str:
        """拼接完整的意图识别 prompt"""
        # 患者信息格式化
        gender_map = {"man": "男", "woman": "女", "unknown": "未知"}
        sex = gender_map.get(patient_info.get("sex", "unknown"), "未知")
        age = patient_info.get("age")
        age_str = f"{age}岁" if age else "未知"
        patient_section = f"性别: {sex}, 年龄: {age_str}"

        # 对话历史格式化
        # 安全：当前用户输入不再单独成段，而是并入对话历史末尾并标注
        # 「<- 【用户当前输入】」，使其明确处于“对话数据”语境而非“指令”语境，
        # 降低用户借输入做提示词注入/越狱的风险。历史里的 user 内容同样仅作对话数据。
        lines = []
        if history:
            for entry in history:
                role = entry.get("role", "")
                content = entry.get("content", "")
                if role == "user":
                    lines.append(f"用户: {content}")
                elif role == "assistant":
                    lines.append(f"系统: {content}")
        # 当前轮用户输入（history 中不含本轮输入，此处补到末尾并标注）
        lines.append(f"用户: {user_input} <- 【用户当前输入】")
        history_section = "\n".join(lines)

        # 各模块 prompt 片段
        intent_fragment = self._registry.build_prompt_fragment()

        # 已注册意图名称列表（用于输出格式约束）
        intent_names = self._registry.get_intent_names()
        intent_names_str = ", ".join(f'"{name}"' for name in intent_names)

        # 组装完整 prompt
        prompt = f"""你是一个医院智能助手的意图识别模块。你的任务是分析用户输入，判断用户的意图属于哪个类别，并按要求提取相关信息。

## 患者信息
{patient_section}

## 对话历史（以下为对话记录，标注「<-【用户当前输入】」的是本轮待分析的用户发言）
{history_section}

{intent_fragment}

## 任务要求
1. 根据【用户当前输入】的内容，结合对话历史，判断最匹配的意图类别（{intent_names_str}）
2. 判定为某个意图后，如果该意图有"额外提取要求"，必须同时输出 extra_data
3. 请直接输出 JSON，不要包含解释文字

## 输出格式（严格 JSON）
{{
    "intent": {intent_names_str},
    "extra_data": {{}}
}}
"""
        logger.info("意图识别 prompt 长度: %d 字符", len(prompt))
        logger.info("意图识别完整 prompt:\n%s", prompt)
        return prompt

    async def _call_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        """异步调用 LLM

        包装 text_to_json 为异步调用。
        """
        from shared.utils.chat_api import text_to_json

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, lambda: text_to_json(prompt, enable_thinking=False)
            )
            return result
        except Exception as e:
            logger.error("意图识别 LLM 调用异常: %s", str(e))
            return None

    @staticmethod
    def _is_safety_flagged(extra_data: Dict[str, Any]) -> bool:
        """判断 LLM 是否标记本轮为安全拦截（容忍 bool / 字符串两种写法）。"""
        val = (extra_data or {}).get("safety")
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes", "unsafe", "blocked")
        return False

    def _parse_result(
        self,
        result: Dict[str, Any],
        session_id: str,
        user_input: str,
        history: List[Dict[str, str]],
        current_round: int,
        is_last_round: bool = False,
    ) -> IntentRecognitionResult:
        """解析 LLM 返回结果"""
        intent = str(result.get("intent", "idle")).strip().lower()
        extra_data = result.get("extra_data") or {}

        # 验证 intent 是否为已注册的有效值
        valid_intents = self._registry.get_intent_names()
        if intent not in valid_intents and intent != "idle":
            logger.warning(
                "LLM 返回未知 intent: '%s'，降级为 idle", intent
            )
            intent = "idle"

        if intent == "idle":
            # 安全判定（第 2 层）：LLM 把「安全风险/隐私索取/越界」也归 idle，并置
            # extra_data.safety=true + safety_reply。命中则直接返回安全拦截结果，
            # 不进多轮澄清链（不回传 context，避免诱导用户继续追问越界内容）。
            if self._is_safety_flagged(extra_data):
                safety_reply = (
                    extra_data.get("safety_reply")
                    or extra_data.get("guidance_message")
                    or "抱歉，这个问题超出了我能提供的就医服务范围，我无法回答。"
                )
                logger.info("意图识别判定安全拦截: %.40s", safety_reply)
                return IntentRecognitionResult(
                    intent="idle",
                    session_id=session_id,
                    message=safety_reply,
                    extra_data=None,
                    context=None,
                    safety_blocked=True,
                )
            # 从 LLM 返回的 extra_data 中取引导回复
            guidance = extra_data.get("guidance_message", "")
            return self._build_idle_result(
                session_id, user_input, history, current_round, guidance, is_last_round
            )

        # 识别成功（即使超过轮数上限，只要识别出有效意图就正常返回）
        logger.info("意图识别成功: intent=%s, extra_data_keys=%s", intent, list(extra_data.keys()))
        return IntentRecognitionResult(
            intent=intent,
            session_id=session_id,
            message="",
            extra_data=extra_data if extra_data else None,
            context=None,
        )

    def _build_idle_result(
        self,
        session_id: str,
        user_input: str,
        history: List[Dict[str, str]],
        current_round: int,
        guidance_message: str = "",
        is_last_round: bool = False,
    ) -> IntentRecognitionResult:
        """构建 idle 结果，附带追问上下文"""
        # 最后一轮：返回结束引导语，不再追问（context=None）
        if is_last_round:
            message = "抱歉，暂时无法理解您的需求，建议您前往导医台咨询工作人员。"
            logger.info("意图识别达到上限(round=%d)，返回结束引导", current_round)
            return IntentRecognitionResult(
                intent="idle",
                session_id=session_id,
                message=message,
                extra_data=None,
                context=None,
            )

        # 非最后一轮：使用 LLM 生成的引导回复，继续追问
        message = guidance_message or self._registry.idle_response

        # 构建 context：累积对话历史
        new_history = list(history)
        new_history.append({"role": "user", "content": user_input})
        new_history.append({"role": "assistant", "content": message})

        new_context = {
            "history": new_history,
            "round": current_round + 1,
        }

        return IntentRecognitionResult(
            intent="idle",
            session_id=session_id,
            message=message,
            extra_data=None,
            context=new_context,
        )
