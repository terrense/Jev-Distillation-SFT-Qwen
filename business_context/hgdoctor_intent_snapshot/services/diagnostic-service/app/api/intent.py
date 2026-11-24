"""意图识别 API 路由模块

提供统一的意图识别接口，前端通过此接口识别用户意图后路由到对应功能模块。
"""

import logging

from app.core.intent_recognition.errors import ModelUnavailableError
import uuid
from typing import Any, Dict, Optional

from app.core.intent_recognition.service import IntentRecognitionService
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# 模块级服务实例（延迟初始化，避免导入时触发 registry 未就绪的问题）
_service: Optional[IntentRecognitionService] = None


def _get_service() -> IntentRecognitionService:
    """获取 IntentRecognitionService 实例（延迟初始化）"""
    global _service
    if _service is None:
        _service = IntentRecognitionService()
    return _service


# =============================================================================
# 请求/响应模型
# =============================================================================


class IntentRecognizeRequest(BaseModel):
    """意图识别请求"""

    user_input: str = Field(..., min_length=1, description="用户输入文本")
    session_id: Optional[str] = Field(None, description="会话 ID（追问场景传入，首次可为空）")
    sex: str = Field(default="unknown", description="患者性别: man/woman/unknown")
    age: Optional[int] = Field(None, description="患者年龄")
    client_identifier: Optional[str] = Field(None, description="客户端标识")
    context: Optional[Dict[str, Any]] = Field(None, description="追问时前端回传的上下文")


class IntentRecognizeResponse(BaseModel):
    """意图识别响应"""

    intent: str = Field(..., description="意图标识: medical/inquiry/unknown")
    session_id: str = Field(..., description="会话 ID")
    message: str = Field(default="", description="追问内容（intent=unknown 时有值）")
    extra_data: Optional[Dict[str, Any]] = Field(None, description="模块预处理结果")
    context: Optional[Dict[str, Any]] = Field(None, description="追问上下文（intent=unknown 时返回，前端下次回传）")


# =============================================================================
# 路由
# =============================================================================


@router.post("/recognize", response_model=IntentRecognizeResponse)
async def recognize_intent(request: IntentRecognizeRequest):
    """识别用户意图

    统一入口接口，分析用户输入并返回意图标识。
    前端根据返回的 intent 路由到对应功能模块的后续接口。

    追问流程：
    1. 首次调用不传 session_id 和 context
    2. 如果返回 intent="unknown"，前端展示 message 追问内容
    3. 用户回答后，前端带上 session_id 和 context 再次调用此接口
    """
    try:
        # 生成或使用传入的 session_id
        session_id = request.session_id or str(uuid.uuid4())
        logger.info("收到意图识别请求: user_input='%s', session_id=%s", request.user_input[:50], session_id)

        # 组装 patient_info
        patient_info = {
            "sex": request.sex,
            "age": request.age,
            "client_identifier": request.client_identifier,
        }

        # 调用意图识别服务
        service = _get_service()
        result = await service.recognize(
            user_input=request.user_input,
            session_id=session_id,
            patient_info=patient_info,
            context=request.context,
        )

        # 构建响应
        response = IntentRecognizeResponse(
            intent=result.intent,
            session_id=result.session_id,
            message=result.message,
            extra_data=result.extra_data,
            context=result.context,
        )

        return JSONResponse(
            content=response.model_dump(),
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

    except ModelUnavailableError as e:
        logger.warning("模型依赖不可用: %s", str(e))
        return JSONResponse(
            status_code=503,
            content={"detail": str(e), "code": "MODEL_UNAVAILABLE"},
            headers={"Retry-After": "10"},
        )
    except Exception as e:
        logger.error("意图识别失败: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"意图识别失败: {str(e)}") from e
