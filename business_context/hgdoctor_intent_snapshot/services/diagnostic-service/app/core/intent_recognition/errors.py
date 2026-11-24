"""Failures of the model dependency are not patient intent classifications."""


class ModelUnavailableError(RuntimeError):
    def __init__(self):
        super().__init__("模型服务暂时不可用，请稍后重试。")
