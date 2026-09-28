from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseAgent(ABC):
    """
    智能体基类，定义所有 Agent 的通用接口。
    """
    def __init__(self, name: str, persona_prompt: str):
        self.name = name
        self.persona_prompt = persona_prompt

    @abstractmethod
    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        处理用户查询并返回响应。
        """
        pass

    def __str__(self):
        return f"Agent(name={self.name})"
