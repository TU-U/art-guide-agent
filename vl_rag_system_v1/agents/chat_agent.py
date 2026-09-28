from agents.base_agent import BaseAgent
from typing import Any, Dict, Optional

class ChatAgent(BaseAgent):
    """
    深度聊天代理：负责跨轮次的、有深度的知识探讨。
    """
    def __init__(self):
        super().__init__(name="ChatAgent", persona_prompt="你是一个极具洞察力的思想者...")

    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        # TODO: 集成长期记忆与逻辑链条
        return f"【深度对话】关于您的观点，我有一些深度的想法：{query}"
