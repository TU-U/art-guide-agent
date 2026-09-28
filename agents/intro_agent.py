from agents.base_agent import BaseAgent
from typing import Any, Dict, Optional

class IntroAgent(BaseAgent):
    """
    展品讲解代理：负责根据 RAG 知识和视觉信息进行专业讲解。
    """
    def __init__(self):
        super().__init__(name="IntroAgent", persona_prompt="你是一个博古通今的博物馆导览员...")

    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        # TODO: 集成 RAG 检索逻辑
        # TODO: 集成视觉分析结果
        return f"【展品讲解】正在为您介绍相关展品：{query}"
