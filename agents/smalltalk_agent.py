from agents.base_agent import BaseAgent
from typing import Any, Dict, Optional

class SmallTalkAgent(BaseAgent):
    """
    闲聊代理：负责情感共鸣和个性化对话。
    """
    def __init__(self):
        super().__init__(name="SmallTalkAgent", persona_prompt="你是一个温和体贴的机器人，性格沉静富有美学感...")

    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        # TODO: 集成共鸣引擎与人格化情感分析
        return f"【温暖闲聊】看到您对这个感兴趣，我也很开心呢：{query}"
