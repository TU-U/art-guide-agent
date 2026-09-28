"""Persistent summaries created only when a conversation session is ended."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from config import Config
from .models import ChatTurn


class SessionSummary(BaseModel):
    session_id: str
    user_id: str = "anonymous"
    content: str
    topic_subject: str = ""
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SessionSummaryArchive:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir or Config.MEMORY_SESSION_SUMMARIES_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def commit(self, summary: SessionSummary) -> None:
        path = self.storage_dir / f"{self._safe_name(summary.session_id)}.json"
        if hasattr(summary, "model_dump_json"):
            payload = summary.model_dump_json(indent=2)
        else:
            payload = summary.json(ensure_ascii=False, indent=2)
        path.write_text(payload, encoding="utf-8")

    def get(self, session_id: str) -> Optional[SessionSummary]:
        path = self.storage_dir / f"{self._safe_name(session_id)}.json"
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            if hasattr(SessionSummary, "model_validate_json"):
                return SessionSummary.model_validate_json(raw)
            return SessionSummary.parse_raw(raw)
        except Exception:
            return None

    @staticmethod
    def _safe_name(session_id: str) -> str:
        return session_id.replace("/", "_").replace("\\", "_") or "default"


class SessionSummarizer:
    def __init__(self, llm_caller: Callable[[List[Dict[str, str]]], str]):
        self.llm_caller = llm_caller

    def summarize(self, turns: List[ChatTurn], session_id: str, user_id: str, topic_subject: str = "") -> SessionSummary:
        transcript = "\n".join(f"{turn.role}: {turn.content}" for turn in turns)
        messages = [
            {"role": "system", "content": "你负责整理一次已结束的对话。只输出一段 80 到 220 字的中文摘要，包含已确认事实、用户偏好、任务结果和未解决事项；忽略寒暄，不编造。"},
            {"role": "user", "content": f"主题：{topic_subject or '未指定'}\n对话：\n{transcript}"},
        ]
        content = self.llm_caller(messages).strip()
        return SessionSummary(session_id=session_id, user_id=user_id, content=content[:800], topic_subject=topic_subject)
