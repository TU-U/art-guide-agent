"""Data models for the memory system."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    """A single short-term conversation turn."""

    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_message(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class UserGroupProfile(BaseModel):
    """Audience group profile."""

    group_id: str
    category_name: str
    aesthetic_pref: str = ""
    communication_pref: str = ""
    typical_tags: List[str] = Field(default_factory=list)
    response_style: Dict[str, str] = Field(default_factory=dict)
