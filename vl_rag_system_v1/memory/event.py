"""Event memory: model, archive, and extractor."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

import chromadb
from pydantic import BaseModel, Field

from config import Config

from .insight import LLMCaller, default_insight_embedding_fn
from .models import ChatTurn

logger = logging.getLogger("Memory.Event")

DEFAULT_PROMPT_FILE = "event_extraction.md"


class EventEntry(BaseModel):
    """Objective event memory extracted by the LLM."""

    user_id: str = "anonymous"
    session_id: Optional[str] = None
    event: str
    content: str = Field(max_length=50)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_chroma_metadata(self) -> Dict[str, str]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id or "",
            "event": self.event,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_chroma(cls, document: str, metadata: Dict[str, str]) -> "EventEntry":
        ts = metadata.get("timestamp")
        try:
            timestamp = datetime.fromisoformat(ts) if ts else datetime.utcnow()
        except ValueError:
            timestamp = datetime.utcnow()
        return cls(
            user_id=metadata.get("user_id", "anonymous"),
            session_id=metadata.get("session_id") or None,
            event=metadata.get("event", ""),
            content=document,
            timestamp=timestamp,
        )


class EventArchive:
    def __init__(
        self,
        db_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_fn=None,
    ):
        self.db_path = Path(db_path or Config.MEMORY_EVENT_DB)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or Config.MEMORY_EVENT_COLLECTION
        self.embedding_fn = embedding_fn or default_insight_embedding_fn()
        self.fallback_mode = self.embedding_fn is None
        self._lock = threading.Lock()
        self._client = chromadb.PersistentClient(path=str(self.db_path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "LLM extracted conversation events"},
        )

    def _build_event_id(self, entry: EventEntry) -> str:
        base = "|".join(
            [
                entry.user_id or "",
                entry.session_id or "",
                entry.event,
                entry.content,
                entry.timestamp.isoformat(),
                uuid4().hex,
            ]
        )
        return sha1(base.encode("utf-8")).hexdigest()

    def _ensure_embedding(self, entry: EventEntry):
        if self.fallback_mode or self.embedding_fn is None:
            return None
        try:
            return self.embedding_fn([f"{entry.event} {entry.content}".strip()])[0]
        except Exception as exc:
            logger.error("Event embedding failed: %s", exc)
            return None

    def _commit_sync(self, entry: EventEntry) -> bool:
        with self._lock:
            embedding = self._ensure_embedding(entry)
            metadata = entry.to_chroma_metadata()
            event_id = self._build_event_id(entry)
            try:
                kwargs: Dict = {
                    "ids": [event_id],
                    "documents": [entry.content],
                    "metadatas": [metadata],
                }
                if embedding is not None:
                    kwargs["embeddings"] = [embedding]
                self._collection.upsert(**kwargs)
                return True
            except Exception as exc:
                logger.error("Event commit failed: %s", exc)
                return False

    async def commit_event(self, entry: EventEntry) -> bool:
        return await asyncio.to_thread(self._commit_sync, entry)

    def commit_event_sync(self, entry: EventEntry) -> bool:
        return self._commit_sync(entry)

    def _build_where(self, user_id: Optional[str], session_id: Optional[str]) -> Optional[Dict]:
        clauses: List[Dict] = []
        if user_id:
            clauses.append({"user_id": user_id})
        if session_id:
            clauses.append({"session_id": session_id})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def search_events(self, query_vector, top_k: int = 3, user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[EventEntry]:
        where = self._build_where(user_id, session_id)
        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("Event vector search failed: %s", exc)
            return []

        entries: List[EventEntry] = []
        for doc, meta in zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]):
            try:
                entries.append(EventEntry.from_chroma(doc, meta or {}))
            except Exception:
                continue
        return entries

    def search_by_text(
        self,
        query: str,
        top_k: int = 3,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[EventEntry]:
        if self.fallback_mode or self.embedding_fn is None:
            return self._keyword_fallback(query, top_k, user_id, session_id)
        try:
            vector = self.embedding_fn([query])[0]
        except Exception as exc:
            logger.error("Event query encode failed, fallback to keyword: %s", exc)
            return self._keyword_fallback(query, top_k, user_id, session_id)
        return self.search_events(vector, top_k, user_id, session_id)

    def _keyword_fallback(self, query: str, top_k: int, user_id: Optional[str], session_id: Optional[str]) -> List[EventEntry]:
        query_lc = (query or "").lower()
        if not query_lc:
            return []
        where = self._build_where(user_id, session_id)
        try:
            payload = self._collection.get(where=where, include=["documents", "metadatas"])
        except Exception as exc:
            logger.error("Event keyword fallback failed: %s", exc)
            return []

        scored: List = []
        for doc, meta in zip(payload.get("documents", []), payload.get("metadatas", [])):
            haystack = f"{meta.get('event', '')} {doc or ''}".lower()
            score = sum(1 for token in query_lc.split() if token and token in haystack)
            if score > 0:
                scored.append((score, doc, meta or {}))
        scored.sort(key=lambda item: item[0], reverse=True)

        entries: List[EventEntry] = []
        for _, doc, meta in scored[:top_k]:
            try:
                entries.append(EventEntry.from_chroma(doc, meta))
            except Exception:
                continue
        return entries

    def delete_event(self, event_id: str) -> bool:
        with self._lock:
            try:
                self._collection.delete(ids=[event_id])
                return True
            except Exception as exc:
                logger.error("Event delete failed: %s", exc)
                return False

    def list_user_events(self, user_id: str, limit: int = 100) -> List[EventEntry]:
        try:
            payload = self._collection.get(
                where={"user_id": user_id},
                include=["documents", "metadatas"],
                limit=limit,
            )
        except Exception as exc:
            logger.error("List user events failed: %s", exc)
            return []

        entries: List[EventEntry] = []
        for doc, meta in zip(payload.get("documents", []), payload.get("metadatas", [])):
            try:
                entries.append(EventEntry.from_chroma(doc, meta or {}))
            except Exception:
                continue
        return entries

    def get_stats(self) -> Dict:
        try:
            return {
                "total_events": self._collection.count(),
                "db_path": str(self.db_path),
                "collection": self.collection_name,
                "fallback_mode": self.fallback_mode,
            }
        except Exception:
            return {
                "total_events": 0,
                "db_path": str(self.db_path),
                "collection": self.collection_name,
                "fallback_mode": self.fallback_mode,
                "status": "error",
            }


class EventExtractor:
    def __init__(
        self,
        llm_caller: LLMCaller,
        prompt_path: Optional[Path] = None,
        max_turns_per_extract: int = 6,
    ):
        self.llm_caller = llm_caller
        self.prompt_path = Path(prompt_path or (Config.PROMPTS_DIR / DEFAULT_PROMPT_FILE))
        self.max_turns_per_extract = max_turns_per_extract
        self._template_cache: Optional[str] = None

    def _load_template(self) -> str:
        if self._template_cache is not None:
            return self._template_cache
        try:
            self._template_cache = self.prompt_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("[EVENT_EXTRACTOR] load prompt failed %s: %s", self.prompt_path, exc)
            self._template_cache = "请输出事件 JSON 数组。对话:\n{conversation}\n话题:{topic_subject}"
        return self._template_cache

    def _format_conversation(self, turns: Iterable[ChatTurn]) -> str:
        lines: List[str] = []
        for turn in list(turns)[-self.max_turns_per_extract:]:
            role = "用户" if turn.role == "user" else "机器人"
            lines.append(f"{role}: {turn.content}")
        return "\n".join(lines)

    def _parse_llm_output(self, raw: str) -> List[dict]:
        if not raw:
            return []
        text = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            array_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
            if not array_match:
                return []
            try:
                parsed = json.loads(array_match.group(0))
            except json.JSONDecodeError:
                return []
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]

    def _build_messages(self, conversation_text: str, topic_subject: str) -> List[dict]:
        template = self._load_template()
        user_prompt = template.format(
            conversation=conversation_text or "（空）",
            topic_subject=topic_subject or "无",
        )
        return [
            {"role": "system", "content": "你是专业、客观的事件记录助手。"},
            {"role": "user", "content": user_prompt},
        ]

    def extract(
        self,
        turns: List[ChatTurn],
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        topic_subject: str = "",
    ) -> List[EventEntry]:
        if not turns:
            return []
        conversation_text = self._format_conversation(turns)
        messages = self._build_messages(conversation_text, topic_subject)
        try:
            raw = self.llm_caller(messages)
        except Exception as exc:
            logger.error("[EVENT_EXTRACTOR] llm call failed: %s", exc)
            return []

        entries: List[EventEntry] = []
        for item in self._parse_llm_output(raw)[:1]:
            event = str(item.get("event") or "").strip()[:10]
            content = str(item.get("content") or "").strip()[:50]
            if not event or not content:
                continue
            entries.append(
                EventEntry(
                    user_id=user_id,
                    session_id=session_id,
                    event=event,
                    content=content,
                )
            )
        return entries
