"""Insight memory: model, archive, and extractor."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional
from uuid import uuid4

import chromadb
from pydantic import BaseModel, Field

from config import Config

from .models import ChatTurn

logger = logging.getLogger("Memory.Insight")

LLMCaller = Callable[[List[dict]], str]
EmbeddingFn = Callable[[List[str]], List[List[float]]]
DEFAULT_PROMPT_FILE = "insight_extraction.md"


class InsightEntry(BaseModel):
    """Structured user insight extracted by the LLM."""

    insight_id: str = Field(default_factory=lambda: uuid4().hex)
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    topic: str = ""
    content: str
    key_entities: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_chroma_metadata(self) -> Dict[str, str]:
        return {
            "insight_id": self.insight_id,
            "user_id": self.user_id,
            "session_id": self.session_id or "",
            "topic": self.topic,
            "key_entities": ",".join(self.key_entities),
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_chroma(cls, document: str, metadata: Dict[str, str]) -> "InsightEntry":
        raw_entities = metadata.get("key_entities", "")
        entities = [item for item in raw_entities.split(",") if item]
        ts = metadata.get("timestamp")
        try:
            timestamp = datetime.fromisoformat(ts) if ts else datetime.utcnow()
        except ValueError:
            timestamp = datetime.utcnow()
        return cls(
            insight_id=metadata.get("insight_id") or uuid4().hex,
            user_id=metadata.get("user_id", "anonymous"),
            session_id=metadata.get("session_id") or None,
            topic=metadata.get("topic", ""),
            content=document,
            key_entities=entities,
            timestamp=timestamp,
        )


def default_insight_embedding_fn() -> Optional[EmbeddingFn]:
    """Reuse the same local Qwen embedding family as the main RAG path."""
    try:
        from rag.retriever_v2_mix_Reranking import QwenEmbeddingFunction

        project_root = Path(__file__).resolve().parent.parent
        model_candidates = [
            project_root / "models" / "qwen3-embedding",
            project_root / "models" / "Qwen3-Embedding-0.6B",
        ]
        model_path = next((path for path in model_candidates if path.exists()), None)
        if model_path is None:
            logger.warning("Local Qwen embedding model not found for insight archive")
            return None

        embedder = QwenEmbeddingFunction(model_path)

        def _encode(texts: List[str]) -> List[List[float]]:
            return embedder.embed_documents(texts)

        return _encode
    except Exception as exc:
        logger.warning("Insight archive embedding init failed: %s", exc)
        return None


class InsightArchive:
    def __init__(
        self,
        db_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_fn: Optional[EmbeddingFn] = None,
    ):
        self.db_path = Path(db_path or Config.MEMORY_INSIGHT_DB)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or Config.MEMORY_INSIGHT_COLLECTION

        self.embedding_fn = embedding_fn or default_insight_embedding_fn()
        self.fallback_mode = self.embedding_fn is None

        self._lock = threading.Lock()
        self._client = chromadb.PersistentClient(path=str(self.db_path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "LLM extracted user insights"},
        )

    def _ensure_embedding(self, entry: InsightEntry) -> Optional[List[float]]:
        if self.fallback_mode or self.embedding_fn is None:
            return None
        try:
            return self.embedding_fn([entry.content])[0]
        except Exception as exc:
            logger.error("Insight embedding failed: %s", exc)
            return None

    def _commit_sync(self, entry: InsightEntry) -> bool:
        with self._lock:
            embedding = self._ensure_embedding(entry)
            metadata = entry.to_chroma_metadata()
            try:
                kwargs: Dict = {
                    "ids": [entry.insight_id],
                    "documents": [entry.content],
                    "metadatas": [metadata],
                }
                if embedding is not None:
                    kwargs["embeddings"] = [embedding]
                self._collection.upsert(**kwargs)
                return True
            except Exception as exc:
                logger.error("Insight commit failed: %s", exc)
                return False

    async def commit_insight(self, entry: InsightEntry) -> bool:
        return await asyncio.to_thread(self._commit_sync, entry)

    def commit_insight_sync(self, entry: InsightEntry) -> bool:
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

    def search_insights(
        self,
        query_vector: List[float],
        top_k: int = 3,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[InsightEntry]:
        where = self._build_where(user_id, session_id)
        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("Insight vector search failed: %s", exc)
            return []

        entries: List[InsightEntry] = []
        for doc, meta in zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]):
            try:
                entries.append(InsightEntry.from_chroma(doc, meta or {}))
            except Exception:
                continue
        return entries

    def search_by_text(
        self,
        query: str,
        top_k: int = 3,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[InsightEntry]:
        if self.fallback_mode or self.embedding_fn is None:
            return self._keyword_fallback(query, top_k, user_id, session_id)
        try:
            vector = self.embedding_fn([query])[0]
        except Exception as exc:
            logger.error("Insight query encode failed, fallback to keyword: %s", exc)
            return self._keyword_fallback(query, top_k, user_id, session_id)
        return self.search_insights(vector, top_k, user_id, session_id)

    def _keyword_fallback(
        self,
        query: str,
        top_k: int,
        user_id: Optional[str],
        session_id: Optional[str],
    ) -> List[InsightEntry]:
        query_lc = (query or "").lower()
        if not query_lc:
            return []
        where = self._build_where(user_id, session_id)
        try:
            payload = self._collection.get(where=where, include=["documents", "metadatas"])
        except Exception as exc:
            logger.error("Insight keyword fallback failed: %s", exc)
            return []

        scored: List = []
        for doc, meta in zip(payload.get("documents", []), payload.get("metadatas", [])):
            text = (doc or "").lower()
            score = sum(1 for token in query_lc.split() if token and token in text)
            if score > 0:
                scored.append((score, doc, meta or {}))
        scored.sort(key=lambda item: item[0], reverse=True)

        entries: List[InsightEntry] = []
        for _, doc, meta in scored[:top_k]:
            try:
                entries.append(InsightEntry.from_chroma(doc, meta))
            except Exception:
                continue
        return entries

    def delete_insight(self, insight_id: str) -> bool:
        with self._lock:
            try:
                self._collection.delete(ids=[insight_id])
                return True
            except Exception as exc:
                logger.error("Insight delete failed: %s", exc)
                return False

    def list_user_insights(self, user_id: str, limit: int = 100) -> List[InsightEntry]:
        try:
            payload = self._collection.get(
                where={"user_id": user_id},
                include=["documents", "metadatas"],
                limit=limit,
            )
        except Exception as exc:
            logger.error("List user insights failed: %s", exc)
            return []

        entries: List[InsightEntry] = []
        for doc, meta in zip(payload.get("documents", []), payload.get("metadatas", [])):
            try:
                entries.append(InsightEntry.from_chroma(doc, meta or {}))
            except Exception:
                continue
        return entries

    def get_stats(self) -> Dict:
        try:
            return {
                "total_insights": self._collection.count(),
                "db_path": str(self.db_path),
                "collection": self.collection_name,
                "fallback_mode": self.fallback_mode,
            }
        except Exception:
            return {
                "total_insights": 0,
                "db_path": str(self.db_path),
                "collection": self.collection_name,
                "fallback_mode": self.fallback_mode,
                "status": "error",
            }


def _preview(text: str, limit: int = 240) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


class InsightExtractor:
    def __init__(
        self,
        llm_caller: LLMCaller,
        prompt_path: Optional[Path] = None,
        max_turns_per_extract: int = 10,
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
            logger.error("[INSIGHT_EXTRACTOR] load prompt failed %s: %s", self.prompt_path, exc)
            self._template_cache = (
                "请从对话里提炼用户见解，以 JSON 数组返回，字段包含 topic/content/key_entities。\n"
                "对话:\n---\n{conversation}\n---\n已知话题:{topic_subject}"
            )
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
                logger.warning("[INSIGHT_EXTRACTOR] cannot parse llm output: %s", _preview(text, 200))
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
            {"role": "system", "content": "你是沉稳专业的对话内省助手。"},
            {"role": "user", "content": user_prompt},
        ]

    def extract(
        self,
        turns: List[ChatTurn],
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        topic_subject: str = "",
    ) -> List[InsightEntry]:
        if not turns:
            return []

        conversation_text = self._format_conversation(turns)
        messages = self._build_messages(conversation_text, topic_subject)

        try:
            raw = self.llm_caller(messages)
        except Exception as exc:
            logger.error("[INSIGHT_EXTRACTOR] llm call failed: %s", exc)
            return []

        entries: List[InsightEntry] = []
        for item in self._parse_llm_output(raw):
            content = (item.get("content") or "").strip()
            if not content:
                continue
            entries.append(
                InsightEntry(
                    user_id=user_id,
                    session_id=session_id,
                    topic=(item.get("topic") or "").strip()[:24],
                    content=content,
                    key_entities=[str(tag).strip() for tag in (item.get("key_entities") or []) if str(tag).strip()][:5],
                )
            )
        return entries
