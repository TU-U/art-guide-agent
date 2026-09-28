"""Lab fact memory: explicit user-taught facts for the robot context."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from pydantic import BaseModel, Field

from config import Config

from .insight import EmbeddingFn, default_insight_embedding_fn

logger = logging.getLogger("Memory.LabFact")


def _stable_fact_id(category: str, subject: str, predicate: str, obj: str) -> str:
    raw = "|".join(
        [
            category.strip().lower(),
            subject.strip().lower(),
            predicate.strip().lower(),
            obj.strip().lower(),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


class LabFactEntry(BaseModel):
    """Stable fact explicitly taught by a user."""

    fact_id: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    category: str = "general"
    subject: str = ""
    predicate: str = ""
    object: str = ""
    content: str
    source: str = "user_taught"
    confidence: float = 1.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_chroma_metadata(self) -> Dict:
        return {
            "fact_id": self.fact_id,
            "user_id": self.user_id,
            "session_id": self.session_id or "",
            "category": self.category,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "source": self.source,
            "confidence": float(self.confidence),
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_chroma(cls, document: str, metadata: Dict) -> "LabFactEntry":
        ts = metadata.get("timestamp")
        try:
            timestamp = datetime.fromisoformat(ts) if ts else datetime.utcnow()
        except ValueError:
            timestamp = datetime.utcnow()
        return cls(
            fact_id=metadata.get("fact_id") or _stable_fact_id(
                metadata.get("category", "general"),
                metadata.get("subject", ""),
                metadata.get("predicate", ""),
                metadata.get("object", ""),
            ),
            user_id=metadata.get("user_id", "anonymous"),
            session_id=metadata.get("session_id") or None,
            category=metadata.get("category", "general"),
            subject=metadata.get("subject", ""),
            predicate=metadata.get("predicate", ""),
            object=metadata.get("object", ""),
            content=document,
            source=metadata.get("source", "user_taught"),
            confidence=float(metadata.get("confidence", 1.0) or 1.0),
            timestamp=timestamp,
        )


class LabFactExtractor:
    """Rule-based extractor for explicit, stable facts taught by users."""

    QUESTION_MARKERS = ("?", "？", "吗", "呢", "哪里", "在哪", "什么", "怎么", "为什么")
    TEACH_MARKERS = ("记住", "你要记住", "请记住", "以后", "告诉你")
    STRIP_PREFIXES = (
        "记住",
        "你要记住",
        "请记住",
        "以后要记住",
        "以后请记住",
        "我告诉你",
        "告诉你",
    )

    def extract(
        self,
        text: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
    ) -> List[LabFactEntry]:
        raw = (text or "").strip()
        normalized = _clean_text(raw)
        if not normalized:
            return []

        has_teach_marker = any(marker in normalized for marker in self.TEACH_MARKERS)
        if self._looks_like_question(normalized) and not has_teach_marker:
            return []

        candidate = self._strip_teach_prefix(normalized)
        entries: List[LabFactEntry] = []

        correction = self._extract_correction(candidate, user_id, session_id)
        if correction is not None:
            entries.append(correction)

        for category, predicate, pattern in [
            ("location", "在", r"^(.{1,24}?)在(.{1,24})$"),
            ("alias", "叫", r"^(.{1,24}?)叫(.{1,24})$"),
            ("definition", "是", r"^(.{1,24}?)是(.{1,36})$"),
        ]:
            match = re.match(pattern, candidate)
            if not match:
                continue
            subject, obj = self._strip_fact_part(match.group(1)), self._strip_fact_part(match.group(2))
            if not self._valid_fact_parts(subject, obj):
                continue
            if category == "definition" and self._is_correction_fragment(subject, obj):
                continue
            entries.append(
                self._build_entry(
                    category=category,
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    user_id=user_id,
                    session_id=session_id,
                )
            )

        dedup: Dict[str, LabFactEntry] = {}
        for entry in entries:
            dedup[entry.fact_id] = entry
        return list(dedup.values())

    def _looks_like_question(self, text: str) -> bool:
        return any(marker in text for marker in self.QUESTION_MARKERS)

    def _strip_teach_prefix(self, text: str) -> str:
        result = text
        changed = True
        while changed:
            changed = False
            for prefix in self.STRIP_PREFIXES:
                if result.startswith(prefix):
                    result = result[len(prefix) :]
                    changed = True
        return result.strip("，,。.!！;；：:")

    def _strip_fact_part(self, value: str) -> str:
        return value.strip("，,。.!！;；：: ")

    def _valid_fact_parts(self, subject: str, obj: str) -> bool:
        if not subject or not obj:
            return False
        if len(subject) > 24 or len(obj) > 40:
            return False
        invalid_subjects = ("我", "你", "他", "她", "它", "这个", "那个")
        if subject in invalid_subjects and len(obj) <= 2:
            return False
        return True

    def _is_correction_fragment(self, subject: str, obj: str) -> bool:
        return subject.startswith("不") or subject.endswith("不") or "不是" in subject or "不是" in obj

    def _extract_correction(
        self,
        text: str,
        user_id: str,
        session_id: Optional[str],
    ) -> Optional[LabFactEntry]:
        match = re.search(
            r"^(?:(?P<context>.{1,24}?)(?:里|里的|里面|里面的|中|中的|上|上的|上面|上面的|下|下的|下面|下面的|旁边|附近))?"
            r"(?:其实|实际|实际上|应该)?不是(?P<wrong>.{1,24}?)"
            r"[，, ]*(?:其实|实际|实际上|应该)?是(?P<correct>.{1,24})$",
            text,
        )
        if not match:
            match = re.search(
                r"(?:其实|实际|实际上|应该)?不是(?P<wrong>.{1,24}?)"
                r"[，, ]*(?:其实|实际|实际上|应该)?是(?P<correct>.{1,24})$",
                text,
            )
        if not match:
            return None
        context = self._strip_fact_part(match.groupdict().get("context") or "")
        wrong = self._strip_fact_part(match.group("wrong"))
        correct = self._strip_fact_part(match.group("correct"))
        if not self._valid_fact_parts(wrong, correct):
            return None
        subject = f"{context}里的{wrong}" if context else wrong
        content = (
            f"用户纠正我：{context}里的物品之前识别成{wrong}不对，正确是{correct}。"
            if context
            else f"用户纠正我：之前识别成{wrong}不对，正确是{correct}。"
        )
        return self._build_entry(
            category="correction",
            subject=subject,
            predicate="纠正为",
            obj=correct,
            user_id=user_id,
            session_id=session_id,
            content=content,
        )

    def _build_entry(
        self,
        category: str,
        subject: str,
        predicate: str,
        obj: str,
        user_id: str,
        session_id: Optional[str],
        content: Optional[str] = None,
    ) -> LabFactEntry:
        fact_id = _stable_fact_id(category, subject, predicate, obj)
        fact_content = content or f"用户告诉我：{subject}{predicate}{obj}。"
        return LabFactEntry(
            fact_id=fact_id,
            user_id=user_id or "anonymous",
            session_id=session_id,
            category=category,
            subject=subject,
            predicate=predicate,
            object=obj,
            content=fact_content,
        )


class LabFactArchive:
    def __init__(
        self,
        db_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_fn: Optional[EmbeddingFn] = None,
    ):
        self.db_path = Path(db_path or Config.MEMORY_LAB_FACT_DB)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or Config.MEMORY_LAB_FACT_COLLECTION

        self.embedding_fn = embedding_fn or default_insight_embedding_fn()
        self.fallback_mode = self.embedding_fn is None

        self._lock = threading.Lock()
        self._client = chromadb.PersistentClient(path=str(self.db_path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Explicit lab and school facts taught by users"},
        )

    def _ensure_embedding(self, entry: LabFactEntry) -> Optional[List[float]]:
        if self.fallback_mode or self.embedding_fn is None:
            return None
        try:
            return self.embedding_fn([entry.content])[0]
        except Exception as exc:
            logger.error("Lab fact embedding failed: %s", exc)
            return None

    def _commit_sync(self, entry: LabFactEntry) -> bool:
        with self._lock:
            embedding = self._ensure_embedding(entry)
            metadata = entry.to_chroma_metadata()
            try:
                kwargs: Dict = {
                    "ids": [entry.fact_id],
                    "documents": [entry.content],
                    "metadatas": [metadata],
                }
                if embedding is not None:
                    kwargs["embeddings"] = [embedding]
                self._collection.upsert(**kwargs)
                return True
            except Exception as exc:
                logger.error("Lab fact commit failed: %s", exc)
                return False

    async def commit_fact(self, entry: LabFactEntry) -> bool:
        return await asyncio.to_thread(self._commit_sync, entry)

    def commit_fact_sync(self, entry: LabFactEntry) -> bool:
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

    def search_by_text(
        self,
        query: str,
        top_k: int = 3,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[LabFactEntry]:
        if self.fallback_mode or self.embedding_fn is None:
            return self._keyword_fallback(query, top_k, user_id, session_id)
        try:
            vector = self.embedding_fn([query])[0]
        except Exception as exc:
            logger.error("Lab fact query encode failed, fallback to keyword: %s", exc)
            return self._keyword_fallback(query, top_k, user_id, session_id)
        return self.search_facts(vector, top_k, user_id, session_id)

    def search_facts(
        self,
        query_vector: List[float],
        top_k: int = 3,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[LabFactEntry]:
        where = self._build_where(user_id, session_id)
        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("Lab fact vector search failed: %s", exc)
            return []

        entries: List[LabFactEntry] = []
        for doc, meta in zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]):
            try:
                entries.append(LabFactEntry.from_chroma(doc, meta or {}))
            except Exception:
                continue
        return entries

    def _keyword_fallback(
        self,
        query: str,
        top_k: int,
        user_id: Optional[str],
        session_id: Optional[str],
    ) -> List[LabFactEntry]:
        query_text = _clean_text(query).lower()
        if not query_text:
            return []
        where = self._build_where(user_id, session_id)
        try:
            payload = self._collection.get(where=where, include=["documents", "metadatas"])
        except Exception as exc:
            logger.error("Lab fact keyword fallback failed: %s", exc)
            return []

        scored: List = []
        for doc, meta in zip(payload.get("documents", []), payload.get("metadatas", [])):
            meta = meta or {}
            fields = [
                doc or "",
                meta.get("subject", ""),
                meta.get("predicate", ""),
                meta.get("object", ""),
                meta.get("category", ""),
            ]
            haystack = _clean_text("".join(fields)).lower()
            score = self._keyword_score(query_text, haystack, meta)
            if score > 0:
                scored.append((score, doc, meta))
        scored.sort(key=lambda item: item[0], reverse=True)

        entries: List[LabFactEntry] = []
        for _, doc, meta in scored[:top_k]:
            try:
                entries.append(LabFactEntry.from_chroma(doc, meta))
            except Exception:
                continue
        return entries

    def _keyword_score(self, query: str, haystack: str, metadata: Dict) -> int:
        score = 0
        for field in ("subject", "object"):
            value = _clean_text(str(metadata.get(field, ""))).lower()
            if value and value in query:
                score += 8
            if value and value in haystack and query in value:
                score += 4
        if query in haystack:
            score += 6
        score += len(set(query) & set(haystack))
        return score

    def list_facts(self, limit: int = 100, user_id: Optional[str] = None) -> List[LabFactEntry]:
        where = {"user_id": user_id} if user_id else None
        try:
            payload = self._collection.get(where=where, include=["documents", "metadatas"], limit=limit)
        except Exception as exc:
            logger.error("List lab facts failed: %s", exc)
            return []

        entries: List[LabFactEntry] = []
        for doc, meta in zip(payload.get("documents", []), payload.get("metadatas", [])):
            try:
                entries.append(LabFactEntry.from_chroma(doc, meta or {}))
            except Exception:
                continue
        return entries

    def delete_fact(self, fact_id: str) -> bool:
        with self._lock:
            try:
                self._collection.delete(ids=[fact_id])
                return True
            except Exception as exc:
                logger.error("Lab fact delete failed: %s", exc)
                return False

    def get_stats(self) -> Dict:
        try:
            return {
                "total_facts": self._collection.count(),
                "db_path": str(self.db_path),
                "collection": self.collection_name,
                "fallback_mode": self.fallback_mode,
            }
        except Exception:
            return {
                "total_facts": 0,
                "db_path": str(self.db_path),
                "collection": self.collection_name,
                "fallback_mode": self.fallback_mode,
                "status": "error",
            }
