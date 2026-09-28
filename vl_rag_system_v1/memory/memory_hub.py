"""Memory hub: single public entry for chat memory flows."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from config import Config

from .event import EventArchive, EventEntry, EventExtractor
from .insight import InsightArchive, InsightEntry, InsightExtractor
from .lab_fact import LabFactArchive, LabFactEntry, LabFactExtractor
from .models import UserGroupProfile
from .short_term_memory import ShortTermMemory
from .user_group_profiles import UserGroupProfiles

logger = logging.getLogger("Memory.Hub")


def _model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@dataclass
class RecallResult:
    raw_history: List[Dict[str, str]] = field(default_factory=list)
    lab_facts: List[LabFactEntry] = field(default_factory=list)
    insights: List[InsightEntry] = field(default_factory=list)
    events: List[EventEntry] = field(default_factory=list)
    user_group: Optional[UserGroupProfile] = None
    combined_context: str = ""

    def to_dict(self) -> Dict:
        return {
            "raw_history": list(self.raw_history),
            "lab_facts": [_model_to_dict(fact) for fact in self.lab_facts],
            "insights": [_model_to_dict(insight) for insight in self.insights],
            "events": [_model_to_dict(event) for event in self.events],
            "user_group": _model_to_dict(self.user_group) if self.user_group else None,
            "combined_context": self.combined_context,
        }


class MemoryHub:
    def __init__(
        self,
        short_term: Optional[ShortTermMemory] = None,
        insight_archive: Optional[InsightArchive] = None,
        event_archive: Optional[EventArchive] = None,
        lab_fact_archive: Optional[LabFactArchive] = None,
        user_groups: Optional[UserGroupProfiles] = None,
        extractor: Optional[InsightExtractor] = None,
        event_extractor: Optional[EventExtractor] = None,
        lab_fact_extractor: Optional[LabFactExtractor] = None,
    ):
        self.short_term = short_term or ShortTermMemory()
        self.insights = insight_archive or InsightArchive()
        self.events = event_archive or EventArchive()
        self.lab_facts = lab_fact_archive or LabFactArchive()
        self.user_groups = user_groups or UserGroupProfiles()
        self.extractor: Optional[InsightExtractor] = extractor
        self.event_extractor: Optional[EventExtractor] = event_extractor
        self.lab_fact_extractor = lab_fact_extractor or LabFactExtractor()
        self._reflection_state_path = Path(Config.MEMORY_DIR) / "reflection_state.json"
        self._reflection_state = self._load_reflection_state()
        logger.info("Memory hub ready")

    def _load_reflection_state(self) -> Dict[str, int]:
        try:
            if self._reflection_state_path.exists():
                raw = json.loads(self._reflection_state_path.read_text(encoding="utf-8"))
                return {str(k): int(v) for k, v in raw.items()}
        except Exception as exc:
            logger.warning("Failed to load reflection state: %s", exc)
        return {}

    def _save_reflection_state(self) -> None:
        try:
            self._reflection_state_path.write_text(
                json.dumps(self._reflection_state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save reflection state: %s", exc)

    def attach_extractor(self, extractor_or_caller, **extractor_kwargs) -> None:
        if isinstance(extractor_or_caller, InsightExtractor):
            self.extractor = extractor_or_caller
        else:
            self.extractor = InsightExtractor(llm_caller=extractor_or_caller, **extractor_kwargs)
            self.event_extractor = EventExtractor(llm_caller=extractor_or_caller)
        logger.info("Memory extractors attached")

    def record_turn(self, session_id: str, role: str, content: str) -> None:
        self.short_term.add_chat_history(session_id, role, content)

    def observe_user_fact(
        self,
        session_id: str,
        user_text: str,
        user_id: str = "anonymous",
    ) -> List[LabFactEntry]:
        entries = self.lab_fact_extractor.extract(user_text, user_id=user_id, session_id=session_id)
        committed: List[LabFactEntry] = []
        for entry in entries:
            if self.lab_facts.commit_fact_sync(entry):
                committed.append(entry)
        return committed

    def commit_visual_summary(
        self,
        session_id: str,
        summary: Dict,
        user_id: str = "anonymous",
    ) -> Optional[LabFactEntry]:
        main_content = str(summary.get("main_content") or summary.get("scene_summary") or "").strip()
        if not main_content:
            return None

        target = str(summary.get("target_object") or summary.get("target_class") or "当前画面").strip()
        image_path = str(summary.get("image_path") or "").strip()
        timestamp = str(summary.get("timestamp") or "").strip()
        content_parts = [f"视觉摘要：{main_content}"]
        if target:
            content_parts.append(f"主要对象：{target}")
        if image_path:
            content_parts.append(f"图片：{image_path}")
        if timestamp:
            content_parts.append(f"时间：{timestamp}")

        entry = LabFactEntry(
            fact_id=f"visual_{uuid4().hex}",
            user_id=user_id or "anonymous",
            session_id=session_id,
            category="visual_summary",
            subject=target[:24],
            predicate="视觉观察",
            object=main_content[:40],
            content="；".join(content_parts),
            source="vlm_visual_summary",
            confidence=float(summary.get("confidence") or 0.8),
        )
        if self.lab_facts.commit_fact_sync(entry):
            return entry
        return None

    def clear_session(self, session_id: str) -> bool:
        self._reflection_state.pop(session_id, None)
        self._save_reflection_state()
        return self.short_term.clear_chat_history(session_id)

    def recall(
        self,
        query: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        top_k: int = 3,
        user_features: Optional[Dict] = None,
        history_tail: int = 6,
    ) -> RecallResult:
        raw_history: List[Dict[str, str]] = []
        if session_id:
            raw_history = self.short_term.get_raw_history(session_id)[-history_tail:]

        filter_user_id = user_id if user_id and user_id != "anonymous" else None
        recall_t0 = time.time()
        query_vector = None
        embedding_fn = (
            self.lab_facts.embedding_fn
            or self.insights.embedding_fn
            or self.events.embedding_fn
        )
        if embedding_fn is not None:
            try:
                query_vector = embedding_fn([query])[0]
            except Exception as exc:
                logger.error("Memory recall query encode failed, fallback to keyword: %s", exc)
        recall_t1 = time.time()

        if query_vector is not None:
            lab_facts = self.lab_facts.search_facts(query_vector, top_k, filter_user_id)
            recall_t2 = time.time()
            insights = self.insights.search_insights(query_vector, top_k, filter_user_id)
            recall_t3 = time.time()
            events = self.events.search_events(query_vector, top_k, filter_user_id)
            recall_t4 = time.time()
        else:
            lab_facts = self.lab_facts.search_by_text(query=query, top_k=top_k, user_id=filter_user_id)
            recall_t2 = time.time()
            insights = self.insights.search_by_text(query=query, top_k=top_k, user_id=filter_user_id)
            recall_t3 = time.time()
            events = self.events.search_by_text(query=query, top_k=top_k, user_id=filter_user_id)
            recall_t4 = time.time()

        logger.info(
            "[MEMORY_RECALL] encode=%.3fs lab=%.3fs insight=%.3fs event=%.3fs total=%.3fs vector=%s",
            recall_t1 - recall_t0,
            recall_t2 - recall_t1,
            recall_t3 - recall_t2,
            recall_t4 - recall_t3,
            recall_t4 - recall_t0,
            bool(query_vector),
        )

        if user_features:
            group_id = self.user_groups.match_group(user_features)
            group = self.user_groups.get_group_config(group_id)
        else:
            group = self.user_groups.get_group_config("general_public")

        return RecallResult(
            raw_history=raw_history,
            lab_facts=lab_facts,
            insights=insights,
            events=events,
            user_group=group,
            combined_context=self._compose_context(raw_history, lab_facts, insights, events, group),
        )

    def _compose_context(
        self,
        history: List[Dict[str, str]],
        lab_facts: List[LabFactEntry],
        insights: List[InsightEntry],
        events: List[EventEntry],
        group: Optional[UserGroupProfile],
    ) -> str:
        blocks: List[str] = []
        if group:
            blocks.append(
                f"[用户群体] {group.category_name} / 审美:{group.aesthetic_pref} / 沟通:{group.communication_pref}"
            )
        if lab_facts:
            lines = [f"- ({fact.category or '事实'}) {fact.content}" for fact in lab_facts]
            blocks.append("[实验室事实记忆]\n" + "\n".join(lines))
        if insights:
            lines = [f"- ({insight.topic or '见解'}) {insight.content}" for insight in insights]
            blocks.append("[过往见解]\n" + "\n".join(lines))
        if events:
            lines = [f"- ({event.event or '事件'}) {event.content}" for event in events]
            blocks.append("[过往事件]\n" + "\n".join(lines))
        if history:
            lines = [f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in history]
            blocks.append("[近期对话]\n" + "\n".join(lines))
        return "\n\n".join(blocks)

    def _pending_turns(self, session_id: str):
        turns = self.short_term.get_turns(session_id)
        start_index = self._reflection_state.get(session_id, 0)
        return turns, turns[start_index:]

    def _commit_reflection_sync(
        self,
        session_id: str,
        user_id: str,
        topic_subject: str,
    ) -> Dict[str, List]:
        turns, pending_turns = self._pending_turns(session_id)
        if len(pending_turns) < 2:
            return {"insights": [], "events": []}

        committed_insights: List[InsightEntry] = []
        committed_events: List[EventEntry] = []

        if self.extractor is not None:
            for entry in self.extractor.extract(pending_turns, user_id, session_id, topic_subject):
                if self.insights.commit_insight_sync(entry):
                    committed_insights.append(entry)

        if self.event_extractor is not None:
            for entry in self.event_extractor.extract(pending_turns, user_id, session_id, topic_subject):
                if self.events.commit_event_sync(entry):
                    committed_events.append(entry)

        self._reflection_state[session_id] = len(turns)
        self._save_reflection_state()
        return {"insights": committed_insights, "events": committed_events}

    async def reflect_on_conversation(
        self,
        session_id: str,
        user_id: str = "anonymous",
        topic_subject: str = "",
    ) -> Dict[str, List]:
        if self.extractor is None and self.event_extractor is None:
            logger.warning("Extractors not attached, skip reflection")
            return {"insights": [], "events": []}
        return await asyncio.to_thread(
            self._commit_reflection_sync,
            session_id,
            user_id,
            topic_subject,
        )

    def reflect_on_conversation_sync(
        self,
        session_id: str,
        user_id: str = "anonymous",
        topic_subject: str = "",
    ) -> Dict[str, List]:
        if self.extractor is None and self.event_extractor is None:
            logger.warning("Extractors not attached, skip reflection")
            return {"insights": [], "events": []}
        return self._commit_reflection_sync(session_id, user_id, topic_subject)

    def sync_persistence(self) -> None:
        self.short_term.sync_persistence()
        self.user_groups.sync_persistence()
        self._save_reflection_state()
        logger.info("Memory persistence synced")

    def get_stats(self) -> Dict:
        return {
            "short_term_sessions": len(self.short_term.list_sessions()),
            "lab_fact_archive": self.lab_facts.get_stats(),
            "insight_archive": self.insights.get_stats(),
            "event_archive": self.events.get_stats(),
            "user_groups": self.user_groups.list_all_groups(),
            "extractor_bound": self.extractor is not None,
            "event_extractor_bound": self.event_extractor is not None,
        }
