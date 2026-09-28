"""Memory hub: single public entry for chat memory flows."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import Config

from .event import EventArchive, EventEntry, EventExtractor
from .insight import InsightArchive, InsightEntry, InsightExtractor
from .lab_fact import LabFactArchive, LabFactEntry, LabFactExtractor
from .models import UserGroupProfile
from .short_term_memory import ShortTermMemory
from .session_summary import SessionSummarizer, SessionSummary, SessionSummaryArchive
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
        session_summaries: Optional[SessionSummaryArchive] = None,
        session_summarizer: Optional[SessionSummarizer] = None,
    ):
        self.short_term = short_term or ShortTermMemory()
        self.insights = insight_archive or InsightArchive()
        self.events = event_archive or EventArchive()
        self.lab_facts = lab_fact_archive or LabFactArchive()
        self.user_groups = user_groups or UserGroupProfiles()
        self.extractor: Optional[InsightExtractor] = extractor
        self.event_extractor: Optional[EventExtractor] = event_extractor
        self.lab_fact_extractor = lab_fact_extractor or LabFactExtractor()
        self.session_summaries = session_summaries or SessionSummaryArchive()
        self.session_summarizer = session_summarizer
        self._consolidation_lock = threading.Lock()
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
            self.session_summarizer = SessionSummarizer(llm_caller=extractor_or_caller)
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
        include_history: bool = True,
        include_long_term: bool = True,
    ) -> RecallResult:
        raw_history: List[Dict[str, str]] = []
        if session_id and include_history:
            raw_history = self.short_term.get_raw_history(session_id)[-history_tail:]

        filter_user_id = user_id if user_id and user_id != "anonymous" else None
        lab_facts = self.lab_facts.search_by_text(query=query, top_k=top_k, user_id=filter_user_id) if include_long_term else []
        insights = self.insights.search_by_text(query=query, top_k=top_k, user_id=filter_user_id) if include_long_term else []
        events = self.events.search_by_text(query=query, top_k=top_k, user_id=filter_user_id) if include_long_term else []

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

    @staticmethod
    def should_recall_long_term(query: str) -> bool:
        markers = ("上次", "之前", "以前", "还记得", "你记得", "历史", "曾经", "我的偏好", "我喜欢")
        return any(marker in (query or "") for marker in markers)

    def end_session_sync(self, session_id: str, user_id: str = "anonymous", topic_subject: str = "") -> Dict:
        # An explicit end request and the idle timer can arrive together.
        # Serialize consolidation so one transcript is never summarized twice.
        with self._consolidation_lock:
            turns, pending_turns = self._pending_turns(session_id)
            if not pending_turns:
                return {"summary": None, "insights": [], "events": [], "skipped": True}
            committed = self._commit_reflection_sync(session_id, user_id, topic_subject)
            summary: Optional[SessionSummary] = None
            if self.session_summarizer is not None and turns:
                summary = self.session_summarizer.summarize(turns, session_id, user_id, topic_subject)
                self.session_summaries.commit(summary)
            return {"summary": summary, **committed, "skipped": False}

    async def end_session(self, session_id: str, user_id: str = "anonymous", topic_subject: str = "") -> Dict:
        return await asyncio.to_thread(self.end_session_sync, session_id, user_id, topic_subject)

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
            "session_summarizer_bound": self.session_summarizer is not None,
        }
