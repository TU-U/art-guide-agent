"""Event-driven social layer: structured perception facts -> one short reply."""

from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Callable, Dict, Optional, Tuple
from uuid import uuid4

from .event_store import EventStore
from .models import PresenceEvent, WorldState

TextCaller = Callable[[list[Dict[str, str]]], str]
TtsCaller = Callable[[str], Optional[str]]


@dataclass(frozen=True)
class SocialReply:
    message_id: str
    source_event_id: str
    text: str
    timestamp: str
    status: str
    audio_path: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "source_event_id": self.source_event_id,
            "text": self.text,
            "timestamp": self.timestamp,
            "status": self.status,
            "audio_path": self.audio_path,
        }


class SocialAgent:
    """Makes social turns from events, never from raw video frames.

    The camera worker only calls ``submit``. A separate worker calls the LLM,
    preventing network latency from blocking detection and tracking.
    """

    def __init__(
        self,
        store: EventStore,
        text_caller: TextCaller,
        cooldown_seconds: float = 45.0,
        tts_caller: Optional[TtsCaller] = None,
    ):
        self._store = store
        self._text_caller = text_caller
        self._tts_caller = tts_caller
        self._cooldown_seconds = cooldown_seconds
        self._queue: Queue[Tuple[PresenceEvent, WorldState]] = Queue(maxsize=10)
        self._stop = Event()
        self._lock = Lock()
        self._last_accepted_at: Optional[datetime] = None
        self._worker: Optional[Thread] = None

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = Thread(target=self._run, name="presence-social-agent", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=3.0)

    def submit(self, event: PresenceEvent, world_state: WorldState) -> bool:
        """Queue an eligible event. Returns false when intentionally silent."""
        if event.event_type != "person_entered":
            return False
        now = datetime.now(timezone.utc)
        with self._lock:
            if self._last_accepted_at and (now - self._last_accepted_at).total_seconds() < self._cooldown_seconds:
                return False
            self._last_accepted_at = now
        try:
            self._queue.put_nowait((event, world_state))
            return True
        except Exception:
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event, world_state = self._queue.get(timeout=0.2)
            except Empty:
                continue
            self._generate(event, world_state)

    def _generate(self, event: PresenceEvent, world_state: WorldState) -> None:
        message_id = f"social_{uuid4().hex}"
        now = datetime.now(timezone.utc).isoformat()
        try:
            reply = self._text_caller(self._messages(event, world_state)).strip()
            reply = self._sanitize(reply)
            if not reply:
                raise RuntimeError("LLM returned an empty social reply")
            audio_path = self._tts_caller(reply) if self._tts_caller else None
            self._store.append_social_message(message_id, event.event_id, reply, now, "sent", audio_path)
        except Exception as exc:
            # Persist errors rather than silently losing an event. The UI can
            # show the issue without repeatedly re-triggering a greeting.
            self._store.append_social_message(
                message_id,
                event.event_id,
                f"[social generation failed: {type(exc).__name__}]",
                now,
                "error",
            )

    @staticmethod
    def _messages(event: PresenceEvent, world_state: WorldState) -> list[Dict[str, str]]:
        facts = [
            "触发事件：检测到一位访客进入摄像头区域。",
            f"当前在场人数：{world_state.active_count}。",
            f"视觉追踪置信度：{event.confidence:.2f}。",
            "身份状态：未确认。不要猜测姓名、性别、年龄、外貌、情绪或任何敏感特征。",
        ]
        return [
            {
                "role": "system",
                "content": (
                    "你是实验室里的友好机器人。根据已给出的事实，生成一句自然的中文主动招呼。"
                    "长度 10 到 35 字；可轻微俏皮，但不能冒犯、评价人、假装认识对方或编造事实。"
                    "只输出要说的话，不要解释、标题、引号或表情符号。"
                ),
            },
            {"role": "user", "content": "\n".join(facts)},
        ]

    @staticmethod
    def _sanitize(text: str) -> str:
        cleaned = " ".join(text.replace("\n", " ").split()).strip("\"'“”")
        # A greeting should be brief; trim instead of allowing an API response
        # to monopolize the audio channel.
        return cleaned[:70]
