"""短时记忆：按 session_id 分桶的 JSON 缓存。"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import Config

from .models import ChatTurn

logger = logging.getLogger("Memory.ShortTerm")


class ShortTermMemory:
    """
    为每个 session 维护一份对话历史 JSON 文件。

    - 存储目录: Config.MEMORY_SESSIONS_DIR
    - 文件命名: {session_id}.json
    - 写入策略: 内存增量 + 原子落盘（tmp -> rename）
    - 线程安全: 进程内一把全局锁即可（I/O 占比低）
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        max_turns: Optional[int] = None,
    ):
        self.storage_dir = Path(storage_dir or Config.MEMORY_SESSIONS_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_turns = max_turns or Config.MEMORY_SHORT_TERM_MAX_TURNS
        self._cache: Dict[str, List[ChatTurn]] = {}
        self._dirty: set = set()
        self._lock = threading.Lock()
        logger.info(
            "🗒️ [SHORT_TERM] 初始化: dir=%s, max_turns=%s",
            self.storage_dir,
            self.max_turns,
        )

    def _session_path(self, session_id: str) -> Path:
        safe = session_id.replace("/", "_").replace("\\", "_") or "default"
        return self.storage_dir / f"{safe}.json"

    def _load_session(self, session_id: str) -> List[ChatTurn]:
        if session_id in self._cache:
            return self._cache[session_id]

        path = self._session_path(session_id)
        turns: List[ChatTurn] = []
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                for item in raw.get("turns", []):
                    try:
                        turns.append(ChatTurn(**item))
                    except Exception as exc:
                        logger.warning("⚠️ [SHORT_TERM] 跳过无效 turn: %s", exc)
            except Exception as exc:
                logger.error("❌ [SHORT_TERM] 会话文件解析失败 %s: %s", path, exc)

        self._cache[session_id] = turns
        return turns

    def _persist_session(self, session_id: str) -> None:
        turns = self._cache.get(session_id, [])
        path = self._session_path(session_id)
        payload = {
            "session_id": session_id,
            "updated_at": datetime.utcnow().isoformat(),
            "turns": [turn.dict() for turn in turns],
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
            self._dirty.discard(session_id)
        except Exception as exc:
            logger.error("❌ [SHORT_TERM] 写入失败 %s: %s", path, exc)

    def add_chat_history(self, session_id: str, role: str, content: str) -> None:
        if not content:
            return
        with self._lock:
            turns = self._load_session(session_id)
            turns.append(ChatTurn(role=role, content=content))
            if len(turns) > self.max_turns:
                del turns[: len(turns) - self.max_turns]
            self._dirty.add(session_id)
            self._persist_session(session_id)

    def get_raw_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            turns = self._load_session(session_id)
            return [turn.to_message() for turn in turns]

    def get_turns(self, session_id: str) -> List[ChatTurn]:
        with self._lock:
            return list(self._load_session(session_id))

    def clear_chat_history(self, session_id: str) -> bool:
        with self._lock:
            self._cache[session_id] = []
            path = self._session_path(session_id)
            try:
                if path.exists():
                    path.unlink()
                self._dirty.discard(session_id)
                logger.info("🧹 [SHORT_TERM] 清空 session=%s", session_id)
                return True
            except Exception as exc:
                logger.error("❌ [SHORT_TERM] 清空失败 %s: %s", session_id, exc)
                return False

    def get_history_count(self, session_id: str) -> int:
        with self._lock:
            return len(self._load_session(session_id))

    def list_sessions(self) -> List[str]:
        sessions: List[str] = []
        for path in self.storage_dir.glob("*.json"):
            sessions.append(path.stem)
        return sorted(sessions)

    def sync_persistence(self) -> None:
        with self._lock:
            for session_id in list(self._dirty):
                self._persist_session(session_id)
