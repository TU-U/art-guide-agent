"""User registry for stable user identity and first-time name capture."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from config import Config


class UserRecord(BaseModel):
    user_id: str
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_session_id: Optional[str] = None


class UserRegistry:
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = Path(storage_path or Config.MEMORY_USER_REGISTRY_PATH)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._users: Dict[str, UserRecord] = {}
        self._session_bindings: Dict[str, str] = {}
        self._pending_name_sessions: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            self._persist()
            return
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            users = raw.get("users", {})
            self._users = {
                user_id: UserRecord(**payload)
                for user_id, payload in users.items()
                if isinstance(payload, dict)
            }
            self._session_bindings = {
                str(k): str(v) for k, v in raw.get("session_bindings", {}).items()
            }
            self._pending_name_sessions = set(raw.get("pending_name_sessions", []))
        except Exception:
            self._users = {}
            self._session_bindings = {}
            self._pending_name_sessions = set()

    def _persist(self) -> None:
        payload = {
            "updated_at": datetime.utcnow().isoformat(),
            "users": {user_id: self._record_to_dict(record) for user_id, record in self._users.items()},
            "session_bindings": dict(self._session_bindings),
            "pending_name_sessions": sorted(self._pending_name_sessions),
        }
        tmp_path = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.storage_path)

    @staticmethod
    def _record_to_dict(record: UserRecord) -> Dict:
        if hasattr(record, "model_dump"):
            return record.model_dump(mode="json")
        return json.loads(record.json())

    def get_user(self, user_id: str) -> Optional[UserRecord]:
        with self._lock:
            return self._users.get(user_id)

    def list_users(self) -> List[UserRecord]:
        with self._lock:
            return list(self._users.values())

    def resolve_user(self, user_id: Optional[str], session_id: Optional[str]) -> Optional[UserRecord]:
        with self._lock:
            if session_id and session_id in self._session_bindings:
                bound_user_id = self._session_bindings[session_id]
                record = self._users.get(bound_user_id)
                if record is not None:
                    record.updated_at = datetime.utcnow()
                    record.last_session_id = session_id
                    self._persist()
                return record
            if user_id and user_id != "anonymous" and user_id in self._users:
                if session_id:
                    self._session_bindings[session_id] = user_id
                record = self._users[user_id]
                record.updated_at = datetime.utcnow()
                record.last_session_id = session_id
                self._persist()
                return record
            return None

    def mark_waiting_name(self, session_id: str) -> None:
        with self._lock:
            self._pending_name_sessions.add(session_id)
            self._persist()

    def is_waiting_name(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._pending_name_sessions

    def clear_waiting_name(self, session_id: str) -> None:
        with self._lock:
            self._pending_name_sessions.discard(session_id)
            self._persist()

    def _find_by_name(self, name: str) -> Optional[UserRecord]:
        normalized = name.strip().lower()
        for record in self._users.values():
            if record.name.strip().lower() == normalized:
                return record
        return None

    def register_name(self, name: str, session_id: str) -> UserRecord:
        with self._lock:
            existing = self._find_by_name(name)
            now = datetime.utcnow()
            if existing is not None:
                existing.updated_at = now
                existing.last_session_id = session_id
                record = existing
            else:
                record = UserRecord(
                    user_id=f"user_{uuid4().hex[:12]}",
                    name=name,
                    created_at=now,
                    updated_at=now,
                    last_session_id=session_id,
                )
                self._users[record.user_id] = record

            self._session_bindings[session_id] = record.user_id
            self._pending_name_sessions.discard(session_id)
            self._persist()
            return record
