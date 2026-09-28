"""Small local event store. SQLite keeps the prototype dependency-free."""

import sqlite3
import threading
from pathlib import Path
from typing import Dict, List

from .models import PresenceEvent


class EventStore:
    def __init__(self, database_path: Path):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(database_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS presence_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    person_id TEXT,
                    timestamp TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_frame_path TEXT
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_presence_events_time "
                "ON presence_events(timestamp DESC)"
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS social_messages (
                    message_id TEXT PRIMARY KEY,
                    source_event_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    audio_path TEXT
                )
                """
            )
            self._connection.commit()

    def append(self, event: PresenceEvent) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO presence_events(
                    event_id, event_type, track_id, person_id, timestamp,
                    confidence, evidence_frame_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.event_type,
                    event.track_id,
                    event.person_id,
                    event.timestamp,
                    event.confidence,
                    event.evidence_frame_path,
                ),
            )
            self._connection.commit()

    def recent(self, limit: int = 100) -> List[Dict]:
        safe_limit = max(1, min(limit, 500))
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM presence_events ORDER BY timestamp DESC LIMIT ?", (safe_limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def today_entry_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count FROM presence_events
                WHERE event_type = 'person_entered' AND date(timestamp, 'localtime') = date('now', 'localtime')
                """
            ).fetchone()
        return int(row["count"])

    def append_social_message(
        self,
        message_id: str,
        source_event_id: str,
        text: str,
        timestamp: str,
        status: str,
        audio_path: str | None = None,
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO social_messages(message_id, source_event_id, text, timestamp, status, audio_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, source_event_id, text, timestamp, status, audio_path),
            )
            self._connection.commit()

    def recent_social_messages(self, limit: int = 50) -> List[Dict]:
        safe_limit = max(1, min(limit, 200))
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM social_messages ORDER BY timestamp DESC LIMIT ?", (safe_limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._connection.close()
