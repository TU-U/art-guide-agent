"""State machine that turns short-lived vision tracks into visit events."""

from datetime import datetime, timezone
from threading import RLock
from typing import Dict, Iterable, List, Tuple
from uuid import uuid4

from .config import PresenceConfig
from .models import PersonDetection, PersonState, PresenceEvent, WorldState


class PresenceManager:
    """Owns the in-memory people section of the World Model.

    Track IDs are deliberately camera-session IDs. A future identity resolver
    may link a track to a consented ``person_id``; it must not overwrite a
    track merely because an appearance match is uncertain.
    """

    def __init__(self, config: PresenceConfig):
        self._config = config
        self._tracks: Dict[str, PersonState] = {}
        self._lock = RLock()

    def in_zone(self, detection: PersonDetection, frame_size: Tuple[int, int]) -> bool:
        width, height = frame_size
        if width <= 0 or height <= 0:
            return False
        center_x, center_y = detection.center
        x, y, zone_w, zone_h = self._config.zone
        return (
            x * width <= center_x <= (x + zone_w) * width
            and y * height <= center_y <= (y + zone_h) * height
        )

    def update(
        self, detections: Iterable[PersonDetection], frame_size: Tuple[int, int]
    ) -> List[PresenceEvent]:
        events: List[PresenceEvent] = []
        visible_ids = set()
        with self._lock:
            for detection in detections:
                if not self.in_zone(detection, frame_size):
                    continue
                visible_ids.add(detection.track_id)
                state = self._tracks.get(detection.track_id)
                if state is None:
                    state = PersonState(
                        track_id=detection.track_id,
                        status="candidate",
                        first_seen_at=detection.timestamp,
                        last_seen_at=detection.timestamp,
                        confidence=detection.confidence,
                        bbox=detection.bbox,
                    )
                    self._tracks[detection.track_id] = state
                else:
                    state.last_seen_at = detection.timestamp
                    state.bbox = detection.bbox
                    state.confidence = detection.confidence
                    state.frames_seen += 1

                if state.status == "candidate" and state.frames_seen >= self._config.confirm_frames:
                    state.status = "present"
                    events.append(self._event("person_entered", state))

            now = datetime.now(timezone.utc)
            expired_candidates = []
            for state in self._tracks.values():
                if state.status == "candidate" and state.track_id not in visible_ids:
                    last_seen = datetime.fromisoformat(state.last_seen_at)
                    if (now - last_seen).total_seconds() >= self._config.exit_after_seconds:
                        expired_candidates.append(state.track_id)
                    continue
                if state.status != "present" or state.track_id in visible_ids:
                    continue
                last_seen = datetime.fromisoformat(state.last_seen_at)
                if (now - last_seen).total_seconds() >= self._config.exit_after_seconds:
                    state.status = "absent"
                    events.append(self._event("person_exited", state))
            for track_id in expired_candidates:
                # Never confirmed as a person entering the zone; discard this
                # transient detection instead of growing the World Model.
                self._tracks.pop(track_id, None)
        return events

    def snapshot(self) -> WorldState:
        with self._lock:
            persons = [state.to_dict() for state in self._tracks.values() if state.status == "present"]
        return WorldState(
            updated_at=datetime.now(timezone.utc).isoformat(),
            persons=persons,
            active_count=len(persons),
        )

    @staticmethod
    def _event(event_type: str, state: PersonState) -> PresenceEvent:
        return PresenceEvent(
            event_id=f"presence_{uuid4().hex}",
            event_type=event_type,
            track_id=state.track_id,
            person_id=state.person_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=state.confidence,
        )
