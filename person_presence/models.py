"""Data contracts shared by webcam perception, World Model, and API."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PersonDetection:
    track_id: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    timestamp: str

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class PersonState:
    track_id: str
    status: str  # candidate | present | absent
    first_seen_at: str
    last_seen_at: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    frames_seen: int = 1
    person_id: Optional[str] = None  # Reserved for opt-in ReID / user confirmation.
    identity_confidence: float = 0.0

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        return payload


@dataclass(frozen=True)
class PresenceEvent:
    event_id: str
    event_type: str  # person_entered | person_exited
    track_id: str
    timestamp: str
    confidence: float
    person_id: Optional[str] = None
    evidence_frame_path: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(frozen=True)
class WorldState:
    updated_at: str
    persons: List[Dict]
    active_count: int

    def to_dict(self) -> Dict:
        return asdict(self)

