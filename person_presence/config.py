"""Configuration for the local webcam person-presence prototype."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Tuple


def _float_tuple(value: str, default: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
        if len(result) == 4 and all(0.0 <= item <= 1.0 for item in result):
            return result  # type: ignore[return-value]
    except (AttributeError, ValueError):
        pass
    return default


@dataclass(frozen=True)
class PresenceConfig:
    """Environment-configurable settings.

    ``zone`` is a normalized rectangle: x, y, width, height. The default is
    the complete image, so simply appearing in front of a webcam counts as an
    entry. Point a doorway camera at the door and narrow this rectangle later.
    """

    camera_index: int = int(os.getenv("PRESENCE_CAMERA_INDEX", "0"))
    model_path: str = os.getenv("PRESENCE_YOLO_MODEL", "yolov8s.pt")
    image_size: int = int(os.getenv("PRESENCE_IMAGE_SIZE", "640"))
    confidence: float = float(os.getenv("PRESENCE_CONFIDENCE", "0.45"))
    exit_after_seconds: float = float(os.getenv("PRESENCE_EXIT_AFTER_SECONDS", "3.0"))
    confirm_frames: int = int(os.getenv("PRESENCE_CONFIRM_FRAMES", "3"))
    zone: Tuple[float, float, float, float] = _float_tuple(
        os.getenv("PRESENCE_ZONE", "0,0,1,1"), (0.0, 0.0, 1.0, 1.0)
    )
    database_path: Path = Path(os.getenv("PRESENCE_DATABASE", "data/person_presence.sqlite3"))
    social_enabled: bool = os.getenv("PRESENCE_SOCIAL_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    social_tts_enabled: bool = os.getenv("PRESENCE_SOCIAL_TTS", "false").lower() in {"1", "true", "yes", "on"}
    social_cooldown_seconds: float = float(os.getenv("PRESENCE_SOCIAL_COOLDOWN_SECONDS", "45"))
