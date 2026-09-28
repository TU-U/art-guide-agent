from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4


VALID_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def find_latest_image(image_dir: Path) -> Optional[Path]:
    if not image_dir.exists() or not image_dir.is_dir():
        return None

    candidates = []
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        candidates.append(path)

    if not candidates:
        return None

    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def consume_latest_image(image_dir: Path) -> Tuple[Optional[bytes], Optional[Path]]:
    image_path = find_latest_image(image_dir)
    if image_path is None:
        return None, None

    claimed_path = image_path.with_name(
        f".consuming_{image_path.stem}_{uuid4().hex}{image_path.suffix}.tmp"
    )

    try:
        image_path.replace(claimed_path)
    except FileNotFoundError:
        return None, None

    try:
        return claimed_path.read_bytes(), image_path
    except OSError:
        return None, None
    finally:
        claimed_path.unlink(missing_ok=True)
