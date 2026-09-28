"""Webcam worker: detector/tracker produces observations; manager owns state."""

import threading
import time
from typing import List, Optional

import cv2

from .config import PresenceConfig
from .event_store import EventStore
from .models import PersonDetection, WorldState, utc_now
from .presence import PresenceManager
from .social import SocialAgent


class WebcamPresenceService:
    def __init__(self, config: PresenceConfig, store: EventStore, social_agent: Optional[SocialAgent] = None):
        self.config = config
        self.store = store
        self.manager = PresenceManager(config)
        self.social_agent = social_agent
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None

    @property
    def error(self) -> Optional[str]:
        return self._error

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="webcam-presence", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def snapshot(self) -> WorldState:
        return self.manager.snapshot()

    def jpeg_frame(self) -> Optional[bytes]:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            ok, buffer = cv2.imencode(".jpg", self._latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buffer.tobytes() if ok else None

    def _run(self) -> None:
        try:
            from ultralytics import YOLO

            model = YOLO(self.config.model_path)
            camera = cv2.VideoCapture(self.config.camera_index)
            if not camera.isOpened():
                self._error = f"Cannot open camera index {self.config.camera_index}"
                return

            while not self._stop.is_set():
                ok, frame = camera.read()
                if not ok:
                    self._error = "Camera frame read failed"
                    time.sleep(0.1)
                    continue
                height, width = frame.shape[:2]
                results = model.track(
                    frame,
                    classes=[0],  # COCO class 0: person
                    conf=self.config.confidence,
                    imgsz=self.config.image_size,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
                detections = self._detections_from_results(results)
                events = self.manager.update(detections, (width, height))
                for event in events:
                    self.store.append(event)
                    if self.social_agent:
                        self.social_agent.submit(event, self.manager.snapshot())
                self._draw_overlay(frame, detections)
                with self._frame_lock:
                    self._latest_frame = frame
        except Exception as exc:  # Worker errors should be observable through /health.
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                camera.release()  # type: ignore[name-defined]
            except (NameError, AttributeError):
                pass

    @staticmethod
    def _detections_from_results(results) -> List[PersonDetection]:
        detections: List[PersonDetection] = []
        timestamp = utc_now()
        for result in results:
            if result.boxes is None:
                continue
            for index, box in enumerate(result.boxes):
                track_id = None if box.id is None else int(box.id[0].item())
                # ByteTrack can omit an ID on its first association; do not
                # manufacture a permanent ID for an untracked one-frame box.
                if track_id is None:
                    continue
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                detections.append(
                    PersonDetection(
                        track_id=f"camera_track_{track_id}",
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf[0].item()),
                        timestamp=timestamp,
                    )
                )
        return detections

    def _draw_overlay(self, frame, detections: List[PersonDetection]) -> None:
        height, width = frame.shape[:2]
        x, y, zone_w, zone_h = self.config.zone
        start = (int(x * width), int(y * height))
        end = (int((x + zone_w) * width), int((y + zone_h) * height))
        cv2.rectangle(frame, start, end, (0, 180, 255), 2)
        cv2.putText(frame, "presence zone", (start[0] + 6, start[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{detection.track_id} {detection.confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
