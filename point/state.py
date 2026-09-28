from collections import deque

import mediapipe as mp
from cv_bridge import CvBridge
from ultralytics import YOLO

from config import YOLO_MODEL_PATH


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

yolo_model = YOLO(YOLO_MODEL_PATH)

bridge = CvBridge()
latest_frame = None
latest_pointing_result = {
    "hit": False,
    "timestamp": 0,
    "target_class": None,
    "bbox": None,
    "confidence": 0,
    "frame_path": "rviz_captured_images/latest.jpg",
}

dist_history = deque(maxlen=5)
last_yolo_time = 0
last_yolo_objects = []
