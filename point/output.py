import os

import cv2

from config import CAPTURE_DIR, LATEST_FRAME_PATH


def save_pointing_images(frame):
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    cv2.imwrite(LATEST_FRAME_PATH, frame)

    return LATEST_FRAME_PATH


def draw_model_frame(raw_frame, objects, pointed_obj=None):
    model_frame = raw_frame.copy()
    pointed_id = pointed_obj.get("object_id") if pointed_obj is not None else None

    for obj in objects:
        x1, y1, x2, y2 = obj["box"]

        if pointed_id is not None and obj.get("object_id") == pointed_id:
            color = (0, 0, 255)
            thickness = 4
        else:
            color = (255, 0, 0)
            thickness = 2

        cv2.rectangle(
            model_frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        if pointed_id is not None and obj.get("object_id") == pointed_id:
            label = obj["class_name"] + " " + str(round(obj["confidence"], 2))
            cv2.putText(
                model_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return model_frame


def make_pointing_result(hit, timestamp, frame_path, pointed_obj=None):
    if not hit or pointed_obj is None:
        return {
            "hit": False,
            "timestamp": timestamp,
            "target_class": None,
            "bbox": None,
            "confidence": 0,
            "frame_path": frame_path,
        }

    return {
        "hit": True,
        "timestamp": timestamp,
        "target_class": pointed_obj["class_name"],
        "bbox": pointed_obj["box"],
        "confidence": round(float(pointed_obj.get("confidence", 0)), 4),
        "frame_path": frame_path,
    }
