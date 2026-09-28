import rclpy
import threading
import cv2
import mediapipe as mp
import numpy as np
import asyncio
import time

from collections import deque
from collections import defaultdict
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from aiohttp import web
from ultralytics import YOLO


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

yolo_model = YOLO("yolov8s.pt")

bridge = CvBridge()
latest_frame = None

TRIGGER_DISTANCE_MM = 2000

PROCESS_W = 480

dist_history = deque(maxlen=5)

last_yolo_time = 0
YOLO_INTERVAL = 1.0
last_yolo_objects = []
yolo_track_id_counter = 0

RAY_STEP_MM = 60
RAY_MAX_MM = 4000
HIT_3D_THRESHOLD_MM = 70
RAY_START_OFFSET_MM = 120
LOCAL_CLOUD_RADIUS = 3
LOCAL_CLOUD_MIN_POINTS = 5
LOCAL_CLOUD_DEPTH_BAND_MM = 180
LOCAL_CLOUD_HIT_THRESHOLD_MM = 65
CROP_SIZE = 224
RERAYCAST_MOVE_THRESHOLD_MM = 40
RERAYCAST_ANGLE_THRESHOLD_DEG = 4
YOLO_TRACK_IOU_THRESHOLD = 0.3
YOLO_TRACK_KEEP_SECONDS = 1.0
YOLO_CLASS_VOTE_WINDOW = 6

SMOOTH_ALPHA_2D = 0.45
SMOOTH_ALPHA_DEPTH = 0.45
SMOOTH_ALPHA_DIR = 0.4


def ema_smooth(old_value, new_value, alpha):
    if old_value is None:
        return new_value

    return old_value * (1.0 - alpha) + new_value * alpha


def get_depth_near_point(depth, x, y, radius=2):
    h, w = depth.shape[:2]

    x = int(np.clip(x, radius, w - radius - 1))
    y = int(np.clip(y, radius, h - radius - 1))

    region = depth[y - radius:y + radius + 1, x - radius:x + radius + 1]
    valid = region[region > 0]

    if len(valid) == 0:
        return None

    return int(np.median(valid))


def pixel_to_3d(u, v, depth_mm, fx, fy, cx, cy):
    if depth_mm is None or depth_mm <= 0:
        return None

    z = float(depth_mm)
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy

    return np.array([x, y, z], dtype=np.float32)


def project_3d_to_pixel(point_3d, fx, fy, cx, cy):
    x = point_3d[0]
    y = point_3d[1]
    z = point_3d[2]

    if z <= 0:
        return None

    u = int(x * fx / z + cx)
    v = int(y * fy / z + cy)

    return u, v


def point_in_box(u, v, box):
    x1, y1, x2, y2 = box
    return x1 <= u <= x2 and y1 <= v <= y2


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return float(inter_area) / float(union_area)


def stable_class_from_votes(class_votes):
    score_by_class = defaultdict(float)
    best_vote = None

    for vote in class_votes:
        score_by_class[vote["class_id"]] += vote["confidence"]

        if best_vote is None or vote["confidence"] > best_vote["confidence"]:
            best_vote = vote

    if best_vote is None:
        return None, None, 0.0

    best_class_id = max(score_by_class, key=score_by_class.get)

    for vote in reversed(class_votes):
        if vote["class_id"] == best_class_id:
            return best_class_id, vote["class_name"], score_by_class[best_class_id]

    return best_vote["class_id"], best_vote["class_name"], best_vote["confidence"]


def update_yolo_tracks(existing_tracks, detections, now):
    global yolo_track_id_counter

    matched_track_ids = set()

    for detection in detections:
        best_track = None
        best_iou = 0.0

        for track in existing_tracks:
            if track["object_id"] in matched_track_ids:
                continue

            iou = box_iou(track["box"], detection["box"])

            if iou > best_iou:
                best_iou = iou
                best_track = track

        if best_track is not None and best_iou >= YOLO_TRACK_IOU_THRESHOLD:
            best_track["box"] = detection["box"]
            best_track["last_seen_time"] = now
            best_track["last_iou"] = round(best_iou, 2)
            best_track["class_votes"].append({
                "class_id": detection["class_id"],
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
            })
            matched_track_ids.add(best_track["object_id"])
        else:
            yolo_track_id_counter += 1
            existing_tracks.append({
                "object_id": yolo_track_id_counter,
                "box": detection["box"],
                "last_seen_time": now,
                "last_iou": 1.0,
                "class_votes": deque([
                    {
                        "class_id": detection["class_id"],
                        "class_name": detection["class_name"],
                        "confidence": detection["confidence"],
                    }
                ], maxlen=YOLO_CLASS_VOTE_WINDOW),
            })

    stable_tracks = []

    for track in existing_tracks:
        if now - track["last_seen_time"] > YOLO_TRACK_KEEP_SECONDS:
            continue

        class_id, class_name, confidence_score = stable_class_from_votes(track["class_votes"])

        if class_id is None:
            continue

        stable_tracks.append({
            "object_id": track["object_id"],
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence_score / max(1, len(track["class_votes"])),
            "box": track["box"],
            "last_seen_age": now - track["last_seen_time"],
            "last_iou": track.get("last_iou", 0.0),
            "vote_count": len(track["class_votes"]),
        })

    return stable_tracks


def get_local_surface_points(depth, u, v, fx, fy, cx, cy, radius=LOCAL_CLOUD_RADIUS):
    h, w = depth.shape[:2]

    u = int(np.clip(u, radius, w - radius - 1))
    v = int(np.clip(v, radius, h - radius - 1))

    region = depth[v - radius:v + radius + 1, u - radius:u + radius + 1]
    valid_mask = region > 0

    if int(np.count_nonzero(valid_mask)) < LOCAL_CLOUD_MIN_POINTS:
        return None

    valid_depths = region[valid_mask].astype(np.float32)
    median_depth = float(np.median(valid_depths))
    foreground_mask = valid_mask & (
        np.abs(region.astype(np.float32) - median_depth) <= LOCAL_CLOUD_DEPTH_BAND_MM
    )

    if int(np.count_nonzero(foreground_mask)) < LOCAL_CLOUD_MIN_POINTS:
        return None

    ys, xs = np.where(foreground_mask)
    pixel_us = u - radius + xs
    pixel_vs = v - radius + ys
    zs = depth[pixel_vs, pixel_us].astype(np.float32)

    points_x = (pixel_us.astype(np.float32) - cx) * zs / fx
    points_y = (pixel_vs.astype(np.float32) - cy) * zs / fy
    points = np.stack([points_x, points_y, zs], axis=1).astype(np.float32)

    return points, int(np.median(zs)), int(len(points))


def local_cloud_matches_ray(point_3d, local_points):
    distances = np.linalg.norm(local_points - point_3d, axis=1)
    close_count = int(np.count_nonzero(distances <= LOCAL_CLOUD_HIT_THRESHOLD_MM))

    if close_count < LOCAL_CLOUD_MIN_POINTS:
        return False, None, close_count

    return True, float(np.min(distances)), close_count


def should_reraycast(
    tip_3d,
    direction_3d,
    last_tip_3d,
    last_direction_3d,
    move_threshold_mm=RERAYCAST_MOVE_THRESHOLD_MM,
    angle_threshold_deg=RERAYCAST_ANGLE_THRESHOLD_DEG,
):
    if tip_3d is None or direction_3d is None:
        return False

    if last_tip_3d is None or last_direction_3d is None:
        return True

    move_mm = float(np.linalg.norm(tip_3d - last_tip_3d))

    if move_mm >= move_threshold_mm:
        return True

    dot_value = float(np.dot(direction_3d, last_direction_3d))
    dot_value = float(np.clip(dot_value, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(dot_value)))

    return angle_deg >= angle_threshold_deg


def crop_hit_region(frame, hit_u, hit_v, crop_size=CROP_SIZE):
    h, w = frame.shape[:2]
    half = crop_size // 2

    x1 = max(0, int(hit_u) - half)
    y1 = max(0, int(hit_v) - half)
    x2 = min(w, int(hit_u) + half)
    y2 = min(h, int(hit_v) + half)

    if x2 <= x1 or y2 <= y1:
        return None, None

    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def raycast_pointed_object(tip_3d, direction_3d, objects, depth, fx, fy, cx, cy, frame_w, frame_h):
    if tip_3d is None or direction_3d is None:
        return None, None, None

    direction_len = np.linalg.norm(direction_3d)

    if direction_len < 0.001:
        return None, None, None

    direction_3d = direction_3d / direction_len

    for ray_len in range(RAY_START_OFFSET_MM, RAY_MAX_MM, RAY_STEP_MM):
        point_3d = tip_3d + direction_3d * ray_len

        pixel = project_3d_to_pixel(
            point_3d,
            fx,
            fy,
            cx,
            cy
        )

        if pixel is None:
            continue

        u, v = pixel

        if u < 0 or u >= frame_w or v < 0 or v >= frame_h:
            break

        local_surface = get_local_surface_points(
            depth,
            u,
            v,
            fx,
            fy,
            cx,
            cy
        )

        if local_surface is None:
            continue

        local_points, real_depth, local_point_count = local_surface
        is_local_hit, hit_error, close_count = local_cloud_matches_ray(
            point_3d,
            local_points
        )

        if not is_local_hit:
            continue

        hit_obj = None

        for obj in objects:
            if point_in_box(u, v, obj["box"]):
                hit_obj = obj
                break

        if hit_obj is None:
            blocked_info = {
                "hit_u": u,
                "hit_v": v,
                "hit_depth_mm": real_depth,
                "ray_depth_mm": int(point_3d[2]),
                "ray_len_mm": ray_len,
                "hit_error_mm": round(float(hit_error), 1),
                "local_cloud_points": local_point_count,
                "local_cloud_close_points": close_count,
            }

            return None, (u, v), blocked_info

        hit_obj["hit_u"] = u
        hit_obj["hit_v"] = v
        hit_obj["hit_depth_mm"] = real_depth
        hit_obj["ray_depth_mm"] = int(point_3d[2])
        hit_obj["ray_len_mm"] = ray_len
        hit_obj["hit_error_mm"] = round(float(hit_error), 1)
        hit_obj["hit_method"] = "ray_step_local_cloud"
        hit_obj["local_cloud_points"] = local_point_count
        hit_obj["local_cloud_close_points"] = close_count

        return hit_obj, (u, v), None

    return None, None, None


class HandDetectNode(Node):
    def __init__(self):
        super().__init__("hand_stream")

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.smooth_base_2d = None
        self.smooth_tip_2d = None
        self.smooth_base_depth = None
        self.smooth_tip_depth = None
        self.smooth_direction_3d = None
        self.last_raycast_tip_3d = None
        self.last_raycast_direction_3d = None

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self.camera_info_callback,
            10
        )

        color_sub = Subscriber(
            self,
            Image,
            "/camera/color/image_raw"
        )

        depth_sub = Subscriber(
            self,
            Image,
            "/camera/depth/image_raw"
        )

        sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=10,
            slop=0.3
        )

        sync.registerCallback(self.callback)

        self.get_logger().info("Hand + YOLO + smoothed 3D ray stream started")

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def reset_raycast_cache(self):
        self.last_raycast_tip_3d = None
        self.last_raycast_direction_3d = None

    def callback(self, color_msg, depth_msg):
        global latest_frame
        global last_yolo_time
        global last_yolo_objects

        frame = bridge.imgmsg_to_cv2(color_msg, "bgr8")
        raw_frame = frame.copy()

        depth = bridge.imgmsg_to_cv2(depth_msg, "16UC1")

        orig_h, orig_w = frame.shape[:2]

        if self.fx is None:
            latest_frame = frame
            return

        now = time.time()

        if now - last_yolo_time > YOLO_INTERVAL:
            last_yolo_time = now

            yolo_results = yolo_model(
                raw_frame,
                imgsz=320,
                verbose=False
            )

            yolo_detections = []

            for result in yolo_results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_id == 0:
                        continue

                    if conf < 0.35:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    yolo_detections.append({
                        "class_id": cls_id,
                        "class_name": yolo_model.names[cls_id],
                        "confidence": conf,
                        "box": [
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2)
                        ]
                    })

            last_yolo_objects = update_yolo_tracks(
                last_yolo_objects,
                yolo_detections,
                now
            )

        for obj in last_yolo_objects:
            x1, y1, x2, y2 = obj["box"]
            label = (
                obj["class_name"]
                + " "
                + str(round(obj["confidence"], 2))
                + " age="
                + str(round(obj.get("last_seen_age", 0.0), 1))
                + " vote="
                + str(obj.get("vote_count", 0))
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        process_w = PROCESS_W
        process_h = int(orig_h * process_w / orig_w)

        small = cv2.resize(raw_frame, (process_w, process_h))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        pointed_text = ""

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            base = hand_landmarks.landmark[5]
            tip = hand_landmarks.landmark[8]

            raw_bx = int(base.x * orig_w)
            raw_by = int(base.y * orig_h)

            raw_tx = int(tip.x * orig_w)
            raw_ty = int(tip.y * orig_h)

            raw_base_2d = np.array([raw_bx, raw_by], dtype=np.float32)
            raw_tip_2d = np.array([raw_tx, raw_ty], dtype=np.float32)

            self.smooth_base_2d = ema_smooth(
                self.smooth_base_2d,
                raw_base_2d,
                SMOOTH_ALPHA_2D
            )

            self.smooth_tip_2d = ema_smooth(
                self.smooth_tip_2d,
                raw_tip_2d,
                SMOOTH_ALPHA_2D
            )

            bx = int(self.smooth_base_2d[0])
            by = int(self.smooth_base_2d[1])

            tx = int(self.smooth_tip_2d[0])
            ty = int(self.smooth_tip_2d[1])

            base_depth = get_depth_near_point(depth, bx, by, radius=2)
            tip_depth = get_depth_near_point(depth, tx, ty, radius=2)

            if base_depth is not None:
                self.smooth_base_depth = ema_smooth(
                    self.smooth_base_depth,
                    float(base_depth),
                    SMOOTH_ALPHA_DEPTH
                )

            if tip_depth is not None:
                self.smooth_tip_depth = ema_smooth(
                    self.smooth_tip_depth,
                    float(tip_depth),
                    SMOOTH_ALPHA_DEPTH
                )

            avg_tip_depth = None

            if self.smooth_tip_depth is not None:
                avg_tip_depth = int(self.smooth_tip_depth)
                dist_history.append(avg_tip_depth)

                avg_tip_depth = int(
                    sum(dist_history) / len(dist_history)
                )

            base_3d = pixel_to_3d(
                bx,
                by,
                int(self.smooth_base_depth) if self.smooth_base_depth is not None else None,
                self.fx,
                self.fy,
                self.cx,
                self.cy
            )

            tip_3d = pixel_to_3d(
                tx,
                ty,
                avg_tip_depth,
                self.fx,
                self.fy,
                self.cx,
                self.cy
            )

            direction_3d = None

            if base_3d is not None and tip_3d is not None:
                raw_direction_3d = tip_3d - base_3d
                raw_direction_len = np.linalg.norm(raw_direction_3d)

                if raw_direction_len > 20:
                    raw_direction_3d = raw_direction_3d / raw_direction_len

                    self.smooth_direction_3d = ema_smooth(
                        self.smooth_direction_3d,
                        raw_direction_3d,
                        SMOOTH_ALPHA_DIR
                    )

                    smooth_len = np.linalg.norm(self.smooth_direction_3d)

                    if smooth_len > 0.001:
                        direction_3d = self.smooth_direction_3d / smooth_len

            effective_tip_3d = tip_3d
            effective_direction_3d = direction_3d
            reraycast_needed = False

            if tip_3d is not None and direction_3d is not None:
                reraycast_needed = should_reraycast(
                    tip_3d,
                    direction_3d,
                    self.last_raycast_tip_3d,
                    self.last_raycast_direction_3d
                )

                if not reraycast_needed and self.last_raycast_tip_3d is not None:
                    effective_tip_3d = self.last_raycast_tip_3d
                    effective_direction_3d = self.last_raycast_direction_3d

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            cv2.circle(frame, (bx, by), 6, (0, 255, 255), -1)
            cv2.circle(frame, (tx, ty), 6, (0, 0, 255), -1)

            cv2.line(
                frame,
                (bx, by),
                (tx, ty),
                (0, 255, 255),
                3
            )

            if effective_tip_3d is not None and effective_direction_3d is not None:
                last_pixel = None

                for ray_len in range(0, 1200, 60):
                    p3d = effective_tip_3d + effective_direction_3d * ray_len

                    pixel = project_3d_to_pixel(
                        p3d,
                        self.fx,
                        self.fy,
                        self.cx,
                        self.cy
                    )

                    if pixel is None:
                        continue

                    u, v = pixel

                    if u < 0 or u >= orig_w or v < 0 or v >= orig_h:
                        break

                    if last_pixel is not None:
                        cv2.line(
                            frame,
                            last_pixel,
                            (u, v),
                            (0, 0, 255),
                            2
                        )

                    last_pixel = (u, v)

            if avg_tip_depth is None:
                cv2.putText(
                    frame,
                    "no valid finger depth",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                latest_frame = frame
                self.reset_raycast_cache()
                return

            cv2.putText(
                frame,
                "finger depth: " + str(avg_tip_depth) + " mm",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            if avg_tip_depth > TRIGGER_DISTANCE_MM:
                cv2.putText(
                    frame,
                    "HAND TOO FAR",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )

                latest_frame = frame
                self.reset_raycast_cache()
                return

            cv2.putText(
                frame,
                "HAND ACTIVE",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                3
            )

            if tip_3d is None or direction_3d is None:
                pointed_obj = None
                hit_pixel = None
                blocked_info = None
                self.reset_raycast_cache()
            elif effective_tip_3d is not None and effective_direction_3d is not None:
                if reraycast_needed:
                    self.last_raycast_tip_3d = effective_tip_3d.copy()
                    self.last_raycast_direction_3d = effective_direction_3d.copy()

                pointed_obj, hit_pixel, blocked_info = raycast_pointed_object(
                    effective_tip_3d,
                    effective_direction_3d,
                    last_yolo_objects,
                    depth,
                    self.fx,
                    self.fy,
                    self.cx,
                    self.cy,
                    orig_w,
                    orig_h
                )
            else:
                pointed_obj = None
                hit_pixel = None
                blocked_info = None

            if pointed_obj is not None:
                x1, y1, x2, y2 = pointed_obj["box"]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    4
                )

                if hit_pixel is not None:
                    cv2.circle(
                        frame,
                        hit_pixel,
                        8,
                        (0, 0, 255),
                        -1
                    )

                    hit_crop, crop_box = crop_hit_region(raw_frame, hit_pixel[0], hit_pixel[1])

                    if crop_box is not None:
                        cx1, cy1, cx2, cy2 = crop_box
                        pointed_obj["crop_box"] = crop_box
                        pointed_obj["crop_shape"] = hit_crop.shape[:2] if hit_crop is not None else None

                        cv2.rectangle(
                            frame,
                            (cx1, cy1),
                            (cx2, cy2),
                            (0, 255, 255),
                            2
                        )

                pointed_text = (
                    "pointing: "
                    + pointed_obj["class_name"]
                    + " method="
                    + str(pointed_obj.get("hit_method", "raycast"))
                    + " hit="
                    + str(pointed_obj.get("hit_error_mm"))
                    + "mm"
                    + " depth="
                    + str(pointed_obj.get("hit_depth_mm"))
                    + "mm"
                )

                if pointed_obj.get("hit_method") == "ray_step_local_cloud":
                    pointed_text += (
                        " lc="
                        + str(pointed_obj.get("local_cloud_close_points"))
                        + "/"
                        + str(pointed_obj.get("local_cloud_points"))
                    )

            else:
                if blocked_info is not None and hit_pixel is not None:
                    cv2.circle(
                        frame,
                        hit_pixel,
                        8,
                        (255, 0, 255),
                        -1
                    )

                    pointed_text = (
                        "blocked: ray_len="
                        + str(blocked_info.get("ray_len_mm"))
                        + "mm"
                        + " depth="
                        + str(blocked_info.get("hit_depth_mm"))
                        + "mm"
                        + " hit="
                        + str(blocked_info.get("hit_error_mm"))
                        + "mm"
                        + " lc="
                        + str(blocked_info.get("local_cloud_close_points"))
                        + "/"
                        + str(blocked_info.get("local_cloud_points"))
                    )
                else:
                    pointed_text = "pointing: none"

        else:
            self.smooth_base_2d = None
            self.smooth_tip_2d = None
            self.smooth_base_depth = None
            self.smooth_tip_depth = None
            self.smooth_direction_3d = None
            self.reset_raycast_cache()

        if pointed_text != "":
            cv2.putText(
                frame,
                pointed_text,
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        latest_frame = frame


async def video_feed(request):
    global latest_frame

    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame"
        }
    )

    await response.prepare(request)

    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode(
                ".jpg",
                latest_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            )

            if ret:
                frame_bytes = buffer.tobytes()

                await response.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )

        await asyncio.sleep(0.03)


async def index(request):
    html = """
    <html>
        <head>
            <title>Hand YOLO 3D Ray Stream</title>
        </head>
        <body>
            <h2>Hand + YOLO + Smoothed 3D Ray Stream</h2>
            <p>Blue boxes: YOLO objects</p>
            <p>Red line: smoothed 3D finger ray</p>
            <p>Red box: first object hit by 3D ray</p>
            <img src="/video" width="960">
        </body>
    </html>
    """

    return web.Response(
        text=html,
        content_type="text/html"
    )


def ros_thread():
    rclpy.init()
    node = HandDetectNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


def main():
    t = threading.Thread(target=ros_thread)
    t.daemon = True
    t.start()

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/video", video_feed)

    web.run_app(
        app,
        host="0.0.0.0",
        port=8081
    )


if __name__ == "__main__":
    main()
