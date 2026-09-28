import rclpy
import threading
import cv2
import mediapipe as mp
import numpy as np
import asyncio
import time

from collections import deque
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

RAY_STEP_MM = 20
RAY_MAX_MM = 4000
HIT_3D_THRESHOLD_MM = 70
RAY_START_OFFSET_MM = 120
HIT_REQUIRED_COUNT = 2

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


def raycast_pointed_object(tip_3d, direction_3d, objects, depth, fx, fy, cx, cy, frame_w, frame_h):
    if tip_3d is None or direction_3d is None:
        return None, None

    direction_len = np.linalg.norm(direction_3d)

    if direction_len < 0.001:
        return None, None

    direction_3d = direction_3d / direction_len

    last_hit_obj = None
    same_hit_count = 0

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

        real_depth = get_depth_near_point(depth, u, v, radius=2)

        if real_depth is None:
            last_hit_obj = None
            same_hit_count = 0
            continue

        surface_3d = pixel_to_3d(
            u,
            v,
            real_depth,
            fx,
            fy,
            cx,
            cy
        )

        if surface_3d is None:
            last_hit_obj = None
            same_hit_count = 0
            continue

        hit_error = np.linalg.norm(surface_3d - point_3d)

        if hit_error > HIT_3D_THRESHOLD_MM:
            last_hit_obj = None
            same_hit_count = 0
            continue

        hit_obj = None

        for obj in objects:
            if point_in_box(u, v, obj["box"]):
                hit_obj = obj
                break

        if hit_obj is None:
            last_hit_obj = None
            same_hit_count = 0
            continue

        if last_hit_obj is not None and hit_obj["class_id"] == last_hit_obj["class_id"]:
            same_hit_count += 1
        else:
            last_hit_obj = hit_obj
            same_hit_count = 1

        if same_hit_count >= HIT_REQUIRED_COUNT:
            hit_obj["hit_u"] = u
            hit_obj["hit_v"] = v
            hit_obj["hit_depth_mm"] = real_depth
            hit_obj["ray_depth_mm"] = int(point_3d[2])
            hit_obj["ray_len_mm"] = ray_len
            hit_obj["hit_error_mm"] = round(float(hit_error), 1)

            return hit_obj, (u, v)

    return None, None


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

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/ob_camera_head/color/camera_info",
            self.camera_info_callback,
            10
        )

        color_sub = Subscriber(
            self,
            Image,
            "/ob_camera_head/color/image_raw"
        )

        depth_sub = Subscriber(
            self,
            Image,
            "/ob_camera_head/depth/image_raw"
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

            last_yolo_objects = []

            for result in yolo_results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_id == 0:
                        continue

                    if conf < 0.35:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    obj = {
                        "class_id": cls_id,
                        "class_name": yolo_model.names[cls_id],
                        "confidence": conf,
                        "box": [
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2)
                        ]
                    }

                    last_yolo_objects.append(obj)

        for obj in last_yolo_objects:
            x1, y1, x2, y2 = obj["box"]
            label = obj["class_name"] + " " + str(round(obj["confidence"], 2))

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

            if tip_3d is not None and direction_3d is not None:
                last_pixel = None

                for ray_len in range(0, 1200, 60):
                    p3d = tip_3d + direction_3d * ray_len

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

            pointed_obj, hit_pixel = raycast_pointed_object(
                tip_3d,
                direction_3d,
                last_yolo_objects,
                depth,
                self.fx,
                self.fy,
                self.cx,
                self.cy,
                orig_w,
                orig_h
            )

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

                pointed_text = (
                    "pointing: "
                    + pointed_obj["class_name"]
                    + " hit="
                    + str(pointed_obj.get("hit_error_mm"))
                    + "mm"
                    + " depth="
                    + str(pointed_obj.get("hit_depth_mm"))
                    + "mm"
                )

            else:
                pointed_text = "pointing: none"

        else:
            self.smooth_base_2d = None
            self.smooth_tip_2d = None
            self.smooth_base_depth = None
            self.smooth_tip_depth = None
            self.smooth_direction_3d = None

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