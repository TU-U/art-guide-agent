import time

import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

import state as st
from config import (
    LATEST_FRAME_PATH,
    POINT_CONFIRM_FRAMES,
    PROCESS_W,
    SMOOTH_ALPHA_2D,
    SMOOTH_ALPHA_DEPTH,
    SMOOTH_ALPHA_DIR,
    TRIGGER_DISTANCE_MM,
    YOLO_INTERVAL,
)
from geometry import ema_smooth, get_depth_near_point, pixel_to_3d
from output import draw_model_frame, make_pointing_result, save_pointing_images
from raycast import raycast_pointed_object, should_reraycast


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
        self.point_candidate_id = None
        self.point_candidate_count = 0
        self.confirmed_pointed_obj = None
        self.confirmed_hit_pixel = None

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

    def reset_point_confirmation(self):
        self.point_candidate_id = None
        self.point_candidate_count = 0
        self.confirmed_pointed_obj = None
        self.confirmed_hit_pixel = None

    def update_point_confirmation(self, candidate_obj, hit_pixel):
        if candidate_obj is None:
            self.reset_point_confirmation()
            return None, None

        candidate_id = candidate_obj.get("object_id")

        if candidate_id == self.point_candidate_id:
            self.point_candidate_count += 1
        else:
            self.point_candidate_id = candidate_id
            self.point_candidate_count = 1
            self.confirmed_pointed_obj = None
            self.confirmed_hit_pixel = None

        if self.point_candidate_count >= POINT_CONFIRM_FRAMES:
            self.confirmed_pointed_obj = candidate_obj
            self.confirmed_hit_pixel = hit_pixel
            return self.confirmed_pointed_obj, self.confirmed_hit_pixel

        return None, None

    def callback(self, color_msg, depth_msg):
        frame = st.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        raw_frame = frame.copy()

        depth = st.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

        orig_h, orig_w = frame.shape[:2]

        if self.fx is None:
            st.latest_frame = frame
            st.latest_pointing_result = make_pointing_result(
                False,
                time.time(),
                LATEST_FRAME_PATH
            )
            return

        now = time.time()

        if now - st.last_yolo_time > YOLO_INTERVAL:
            st.last_yolo_time = now

            yolo_results = st.yolo_model(
                raw_frame,
                imgsz=320,
                verbose=False
            )

            current_yolo_objects = []

            for result in yolo_results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_id == 0:
                        continue

                    if conf < 0.35:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    object_id = len(current_yolo_objects) + 1
                    current_yolo_objects.append({
                        "object_id": object_id,
                        "class_id": cls_id,
                        "class_name": st.yolo_model.names[cls_id],
                        "confidence": conf,
                        "box": [
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2)
                        ]
                    })

            st.last_yolo_objects = current_yolo_objects

        for obj in st.last_yolo_objects:
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
        result = st.hands.process(rgb)

        pointed_text = ""
        pointing_result_updated = False

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
                st.dist_history.append(avg_tip_depth)

                avg_tip_depth = int(
                    sum(st.dist_history) / len(st.dist_history)
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

            st.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                st.mp_hands.HAND_CONNECTIONS
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

                self.reset_raycast_cache()
                self.reset_point_confirmation()
                st.latest_pointing_result = make_pointing_result(
                    False,
                    time.time(),
                    LATEST_FRAME_PATH
                )

                st.latest_frame = frame
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

                self.reset_raycast_cache()
                self.reset_point_confirmation()
                st.latest_pointing_result = make_pointing_result(
                    False,
                    time.time(),
                    LATEST_FRAME_PATH
                )

                st.latest_frame = frame
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
                candidate_obj = None
                hit_pixel = None
                blocked_info = None
                ray_pixels = []
                self.reset_raycast_cache()
            elif effective_tip_3d is not None and effective_direction_3d is not None:
                if reraycast_needed:
                    self.last_raycast_tip_3d = effective_tip_3d.copy()
                    self.last_raycast_direction_3d = effective_direction_3d.copy()

                candidate_obj, hit_pixel, blocked_info, ray_pixels = raycast_pointed_object(
                    effective_tip_3d,
                    effective_direction_3d,
                    st.last_yolo_objects,
                    depth,
                    self.fx,
                    self.fy,
                    self.cx,
                    self.cy,
                    orig_w,
                    orig_h
                )
            else:
                candidate_obj = None
                hit_pixel = None
                blocked_info = None
                ray_pixels = []

            if hit_pixel is not None and (
                len(ray_pixels) == 0 or ray_pixels[-1] != hit_pixel
            ):
                ray_pixels.append(hit_pixel)

            for index in range(1, len(ray_pixels)):
                cv2.line(
                    frame,
                    ray_pixels[index - 1],
                    ray_pixels[index],
                    (0, 0, 255),
                    2
                )

            pointed_obj, confirmed_hit_pixel = self.update_point_confirmation(
                candidate_obj,
                hit_pixel
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

                pointed_label = (
                    pointed_obj["class_name"]
                    + " "
                    + str(round(pointed_obj.get("confidence", 0), 2))
                )
                cv2.putText(
                    frame,
                    pointed_label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                if candidate_obj is not None and hit_pixel is not None:
                    cv2.circle(
                        frame,
                        hit_pixel,
                        8,
                        (0, 0, 255),
                        -1
                    )

                model_frame = draw_model_frame(raw_frame, st.last_yolo_objects, pointed_obj)
                frame_path = save_pointing_images(model_frame)
                st.latest_pointing_result = make_pointing_result(
                    True,
                    time.time(),
                    frame_path,
                    pointed_obj
                )
                pointing_result_updated = True

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
                if candidate_obj is not None:
                    pointed_text = (
                        "candidate: "
                        + candidate_obj["class_name"]
                        + " count="
                        + str(self.point_candidate_count)
                        + "/"
                        + str(POINT_CONFIRM_FRAMES)
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
            self.reset_point_confirmation()

        if not pointing_result_updated:
            st.latest_pointing_result = make_pointing_result(
                False,
                time.time(),
                LATEST_FRAME_PATH
            )

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

        st.latest_frame = frame
