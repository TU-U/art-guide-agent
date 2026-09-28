import numpy as np

from config import (
    LOCAL_CLOUD_DEPTH_BAND_MM,
    LOCAL_CLOUD_HIT_THRESHOLD_MM,
    LOCAL_CLOUD_MIN_POINTS,
    LOCAL_CLOUD_RADIUS,
    RAY_MAX_MM,
    RAY_START_OFFSET_MM,
    RAY_STEP_MM,
    RERAYCAST_ANGLE_THRESHOLD_DEG,
    RERAYCAST_MOVE_THRESHOLD_MM,
)
from geometry import point_in_box, project_3d_to_pixel


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


def raycast_pointed_object(tip_3d, direction_3d, objects, depth, fx, fy, cx, cy, frame_w, frame_h):
    if tip_3d is None or direction_3d is None:
        return None, None, None, []

    direction_len = np.linalg.norm(direction_3d)

    if direction_len < 0.001:
        return None, None, None, []

    direction_3d = direction_3d / direction_len
    ray_pixels = []

    tip_pixel = project_3d_to_pixel(
        tip_3d,
        fx,
        fy,
        cx,
        cy
    )

    if tip_pixel is not None:
        tip_u, tip_v = tip_pixel

        if 0 <= tip_u < frame_w and 0 <= tip_v < frame_h:
            ray_pixels.append((tip_u, tip_v))

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

        ray_pixels.append((u, v))

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

            return None, (u, v), blocked_info, ray_pixels

        hit_obj["hit_u"] = u
        hit_obj["hit_v"] = v
        hit_obj["hit_depth_mm"] = real_depth
        hit_obj["ray_depth_mm"] = int(point_3d[2])
        hit_obj["ray_len_mm"] = ray_len
        hit_obj["hit_error_mm"] = round(float(hit_error), 1)
        hit_obj["hit_method"] = "ray_step_local_cloud"
        hit_obj["local_cloud_points"] = local_point_count
        hit_obj["local_cloud_close_points"] = close_count

        return hit_obj, (u, v), None, ray_pixels

    return None, None, None, ray_pixels
