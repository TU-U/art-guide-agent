import numpy as np


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
