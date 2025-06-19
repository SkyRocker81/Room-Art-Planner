import numpy as np
import cv2
from shapely.geometry import Polygon
from projection import project_rectangle_onto_wall


def quad_too_close(quad1, quad2, min_distance_px):
    for pt1 in quad1:
        for pt2 in quad2:
            if np.linalg.norm(pt1 - pt2) < min_distance_px:
                return True
    return False


def suggest_mixed_zones(
    wall_pts,
    sizes_cm,
    scale_w,
    scale_h,
    min_spacing_cm=15,
    openings_quads=None,
    center_aligned=False
):
    if openings_quads is None:
        openings_quads = []

    wall_width_px = np.linalg.norm(wall_pts[3] - wall_pts[0])
    wall_height_px = np.linalg.norm(wall_pts[1] - wall_pts[0])
    min_spacing_px = min_spacing_cm * scale_w
    cy = wall_height_px / 2  # Центр по высоте

    # Преобразования
    src_rect = np.array([
        [0, wall_height_px],
        [0, 0],
        [wall_width_px, 0],
        [wall_width_px, wall_height_px]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_rect, wall_pts.astype(np.float32))
    M_inv = cv2.getPerspectiveTransform(wall_pts.astype(np.float32), src_rect)

    def project_quad(cx, width_px, height_px):
        local_quad = np.array([
            [cx - width_px / 2, cy - height_px / 2],
            [cx + width_px / 2, cy - height_px / 2],
            [cx + width_px / 2, cy + height_px / 2],
            [cx - width_px / 2, cy + height_px / 2]
        ], dtype=np.float32)
        return cv2.perspectiveTransform(local_quad[None, :, :], M)[0]

    def try_place_in_range(x_start, x_end, sizes_px, openings_local=None):
        total_width = sum(w for w, _ in sizes_px) + min_spacing_px * (len(sizes_px) - 1)
        if x_end - x_start < total_width:
            return None

        cx = x_start + (x_end - x_start - total_width) / 2 + sizes_px[0][0] / 2
        result = []
        for i, (w_px, h_px) in enumerate(sizes_px):
            quad = project_quad(cx, w_px, h_px)
            if openings_local and any(quad_too_close(quad, o, min_spacing_px) for o in openings_local):
                return None
            result.append(quad)
            if i < len(sizes_px) - 1:
                cx += (w_px + sizes_px[i + 1][0]) / 2 + min_spacing_px
        return result

    sizes_px = [(w * scale_w, h * scale_h) for (w, h) in sizes_cm]

    if not openings_quads:
        for i in range(len(sizes_px), 0, -1):
            group = sizes_px[:i]
            total = sum(w for w, _ in group) + min_spacing_px * (len(group) - 1)
            if total > wall_width_px - 2 * min_spacing_px:
                continue
            return try_place_in_range(min_spacing_px, wall_width_px - min_spacing_px, group)

        return None

    # === Если есть проёмы ===
    openings_local = [
        cv2.perspectiveTransform(np.array([opening], dtype=np.float32), M_inv)[0]
        for opening in openings_quads
    ]

    x_ranges = []
    for local in openings_local:
        x_coords = [pt[0] for pt in local]
        x_min, x_max = min(x_coords), max(x_coords)
        x_ranges.append((x_min, x_max))

    x_min_all = min(r[0] for r in x_ranges)
    x_max_all = max(r[1] for r in x_ranges)

    left_range = (min_spacing_px, x_min_all - min_spacing_px)
    right_range = (x_max_all + min_spacing_px, wall_width_px - min_spacing_px)

    for i in range(len(sizes_px), 0, -1):
        group = sizes_px[:i]

        # слева
        result = try_place_in_range(left_range[0], left_range[1], group, openings_local)
        if result:
            return result

        # справа
        result = try_place_in_range(right_range[0], right_range[1], group, openings_local)
        if result:
            return result

        # разделение между сторонами
        for split in range(1, len(group)):
            left_part = try_place_in_range(left_range[0], left_range[1], group[:split], openings_local)
            right_part = try_place_in_range(right_range[0], right_range[1], group[split:], openings_local)
            if left_part and right_part:
                return left_part + right_part

    return None
