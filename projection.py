import numpy as np
import cv2

def project_rectangle_onto_wall(wall_pts, picture_width_cm, picture_height_cm, scale_w, scale_h, x_offset=0):
    """
    wall_pts — 4 точки стены: [нижний левый, верхний левый, верхний правый, нижний правый]
    x_offset — сдвиг начала размещения по ширине стены в пикселях
    """

    # Размеры картины в пикселях
    picture_width_px = picture_width_cm * scale_w
    picture_height_px = picture_height_cm * scale_h

    # Размеры стены в пикселях
    wall_width_px = np.linalg.norm(wall_pts[3] - wall_pts[0])
    wall_height_px = np.linalg.norm(wall_pts[1] - wall_pts[0])

    # Верхний левый угол картины в локальной системе
    top_left_x = x_offset
    top_left_y = (wall_height_px - picture_height_px) / 2

    # Координаты картины в локальной системе стены (правильный порядок)
    local_picture = np.array([
        [top_left_x, top_left_y],                             # верхний левый
        [top_left_x + picture_width_px, top_left_y],          # верхний правый
        [top_left_x + picture_width_px, top_left_y + picture_height_px],  # нижний правый
        [top_left_x, top_left_y + picture_height_px]          # нижний левый
    ], dtype=np.float32)

    # Стена в локальной системе координат
    src_rect = np.array([
        [0, wall_height_px],     # нижний левый
        [0, 0],                  # верхний левый
        [wall_width_px, 0],     # верхний правый
        [wall_width_px, wall_height_px]  # нижний правый
    ], dtype=np.float32)

    # Перспективная трансформация
    M = cv2.getPerspectiveTransform(src_rect, wall_pts.astype(np.float32))
    projected_quad = cv2.perspectiveTransform(local_picture[None, :, :], M)[0]

    return projected_quad
