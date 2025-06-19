import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from placement import suggest_mixed_zones
from projection import project_rectangle_onto_wall

def draw_image_with_paintings(image_path, walls, save_path=None):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for wall in walls:
        zones = wall.get("zones", [])
        pictures = wall.get("pictures", [])

        for pic, quad in zip(pictures, zones):
            painting_img = cv2.imread(pic['path'])
            painting_img = cv2.cvtColor(painting_img, cv2.COLOR_BGR2RGB)

            dst_pts = np.array(quad, dtype=np.float32)
            if dst_pts.shape != (4, 2):
                continue

            # Вычисляем целевой размер изображения в пикселях
            w_target = int(np.linalg.norm(dst_pts[1] - dst_pts[0]))  # ширина по верху
            h_target = int(np.linalg.norm(dst_pts[3] - dst_pts[0]))  # высота по левому краю

            if w_target <= 0 or h_target <= 0:
                continue  # пропускаем некорректные зоны

            # Ресайзим картину под размеры на стене
            painting_img_resized = cv2.resize(painting_img, (w_target, h_target), interpolation=cv2.INTER_LANCZOS4)

            src_pts = np.array([[0, 0], [w_target, 0], [w_target, h_target], [0, h_target]], dtype=np.float32)

            # Применяем перспективное преобразование
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(painting_img_resized, M, (img.shape[1], img.shape[0]))

            mask = np.zeros_like(img_rgb, dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), (255, 255, 255))
            img_rgb = np.where(mask == 255, warped, img_rgb)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)
        print(f"💾 Сохранено изображение с картинами: {save_path}")

    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title("Размещение изображений картин по JSON")
    plt.show()


if __name__ == "__main__":
    config_path = input("Введите путь к JSON-файлу комнаты: ").strip()
    with open(config_path, 'r', encoding='utf-8') as f:
        project = json.load(f)

    image_path = project['image_path']
    walls = project['walls']

    picture_count = int(input("→ Сколько картин нужно разместить? "))
    sizes_cm = []
    for i in range(picture_count):
        print(f"\n📐 Картина №{i+1}")
        path = input("→ Путь к изображению: ").strip()
        w = float(input("→ Ширина (см): "))
        h = float(input("→ Высота (см): "))
        sizes_cm.append({"path": path, "width_cm": w, "height_cm": h})

    if picture_count > 1:
        min_spacing_cm = float(input("\n→ Минимальное расстояние между картинами (в см): "))
    else:
        min_spacing_cm = 15.0

    remaining = sizes_cm.copy()

    for wall in walls:
        if not remaining:
            break

        points = np.array(wall['points'], dtype=np.float32)
        real_w = wall['width_cm']
        real_h = wall['height_cm']

        scale_w = np.linalg.norm(points[3] - points[0]) / real_w
        scale_h = np.linalg.norm(points[1] - points[0]) / real_h

        openings = [np.array(o, dtype=np.float32) for o in wall.get('openings', []) if len(o) == 4]

        sizes_for_wall = [(p['width_cm'], p['height_cm']) for p in remaining]
        zones = suggest_mixed_zones(
            points,
            sizes_for_wall,
            scale_w,
            scale_h,
            min_spacing_cm=min_spacing_cm,
            openings_quads=openings,
            center_aligned=True
        )

        if zones:
            wall['zones'] = zones
            wall['pictures'] = remaining[:len(zones)]
            remaining = remaining[len(zones):]
        else:
            wall['zones'] = []
            wall['pictures'] = []

    # === Центровка одиночных картин только если на стене нет проёмов ===
    for wall in walls:
        zones = wall.get("zones", [])
        pictures = wall.get("pictures", [])
        openings = wall.get("openings", [])
        if len(zones) == 1 and len(pictures) == 1 and not openings:
            points = np.array(wall['points'], dtype=np.float32)
            real_w = wall['width_cm']
            real_h = wall['height_cm']
            scale_w = np.linalg.norm(points[3] - points[0]) / real_w
            scale_h = np.linalg.norm(points[1] - points[0]) / real_h

            pic = pictures[0]
            wall_width_px = np.linalg.norm(points[3] - points[0])
            pic_width_px = pic['width_cm'] * scale_w
            x_offset = (wall_width_px - pic_width_px) / 2

            new_zone = project_rectangle_onto_wall(
                points,
                pic['width_cm'],
                pic['height_cm'],
                scale_w,
                scale_h,
                x_offset=x_offset
            )
            wall['zones'][0] = new_zone

    if any(w.get('zones') for w in walls):
        print("✅ Картины успешно размещены.")
        for wall in walls:
            print(f"📐 Стена '{wall['name']}', зон: {len(wall.get('zones', []))}, картин: {len(wall.get('pictures', []))}")
        output_path = os.path.join("done", os.path.basename(image_path))
        draw_image_with_paintings(image_path, walls, save_path=output_path)
    else:
        print("❌ Недостаточно места для размещения картин.")
