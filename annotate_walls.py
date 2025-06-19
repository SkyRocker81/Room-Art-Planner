import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import numpy as np
import os

def annotate_walls(image_path, output_json_path):
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)

    walls = []
    current_points = []
    in_opening_mode = False  # 🧠 флаг, отключающий клики при разметке проёма

    def onclick(event):
        nonlocal current_points, in_opening_mode
        if in_opening_mode:
            return  # 🛑 игнорируем клики, пока размечается проём
        if event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            current_points.append([x, y])
            ax.plot(x, y, 'ro')
            ax.text(x + 5, y - 10, f'{len(current_points)}', color='red')
            fig.canvas.draw()

            if len(current_points) == 4:
                print("\n✅ 4 точки выбраны. Теперь введите параметры стены:")
                name = input("→ Название стены: ")
                width = float(input("→ Ширина стены (см): "))
                height = float(input("→ Высота стены (см): "))

                wall_openings = []
                add_opening = input("Добавить проёмы на этой стене? (y/n): ").lower().strip()
                while add_opening == 'y':
                    print("\n🔳 Укажи 4 точки проёма:")
                    opening_pts = []
                    count = 1

                    in_opening_mode = True  # ✅ включаем флаг, чтобы остановить onclick

                    while len(opening_pts) < 4:
                        print(f"    🔹 Кликни точку {count}/4")
                        pts = plt.ginput(1, timeout=-1)
                        if pts:
                            px, py = int(pts[0][0]), int(pts[0][1])
                            opening_pts.append([px, py])
                            ax.plot(px, py, 'bo')
                            ax.text(px + 5, py - 5, f'{count}', color='blue')
                            fig.canvas.draw()
                            count += 1

                    in_opening_mode = False  # 🔄 отключаем флаг

                    wall_openings.append(opening_pts)
                    print(f"✅ Проём добавлен: {opening_pts}")
                    add_opening = input("Добавить ещё проём? (y/n): ").lower().strip()

                walls.append({
                    "name": name,
                    "points": current_points.copy(),
                    "width_cm": width,
                    "height_cm": height,
                    "openings": wall_openings
                })
                print(f"🔸 Стена '{name}' добавлена. Вы можете кликать следующую.")
                current_points.clear()

    def onkey(event):
        if event.key == 'enter':
            print("\n💾 Разметка завершена. Сохраняем JSON...")
            data = {
                "image_path": image_path,  # ✅ сохраняем путь как есть
                "walls": walls
            }
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ Сохранено в {output_json_path}")
            plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.title("Кликни 4 точки на каждую стену. Нажми Enter, чтобы завершить.")
    plt.show()

# Пример запуска
if __name__ == "__main__":
    annotate_walls("rooms/room2.png", "room_configs/room2.json")
