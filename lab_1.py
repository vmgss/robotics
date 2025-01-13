import cv2
import numpy as np

# Список путей к изображениям
image_paths = [
    "original_pics/pic.png",  # Укажи путь к первой картинке
    "original_pics/apples.png",
    "original_pics/circles.png"   # Укажи путь к третьей картинке
]

# Обработка всех изображений
for image_path in image_paths:
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Преобразование в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение размытия для устранения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Применение метода Canny для выделения контуров
    edges = cv2.Canny(blurred, 100, 200)

    # Поиск контуров
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Инициализация переменных
    largest_contour = None
    smallest_contour = None
    max_area = 0
    min_area = float("inf")
    centers = []

    # Обработка контуров
    for contour in contours:
        # Вычисление площади
        area = cv2.contourArea(contour)
        
        # Определение центра масс
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        centers.append((cx, cy))
        
        # Проверка на максимальный и минимальный объект
        if area > max_area:
            max_area = area
            largest_contour = (contour, (cx, cy))
        if area < min_area:
            min_area = area
            smallest_contour = (contour, (cx, cy))

    # Отображение контуров и центров
    for contour, center in zip(contours, centers):
        # Отображение контуров
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)  # Зеленые контуры
        # Отображение точки на центре объекта
        cv2.circle(image, center, 5, (0, 0, 255), -1)  # Красная точка на центре

    # Подпись самого большого и самого маленького объекта с увеличенным текстом и черным цветом
    if largest_contour:
        largest_text_position = (largest_contour[1][0], largest_contour[1][1] - 10)
        cv2.putText(image, "Largest", largest_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)  # Черный текст

    if smallest_contour:
        smallest_text_position = (smallest_contour[1][0], smallest_contour[1][1] - 10)
        cv2.putText(image, "Smallest", smallest_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)  # Черный текст

    # Сохранение результатов для каждого изображения
    output_path = f"results/result_{image_path.split('/')[-1]}"  # Имя файла с результатом
    cv2.imwrite(output_path, image)

    # Количество объектов
    print(f"Processed {image_path}:")
    print(f"Number of objects: {len(contours)}")
    if largest_contour:
        print(f"Center of the largest object: {largest_contour[1]}")
    if smallest_contour:
        print(f"Center of the smallest object: {smallest_contour[1]}")
    print(f"Processed image saved to: {output_path}")
