import math

def calculate_vertical_fov(fov_h_deg, aspect_ratio):
    fov_h_rad = math.radians(fov_h_deg)
    fov_v_rad = 2 * math.atan(math.tan(fov_h_rad / 2) / aspect_ratio)
    return math.degrees(fov_v_rad)    

import math

def calculate_fov(diagonal_fov_deg, resolution_width, resolution_height):
    diagonal_fov_rad = math.radians(diagonal_fov_deg)

    diagonal = math.sqrt(resolution_width**2 + resolution_height**2)
    ratio_w = resolution_width / diagonal
    ratio_h = resolution_height / diagonal

    fov_h_rad = 2 * math.atan(math.tan(diagonal_fov_rad / 2) * ratio_w)
    fov_v_rad = 2 * math.atan(math.tan(diagonal_fov_rad / 2) * ratio_h)

    return math.degrees(fov_h_rad), math.degrees(fov_v_rad)

def calculate_angular_offset(x, y, image_width, image_height, fov_h, fov_v):
    """
    Вычисляет угловое смещение объекта (x, y) от центра изображения
    по горизонтали и вертикали в градусах.

    :param x: координата X объекта в пикселях
    :param y: координата Y объекта в пикселях
    :param image_width: ширина изображения
    :param image_height: высота изображения
    :param fov_h: горизонтальный угол обзора (в градусах)
    :param fov_v: вертикальный угол обзора (в градусах)
    :return: (угол по горизонтали, угол по вертикали - в радианах)
    """
    cx = image_width / 2
    cy = image_height / 2

    dx = x - cx
    dy = y - cy

    angle_x = (dx / (image_width/2)) * (fov_h/2)
    angle_y = (dy / (image_height/2)) * (fov_v/2)

    return math.radians(angle_x), math.radians(angle_y)

# Пример использования:
if __name__ == "__main__":
    x = 1000  # координата объекта по X
    y = 400   # координата объекта по Y
    image_width = 1920
    image_height = 1080
    fov_h = 1.36   # поле зрения по горизонтали
    fov_v = 1.36   # поле зрения по вертикали

    angle_x, angle_y = calculate_angular_offset(x, y, image_width, image_height, fov_h, fov_v)

    print(f"Угловое смещение от центра:")
    print(f"По горизонтали: {angle_x:.2f}°")
    print(f"По вертикали: {angle_y:.2f}°")
