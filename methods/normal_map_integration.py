import numpy as np
from scipy.fft import fft2, fftfreq, ifft2
def normal_map_naive_integration(normal_map: np.ndarray) -> np.ndarray:
    """Наивный алгоритм нахождения карты высот по карте нормалей

    Args:
        normal_map (np.ndarray): Карта нормалей

    Returns:
        np.ndarray: Карта высот
    """
    print(normal_map.shape)
    depth_map = np.zeros(shape = (normal_map.shape[0], normal_map.shape[1]))
    #предположим что высота крайнего пикселя 0:
    depth_map[0, 0] = 0
    #интегрирование по первому столбцу
    for y in range(1, depth_map.shape[1]):
        depth_map[0, y] = depth_map[0, y - 1] - normal_map[0, y, 1]
    #интегрирование по оставшмся строкам
    for y in range(0, depth_map.shape[1]):
        for x in range(1, depth_map.shape[0]):
            depth_map[x, y] = depth_map[x - 1, y] - normal_map[x, y, 0]
    return depth_map

def normal_map_least_square_integration(normal_map: np.ndarray) -> np.ndarray:
    """Минимизация среднеквадратичной ошибки для вычисления карты высот

    Args:
        normal_map (np.ndarray): Карта нормалей

    Returns:
        np.ndarray: Карта высот
    """
    p, q = normal_map[:, :, 0], normal_map[:, :, 1] 
    u, v = fftfreq(p.shape[0]), fftfreq(p.shape[1])
    V, U = np.meshgrid(v, u)
    P_Forier_map = fft2(p)
    Q_Forier_map = fft2(q)
    Z_forier_map = (1j * U * P_Forier_map + 1j * V * Q_Forier_map) / (np.power(U, 2) + np.power(V, 2) + 1e-6)
    return ifft2(Z_forier_map).real 
