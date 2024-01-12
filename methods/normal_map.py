import numpy as np


def get_normal_map(aop: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Создание карты нормалей по двум углам

    Args:
        aop (np.ndarray): Угол поляризации
        theta (np.ndarray): азимутальный угол

    Returns:
        np.ndarray: Карта нормалей вида (p(x, y), q(x, y), 1)
    """
    normal_map = np.array([np.sin(theta) * np.cos(aop) / (np.cos(theta) + 1e-2),
                           np.sin(theta * np.sin(aop)) /
                           (np.cos(theta) + 1e-2),
                           np.ones_like(theta)]).T
    normal_map = np.swapaxes(normal_map, 0, 1)
    normal_map /= np.linalg.norm(normal_map, ord=2, keepdims=True, axis=2)
    normal_map = (np.cos(theta) > 1e-3)[:, :, np.newaxis] * normal_map
    return normal_map
