import numpy as np

stokes_parametrs = [[0.5, 0.5, 0.5, 0.5],
                    [1, -1, 0, 0],
                    [0, 0, 1, -1]]
def polarization(polarization_0: np.ndarray,
                 polarization_90: np.ndarray,
                 polarization_45: np.ndarray,
                 polarization_135: np.ndarray) -> dict:
    """Рассчитавает степень и угол поляризации каждого пикселя

    Args:
        polarization_0 (np.ndarray): фотография через поляроид 
        polarization_90 (np.ndarray): фотография через поляроид, скрещенный к исходному
        polarization_45 (np.ndarray): фотография через поляроид с осью под 45 к исходному
        polarization_135 (np.ndarray): фотография через поляроид с осью под 135 к исходному

    Returns:
        dict: Степень поляризации и угол поляризации каждого из пикселей
    """
    global stokes_parametrs
    if len(polarization_0.shape) - 2:
        pass
    intensity_matrix = np.array([polarization_0,  
                                 polarization_90,  
                                 polarization_45,  
                                 polarization_135],
                                 dtype = np.float64)
    s = np.tensordot(stokes_parametrs, intensity_matrix, axes = 1) 
    dolp = np.sqrt(np.power(s[1], 2) + np.power(s[2], 2)) / (s[0] + 1e-4)
    aop = 0.5 * np.angle(s[1] + 1j * s[2])
    return {'linear_polarizatioin_degree': dolp,
            'angle_of_polarization': aop,
            's0': s[0],
            's1': s[1],
            's2': s[2]}

def polarization_degree_to_reflection_angle(dolp: np.ndarray, n: float = 1.7) -> np.ndarray:
    """Функция, обратная к `diffuse_polarization_degree`. Рассчет угла отражения по степени поляризации.

    Args:
        dolp (np.ndarray): Степень поляризации
        n (float, optional): Показатель преломления. Defaults to 1.7.

    Returns:
        np.ndarray: Угол отражения
    """
    temp = np.sqrt((2 * dolp + 2 * n**2 * dolp - 2 * n**2 + n**4 + dolp**2 + 4 * n**2 * dolp**2 - n**4 * dolp**2 - 4 * n**3 * dolp * np.sqrt(-(dolp - 1) * (dolp + 1))) + 1) / np.sqrt(n**4 * dolp**2 + 2 * n**4 * dolp + n**4 + 6 * n**2 * dolp**2 + 4 * n**2 * dolp - 2 * n**2 + dolp**2 + 2 * dolp + 1)
    temp = np.minimum(np.real(temp), 1)
    theta_val = np.arccos(temp)
    return theta_val