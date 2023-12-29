import time
import cv2
import numpy as np
def get_img(polarization_angle: int, port, cam) -> np.ndarray:
    """Сделать фотографию с определённым разрешенным направлением поляроида

    Args:
        polarization_angle (int): Разрешённое направление поляроида
        port (_type_): Serial port для общения с поворотным устройством
        cam (_type_): камера

    Returns:
        np.ndarray: чб фотография
    """
    polarization_angle_to_angle = {0: 15, 45: 120, 135: 65, 90: 180}
    port.write(bytes(str(polarization_angle_to_angle[polarization_angle]), 'utf-8'))
    time.sleep(2)
    _, frame = cam.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)