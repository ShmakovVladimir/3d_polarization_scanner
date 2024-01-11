import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def visualize_dolp_and_aop(dolp: np.ndarray, aop: np.ndarray, s0: np.ndarray, save_path: str) -> None:
    """Визуализация распределения степени и угла поляризации на фотографии

    Args:
        dolp (np.ndarray): Степень линейной поляризации
        aop (np.ndarray): Угол линейной поляризации
        s0 (np.ndarray): <<Нулевой>> параметр Стокса
        save_path (str): Путь для сохранения результата
    """
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), dpi=200)
    ax[0].imshow(s0, cmap='greys')
    ax[0].set_title("$S_{0}$", fontsize=16)
    im = ax[1].imshow(aop, cmap='rainbow')
    plt.colorbar(im, ax=ax[1])
    ax[1].set_title("Угол поляризации", fontsize=16)
    im = ax[2].imshow(dolp, cmap='jet')
    plt.colorbar(im, ax=ax[2])
    ax[2].set_title("Степень поляризации", fontsize=16)
    plt.savefig(save_path + "dolp_aop.png")


def visualize_depth_map(depth_map: np.ndarray, save_path: str) -> None:
    x, y = np.arange(depth_map.shape[0]), np.arange(depth_map.shape[1])
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(12, 12), dpi=200)
    angles = [0, 20, 60, 140]
    for i, a in enumerate(angles):
        ax = fig.add_axes(int(f'22{i + 1}'), projection='3d')
        ax.plot_surface(X, Y, depth_map.T, antialiased=True, cmap=cm.coolwarm)
        ax.view_init(30, a, 0)
        ax.set_box_aspect([1, 1, 0.5])
        ax.set_zlim(-1, 1)
    fig.suptitle("Восстановленная карта высот", fontsize=16)
    fig.tight_layout()
    plt.savefig(save_path + 'depth_map_3d.png')
    plt.show()
