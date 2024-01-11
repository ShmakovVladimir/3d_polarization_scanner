from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import os
import serial
import sys
from methods.normal_map_integration import *
from methods.photos_to_stokes import *
from methods.experimental_setup_control import *
from methods.result_visualization import *
from methods.normal_map import *


def main(n_value: float, model_name: str, camera_port: int):
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, f'results/{model_name}')
    os.mkdir(path)
    port = serial.Serial('/dev/ttyACM0', 9600)
    cam = cv2.VideoCapture(camera_port)
    angle_values = [0, 45, 90, 135]
    images = {a: 0 for a in angle_values}
    for angle in [0, 45, 90, 135]:
        images[angle] = get_img(angle, port=port, cam=cam)
        cv2.imwrite(
            f'results/{model_name}/polarization_{angle}.jpg', images[angle])
        print(f"Фотографирую под углом {angle}")
    res = polarization(images[0], images[90], images[45], images[135])
    aop, dolp = res['angle_of_polarization'], res['linear_polarizatioin_degree']
    visualize_dolp_and_aop(
        dolp, aop, res['s0'], save_path=f'results/{model_name}/')
    theta = polarization_degree_to_reflection_angle(dolp, n_value)
    normal_map = get_normal_map(aop, theta)
    low_dolp = res['linear_polarizatioin_degree'] < 1e-2
    normal_map[low_dolp] = np.zeros(3)
    cv2.imwrite(f'results/{model_name}/normal_map.png', normal_map * 255)
    depth_map = normal_map_least_square_integration(normal_map)
    depth_map /= np.max(np.abs(depth_map))
    mask = dolp < 1e-2
    depth_map[mask] = 0
    visualize_depth_map(gaussian_filter(np.abs(depth_map), sigma=3),
                        save_path=f'results/{model_name}/')


if __name__ == '__main__':
    command_args = sys.argv
    n_index = command_args.index("--n") + 1
    model_name = command_args.index("--name") + 1
    camera_port = command_args.index("--port") + 1
    print(command_args[n_index], command_args[model_name],
          command_args[camera_port])
    main(float(command_args[n_index]), command_args[model_name], int(
        command_args[camera_port]))
