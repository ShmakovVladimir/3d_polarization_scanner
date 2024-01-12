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
import glob


def find_ports():
    ports = glob.glob('/dev/ttyACM[0-9]*')

    res = []
    for port in ports:
        try:
            s = serial.Serial(port, 9600)
            return s
        except:
            pass


def threshold_black(res: dict, polarization_0, polarization_45, polarization_135, polarization_90, thr=20) -> dict:
    max_intensity = np.stack(
        [polarization_0, polarization_45, polarization_135, polarization_90], axis=2)
    max_intensity = np.min(max_intensity, axis=2)
    res['linear_polarizatioin_degree'] = (
        max_intensity > thr) * res['linear_polarizatioin_degree']


def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]


def main(n_value: float, model_name: str, camera_port: int):
    # port = serial.Serial('/dev/ttyACM0', 9600)
    port = find_ports()
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, f'lal/{model_name}')
    os.mkdir(path)
    angle_values = [0, 45, 90, 135]
    images = {a: 0 for a in angle_values}
    for angle in [0, 45, 90, 135]:
        cam = cv2.VideoCapture(camera_port)
        print(f"Фотографирую под углом {angle}")
        images[angle] = get_img(angle, port=port, cam=cam)
        cv2.imwrite(
            f'lal/{model_name}/polarization_{angle}.jpg', images[angle])
        cam.release()
    res = polarization(polarization_0=images[0],
                       polarization_90=images[90],
                       polarization_45=images[45],
                       polarization_135=images[135])
    # threshold_black(res, polarization_0=images[0],
    #                 polarization_90=images[90],
    #                 polarization_45=images[45],
    #                 polarization_135=images[135], thr=10)
    aop, dolp = res['angle_of_polarization'], res['linear_polarizatioin_degree']
    visualize_dolp_and_aop(
        dolp, aop, res['s0'], save_path=f'lal/{model_name}/')
    theta = polarization_degree_to_reflection_angle(dolp, n_value)
    normal_map = get_normal_map(aop, theta)
    low_dolp = res['linear_polarizatioin_degree'] < 1e-2
    normal_map[low_dolp] = np.zeros(3)
    cv2.imwrite(f'lal/{model_name}/normal_map.png', normal_map * 255)
    depth_map = normal_map_least_square_integration(normal_map)
    depth_map /= np.max(np.abs(depth_map))
    mask = dolp < 1e-2
    # depth_map[mask] = 0
    visualize_depth_map(gaussian_filter(np.abs(depth_map), sigma=3),
                        save_path=f'lal/{model_name}/')


if __name__ == '__main__':
    command_args = sys.argv
    n_index = command_args.index("--n") + 1
    model_name = command_args.index("--name") + 1
    camera_port = command_args.index("--port") + 1
    print(command_args[n_index], command_args[model_name],
          command_args[camera_port])
    main(float(command_args[n_index]), command_args[model_name], int(
        command_args[camera_port]))
