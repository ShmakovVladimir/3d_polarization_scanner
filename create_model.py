from methods.normal_map_integration import *
from methods.photos_to_stokes import *
from methods.experimental_setup_control import *
import sys
import serial
import os
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def main(n_value: float, model_name: str, camera_port: int):
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, f'results/{model_name}') 
    os.mkdir(path) 
    port = serial.Serial('/dev/ttyACM0', 9600)
    cam = cv2.VideoCapture(camera_port)
    angle_values = [0, 45, 90, 135]
    images = {a: 0 for a in angle_values}
    for angle in [0, 45, 90, 135]:
        images[angle] = get_img(angle, port = port, cam = cam)
        cv2.imwrite(f'results/{model_name}/polarization_{angle}.jpg', images[angle])
    res = polarization(images[0], images[90], images[45], images[135])
    aop, dolp = res['angle_of_polarization'], res['linear_polarizatioin_degree']
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 200)
    im = ax[0].imshow(aop, cmap = 'rainbow')
    plt.colorbar(im, ax = ax[0])
    im = ax[1].imshow(dolp, cmap = 'jet')
    plt.colorbar(im, ax = ax[1])
    plt.savefig("dolp_aop.png")
    theta = polarization_degree_to_reflection_angle(dolp, n_value)
    normal_map = np.array([np.tan(theta) * np.cos(aop),
                        np.tan(theta * np.sin(aop)),
                        np.ones_like(theta)]).T
    normal_map = np.swapaxes(normal_map, 0, 1)
    normal_map /= np.linalg.norm(normal_map, ord = 2, keepdims = True, axis = 2)
    normal_map = (np.cos(theta) > 1e-2)[:, :, np.newaxis] * normal_map
    low_dolp = res['linear_polarizatioin_degree'] < 1e-2
    normal_map[low_dolp] = np.zeros(3) 
    cv2.imwrite(f'results/{model_name}/normal_map.png', normal_map * 255)
    depth_map = normal_map_least_square_integration(normal_map)
    depth_map /= np.max(np.abs(depth_map))
    mask = dolp < 1e-5
    x, y = np.arange(depth_map.shape[0]), np.arange(depth_map.shape[1])
    X, Y = np.meshgrid(x, y)
    Z = np.copy(depth_map)
    Z[mask] = 0
    Z = gaussian_filter(Z, sigma = 0)
    fig = plt.figure(figsize = (12, 12), dpi = 200)
    angles = [0, 20, 60, 140]
    for i, a in enumerate(angles): 
        ax = fig.add_axes(int(f'22{i + 1}'), projection = '3d')
        ax.plot_surface(X, Y, Z.T, antialiased = True, cmap = cm.coolwarm)
        ax.view_init(30, a, 0)
        ax.set_box_aspect([1, 1, 0.5])
        ax.set_zlim(-1, 1)

    fig.suptitle("Восстановленная карта высот", fontsize = 16)
    fig.tight_layout()
    plt.savefig(f'results/{model_name}/3d_model.png')
    plt.show()
    
        

if __name__ == '__main__':
    command_args = sys.argv
    n_index = command_args.index("--n") + 1
    model_name = command_args.index("--name") + 1
    camera_port = command_args.index("--port") + 1
    print(command_args[n_index], command_args[model_name], command_args[camera_port])
    main(float(command_args[n_index]), command_args[model_name], int(command_args[camera_port]))
    
