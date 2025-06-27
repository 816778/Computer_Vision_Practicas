import cv2
import glob
import numpy as np
import argparse
import json
import os

import utils.camera_calibration as cam_calib
import utils.plot_utils as plot_utils
import utils.projection as projection
import utils.triangulation as triangulation
import utils.bundle as bundle
from utils.reconstruction import Reconstruccion3D


def draw_3d_points(Xw, X_w_opt, cameras, cameras_opt, T_wc1, T_wc2, T_wc1_opt, T_wc2_opt, K, x1Data, x2Data, params):
    plot_utils.plot3DPoints(Xw, cameras, world_ref=False)
    x1_no_opt = projection.project_points(K, T_wc1, Xw) # 3xN
    x2_no_opt = projection.project_points(K, T_wc2, Xw) # 3xN

    plot_utils.plot3DPoints(X_w_opt, cameras_opt, world_ref=False)

    x1_p_opt = projection.project_points(K, T_wc1_opt, X_w_opt)
    x2_p_opt = projection.project_points(K, T_wc2_opt, X_w_opt)

    image1 = cv2.imread(params['path_images'][0][0])
    image2 = cv2.imread(params['path_images'][0][1])

    plot_utils.visualize_projection_2(image1, x1Data, x1_no_opt, x1_p_opt, 'Image 1')
    plot_utils.visualize_projection_2(image2, x2Data, x2_no_opt, x2_p_opt, 'Image 2')

def process_matches(params, K, draw_graphic=False):

    """
    path_new_images = params['path_new_images']
    path_new_images = glob.glob(path_new_images)
    path_new_images = sorted(path_new_images)
    path_images = params['path_images']
    resize_dim = tuple(params['resize_dim']) if params['resize_dim'] else None
    """
    match_files = params['path_superglue']
    resize_dim = tuple(params['resize_dim']) if params['resize_dim'] else None

    file_path = match_files[0]
    data = np.load(file_path)

    state = Reconstruccion3D()
    Xw, x1, x2, T_wc1, T_wc2, state = triangulation.triangulate_from_superglue_pair(data, K, params, state, idx_file=0)

    print("Shape de Xw:", Xw.shape) # 3xN
    print("Shape de x1:", x1.shape) # 3xN
    print("Shape de x2:", x2.shape) # 3xN

    x1Data = x1.copy()  # 3xN
    x2Data = x2.copy()  # 3xN

    cameras = {'C1': T_wc1, 'C2': T_wc2}
    # plot_utils.plot3DPoints(Xw, cameras, world_ref=False)
    
    T_opt, X_w_opt = bundle.run_bundle_adjustment([T_wc1, T_wc2], K, Xw, [x1, x2], verbose=True)

    mean_reproj_error = projection.compute_reprojection_error(K, T_opt, X_w_opt, [x1Data, x2Data])
    print(f"Reprojection Error: {mean_reproj_error:.4f} Pixels")

    T_wc1_opt = T_opt[0]
    T_wc2_opt = T_opt[1]

    cameras = {'C1': T_wc1, 'C2': T_wc2}
    cameras_opt = {'C1': T_wc1_opt, 'C2': T_wc2_opt}
    
    if draw_graphic:
        draw_3d_points(Xw, X_w_opt, cameras, cameras_opt, T_wc1, T_wc2, T_wc1_opt, T_wc2_opt, K, x1Data, x2Data, params)
    
    path_new_images = params['path_new_images']
    path_new_images = glob.glob(path_new_images)
    path_new_images = sorted(path_new_images)


    for idx, path in enumerate(path_new_images[2:], start=3):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"[INFO] Processing new image: {name}")
        pts3, pts2, state, T_wc, state = triangulation.calculate_new_pose(params, name, state, K, name=name, verbose=False)     
        print(f"[INFO] Points 3D: {pts3.shape}, Points 2D: {pts2.shape}")
        
        state = triangulation.triagulate_new_image(params, name, state, K, verbose=False)
        print(f"point cloud shape: {state.point_cloud.shape}")

        cameras_opt[f'C{idx}'] = T_wc
        # plot_utils.plot3DPoints(state.point_cloud.T, cameras_opt, world_ref=False)
        

    state = triangulation.dlt_old_camera(state, params, "old-image")

    cameras_opt[f'C_old'] = state.image_data["old-image"]["t_wc"]
    plot_utils.plot3DPoints(state.point_cloud.T, cameras_opt, world_ref=False)
    



if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)
    parser = argparse.ArgumentParser(description="Ejemplo de script con argumento 'test' desde la línea de comandos.")
    
    # Definir el argumento para 'test', con valor por defecto 0
    parser.add_argument(
        '--test', 
        type=int, 
        default=0, 
        help="Valor de la variable 'test'. Valor por defecto es 0."
    )


    args = parser.parse_args()
    test = args.test

    with open('data/config.json', 'r') as f:
        config = json.load(f)

    if test == 0:
        params = config['test_0']
        K = np.loadtxt("data/K_c.txt")
    elif test == 1:
        params = config['test_1']
        K, dist, rvecs, tvecs = cam_calib.load_calibration_data("data/camera_calibration_2.npz")  
    else:
        params = config['default']
        # K = np.loadtxt("data/K_c.txt")
        K, dist, rvecs, tvecs = cam_calib.load_calibration_data("data/camera_calibration_2.npz")
    
    dist = None
    
    process_matches(params, K, draw_graphic=False)

