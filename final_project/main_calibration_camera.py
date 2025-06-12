####################################################################################################
#
# Title: main.py
# Project: 
# Authors: Eryka Rimacuna
# Description: This file contains the necessary imports for the main.py file.
#
####################################################################################################

# Import the necessary libraries
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Import the necessary functions from the files
import utils.camera_calibration as cam_calib
import utils.utils as utils
import utils.math as ut_math
import utils.plot_utils as plot_utils
import utils.bundle_adjustment as bun_adj
import utils.functions_cv as fcv

# Constants
IMAGE_PATH = 'images/'
DATA_PATH = 'data/'


def calibration_my_camera(pattern_size, image_downsize_factor, path_images_calibration='mobilePhoneCameraCalibration/calib_*.jpg', fov=1):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibration_images = glob.glob(path_images_calibration)

    # Calibration process
    objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs = cam_calib.calibrate_camera(calibration_images, pattern_size, image_downsize_factor, criteria)

    if len(objpoints) == 0 or len(imgpoints) == 0:
        print("No se encontraron suficientes imágenes para la calibración.")
        exit()

    # Guarda los resultados relevantes
    cam_calib.save_calibration_data(mtx, dist, rvecs, tvecs, folder="data", fov=fov)

    # Error de reproyección
    reprojection_error = cam_calib.calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    print(f"Reprojection Error: {reprojection_error}")

    print("\nCalibration Results:")
    print("Intrinsic Matrix (K):\n", mtx)
    print("Distortion Coefficients:\n", dist)


def calibration_old_camera():
    # Load data
    T_list, X_w, points_2d = fcv.load_data_3d_points()
    points_2d_old = fcv.load_data_old_points()
    K_old_estimated = fcv.aprox_K()

    T_wc1, T_wc2, T_wc3, T_wc4 = T_list
    cameras = {
            'C1': T_wc1,  
            'C2': T_wc2,
            'C3': T_wc3,
            'C4': T_wc4,
        }
    print(f"X_w_opt: {X_w.shape}")
    plot_utils.plot3DPoints(X_w, cameras, world_ref=False)
    
    
    x1, x2, x3, x4 = points_2d[0].T, points_2d[1].T, points_2d[2].T, points_2d[3].T
    x1_old, x_old = points_2d_old[0].T, points_2d_old[4].T

    indices = []

    for i in range(x1.shape[1]):
        for j in range(x1_old.shape[1]):
            if np.array_equal(x1[:, i], x1_old[:, j]):
                indices.append(i)

    X_w_match = X_w[:, indices]

    X_w_match = np.array(X_w_match, dtype=np.float32)  # Asegúrate de que sea N x 3
    x_old = np.array(x_old, dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))
    
    print("Puntos 3D (formato final):", X_w_match.shape, X_w_match.dtype)
    print("Puntos 2D (formato final):", x_old.shape, x_old.dtype)
    success, rvec, tvec = cv2.solvePnP(
        X_w_match, x_old, K_old_estimated, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
       print("solvePnP no pudo estimar la pose.") 



    




if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)
    
    # parameters of the camera calibration pattern
    pattern_num_rows = 9
    pattern_num_cols = 6
    pattern_size = (pattern_num_rows, pattern_num_cols)
    image_downsize_factor = 4
    fov = "1-5"

    calibration_my_camera(pattern_size, image_downsize_factor, path_images_calibration=f'mobilePhoneCameraCalibration/my_phone_{fov}/calib_*.jpeg', fov=fov)
