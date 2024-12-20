from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.linalg as scAlg
import csv
import scipy as sc
import scipy.optimize as scOptim
import scipy.io as sio

import utils.utils as utils
import utils.plot_utils as plot_utils


RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


def load_data():
    work_dir = "/home/hsunekichi/Desktop/Computer_Vision_Practicas/p5/data/"
    work_dir = "data/"

    D1_k_array = np.loadtxt(work_dir+"D1_k_array.txt")
    D2_k_array = np.loadtxt(work_dir+"D2_k_array.txt")

    K_1 = np.loadtxt(work_dir+"K_1.txt")
    K_2 = np.loadtxt(work_dir+"K_2.txt")

    T_leftRight = np.loadtxt(work_dir+"T_leftRight.txt")
    T_wAwB_gt = np.loadtxt(work_dir+"T_wAwB_gt.txt")
    T_wAwB_seed = np.loadtxt(work_dir+"T_wAwB_seed.txt")

    T_wc1 = np.loadtxt(work_dir+"T_wc1.txt")
    T_wc2 = np.loadtxt(work_dir+"T_wc2.txt")

    x1 = np.loadtxt(work_dir+"x1.txt")
    x2 = np.loadtxt(work_dir+"x2.txt")
    x3 = np.loadtxt(work_dir+"x3.txt")
    x4 = np.loadtxt(work_dir+"x4.txt")

    return D1_k_array, D2_k_array, K_1, K_2, T_leftRight, T_wAwB_gt, T_wAwB_seed, x1, x2, x3, x4, T_wc1, T_wc2


def load_images():
    work_dir = "/home/hsunekichi/Desktop/Computer_Vision_Practicas/p5/images/"
    work_dir = "images/"

    fisheye1_frameA = work_dir+"fisheye1_frameA.png"
    fisheye1_frameB = work_dir+"fisheye1_frameB.png"
    fisheye2_frameA = work_dir+"fisheye2_frameA.png"
    fisheye2_frameB = work_dir+"fisheye2_frameB.png"

    fisheye1_frameA = cv2.imread(fisheye1_frameA)
    fisheye1_frameB = cv2.imread(fisheye1_frameB)
    fisheye2_frameA = cv2.imread(fisheye2_frameA)   
    fisheye2_frameB = cv2.imread(fisheye2_frameB)

    return fisheye1_frameA, fisheye1_frameB, fisheye2_frameA, fisheye2_frameB



import numpy as np

def test_transformation_equality(T_wAwB_opt, T_wAwB_gt, tol_rotation=1e-6, tol_translation=1e-6):
    """
    Test if T_wAwB_opt is equal to T_wAwB_gt within tolerances.
    
    Args:
        T_wAwB_opt: Optimized transformation matrix (4x4).
        T_wAwB_gt: Ground truth transformation matrix (4x4).
        tol_rotation: Tolerance for rotational difference (radians).
        tol_translation: Tolerance for translational difference (meters).
    
    Returns:
        result: Boolean indicating if the two transformations are approximately equal.
        details: Dictionary with rotation and translation errors.
    """
    # Extract rotation and translation components
    R_opt = T_wAwB_opt[:3, :3]
    t_opt = T_wAwB_opt[:3, 3]
    
    R_gt = T_wAwB_gt[:3, :3]
    t_gt = T_wAwB_gt[:3, 3]
    
    # Calculate rotation difference using the angle of the rotation matrix
    R_diff = np.dot(R_gt.T, R_opt)
    trace_R_diff = np.trace(R_diff)
    rotation_error = np.arccos(np.clip((trace_R_diff - 1) / 2, -1.0, 1.0))  # Angle in radians
    
    # Calculate translation difference (Euclidean distance)
    translation_error = np.linalg.norm(t_opt - t_gt)
    
    # Check if errors are within tolerances
    rotation_match = rotation_error <= tol_rotation
    translation_match = translation_error <= tol_translation
    
    result = rotation_match and translation_match
    
    # Return detailed results
    details = {
        "rotation_error (rad)": rotation_error,
        "translation_error": translation_error,
        "rotation_match": rotation_match,
        "translation_match": translation_match,
        "result": result
    }
    

    # Mostrar resultados
    if result:
        print(f"{GREEN}Las matrices de transformación son equivalentes dentro de las tolerancias.{RESET}")
    else:
        print(f"{RED}Las matrices de transformación NO son equivalentes.{RESET}")
    print(f"{RED}########################################################################")
    print(details)
    print(f"########################################################################{RESET}")
    




if __name__ == "__main__":

    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    D1_k_array, D2_k_array, K_1, K_2, T_leftRight, T_wAwB_gt, T_wAwB_seed, x1, x2, x3, x4, T_wc1, T_wc2 = load_data()
    fisheye1_frameA, fisheye1_frameB, fisheye2_frameA, fisheye2_frameB = load_images()

    calibration1 = (K_1, D1_k_array, None)
    calibration2 = (K_2, D2_k_array, None)


    X_w = utils.triangulate_points(x1, x2, T_wc1, T_wc2, calibration1, calibration2)
    
    x_proj1A = utils.project_points_fisheye(T_wc1, calibration1, X_w)
    x_proj2A = utils.project_points_fisheye(T_wc2, calibration2, X_w)

    xData = [x1, x2, x3, x4]

    T_wAwB, X_w = utils.run_bundle_adjustment_fisheye(T_wAwB_seed, K_1, K_2, D1_k_array, D2_k_array, X_w, xData, T_wc1, T_wc2)

    T_wc1B = T_wAwB @ T_wc1
    T_wc2B = T_wAwB @ T_wc2

    # test_transformation_equality(T_wAwB, T_wAwB_gt)

    cameras = {
        'C1': T_wc1,  
        'C2': T_wc2,
        'C1_B': T_wc1B,
        'C2_B': T_wc2B
    }

    plot_utils.plot3DPoints(X_w, cameras, world_ref=False, block=False)

    x_proj1A = utils.project_points_fisheye(T_wc1, (K_1, D1_k_array, None), X_w)
    x_proj2A = utils.project_points_fisheye(T_wc2, (K_2, D2_k_array, None), X_w)
    x_proj1B = utils.project_points_fisheye(T_wc1B, (K_1, D1_k_array, None), X_w)
    x_proj2B = utils.project_points_fisheye(T_wc2B, (K_2, D2_k_array, None), X_w)


    plot_utils.visualize_projection(fisheye1_frameA, x1, x_proj1A, "C1A", False)
    plot_utils.visualize_projection(fisheye2_frameA, x2, x_proj2A, "C2A", False)

    plot_utils.visualize_projection(fisheye1_frameB, x3, x_proj1B, "C1B", False)
    plot_utils.visualize_projection(fisheye2_frameB, x4, x_proj2B, "C2B")

    print("C1A", plot_utils.computeRMSE(x1, x_proj1A))
    print("C2A", plot_utils.computeRMSE(x2, x_proj2A))
    print("C1B", plot_utils.computeRMSE(x3, x_proj1B))
    print("C2B", plot_utils.computeRMSE(x4, x_proj2B))
