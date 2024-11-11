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

    x1 = np.loadtxt(work_dir+"x1.txt")
    x2 = np.loadtxt(work_dir+"x2.txt")
    x3 = np.loadtxt(work_dir+"x3.txt")
    x4 = np.loadtxt(work_dir+"x4.txt")

    return D1_k_array, D2_k_array, K_1, K_2, T_leftRight, T_wAwB_gt, T_wAwB_seed, x1, x2, x3, x4


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


def testing_2_1(K_1, D1_k_array):
    X_1 = np.array([[3, -5, 1], [2, 6, 5], [10, 7, 14], [1, 1, 1]])

    expected_points = np.array([
        [503.387, 450.1594],
        [267.9465, 580.4671],
        [441.0609, 493.0671]
    ]).T

    projected_points = utils.kannala_brandt_projection(X_1, K_1, D1_k_array)
    tolerance = 1e-3  

    if np.allclose(projected_points, expected_points, atol=tolerance):
        print(f"{GREEN}Test Correcto{RESET}")
        print(f"{GREEN}{projected_points}{RESET}")
    else:
        print(f"{RED}Test Fallido{RESET}")
        print("Diferencia entre los puntos esperados y proyectados:\n", np.abs(projected_points - expected_points))
        return 
    
    print()
    unprojected_directions = utils.kannala_brandt_unprojection(projected_points, K_1, D1_k_array)
    original_directions = X_1[:3, :] / np.linalg.norm(X_1[:3, :], axis=0)
    unprojected_directions = unprojected_directions / np.linalg.norm(unprojected_directions, axis=0)
    if np.allclose(original_directions, unprojected_directions, atol=tolerance):
        print(f"{GREEN}Test de Desproyección: Correcto{RESET}")
        print(f"{GREEN}{unprojected_directions}{RESET}")
    else:
        print(f"{RED}Test de Desproyección: Fallido{RESET}")
        print("Direcciones originales:\n", original_directions)
        print("Direcciones desproyectadas:\n", unprojected_directions)
        print("Diferencia:\n", np.abs(original_directions - unprojected_directions))




if __name__ == "__main__":
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    D1_k_array, D2_k_array, K_1, K_2, T_leftRight, T_wAwB_gt, T_wAwB_seed, x1, x2, x3, x4 = load_data()
    testing_2_1(K_1, D1_k_array)

    directions1 = utils.kannala_brandt_unprojection(x1, K_1, D1_k_array)  
    directions2 = utils.kannala_brandt_unprojection(x2, K_1, D1_k_array) 

    T_wc1 = np.eye(4)  # La cámara izquierda es el sistema de referencia
    T_wc2 = T_leftRight

    fisheye1_frameA, fisheye1_frameB, fisheye2_frameA, fisheye2_frameB = load_images()

    points_3d = utils.triangulate_points(directions1, directions2, T_wc1, T_wc2)

    plot_utils.project_points_plot(fisheye1_frameA, x1, points_3d, "Puntos 2D en la imagen izquierda")

