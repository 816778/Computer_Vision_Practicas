#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 3
#
# Title: Bundle Adjustment and Multiview Geometry
#
# Date: 26 October 2020
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

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


def load_data():
    T_wc1 = np.loadtxt('data/T_w_c1.txt')
    T_wc2 = np.loadtxt('data/T_w_c2.txt')
    T_wc3 = np.loadtxt('data/T_w_c3.txt')
    K_c = np.loadtxt('data/K_c.txt')
    X_w = np.loadtxt('data/X_w.txt')
    x1Data = np.loadtxt('data/x1Data.txt')
    x2Data = np.loadtxt('data/x2Data.txt')
    x3Data = np.loadtxt('data/x3Data.txt')
    return T_wc1, T_wc2, T_wc3, K_c, X_w, x1Data, x2Data, x3Data



if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    T_wc1, T_wc2, T_wc3, K_c, X_w, x1Data, x2Data, x3Data = load_data()
    image1 = cv2.imread('images/image1.png')
    image2 = cv2.imread('images/image2.png')
    image3 = cv2.imread('images/image3.png')

    theta_21_init = np.zeros(3)
    t_21_init = np.zeros(3)
    theta_31_init = np.zeros(3)
    t_31_init = np.zeros(3)
    X1_init = X_w[:3, :]

    initial_params = np.hstack((theta_21_init, t_21_init, theta_31_init, t_31_init, X1_init.flatten()))


    result = scOptim.least_squares(utils.resBundleProjectionThreeViews, initial_params,
                       args=(x1Data, x2Data, x3Data, K_c, X1_init.shape[1]), method='lm')

    # Reconstruir los parámetros optimizados
    optimized_params = result.x
    theta_21_opt = optimized_params[:3]
    t_21_opt = optimized_params[3:6]
    theta_31_opt = optimized_params[6:9]
    t_31_opt = optimized_params[9:12]
    X1_opt = optimized_params[12:].reshape(3, -1)

    # Convertir theta_21_opt y theta_31_opt a matrices de rotación
    R_21_opt = scAlg.expm(utils.crossMatrix(theta_21_opt))
    R_31_opt = scAlg.expm(utils.crossMatrix(theta_31_opt))

    # Construir las matrices de transformación optimizadas T_21 y T_31
    T_21_opt = np.eye(4)
    T_21_opt[:3, :3] = R_21_opt
    T_21_opt[:3, 3] = t_21_opt

    T_31_opt = np.eye(4)
    T_31_opt[:3, :3] = R_31_opt
    T_31_opt[:3, 3] = t_31_opt

    x1_proj_opt = utils.project_points(K_c, np.eye(4), X1_opt)      # Primera cámara en el origen
    x2_proj_opt = utils.project_points(K_c, T_21_opt, X1_opt)       # Proyección en la segunda cámara
    x3_proj_opt = utils.project_points(K_c, T_31_opt, X1_opt)

    # Visualización de proyecciones y residuales para cada imagen
    plot_utils.visualize_projection(image1, x1Data, x1_proj_opt, 'Image 1 - Optimized Projection')
    plot_utils.visualize_projection(image2, x2Data, x2_proj_opt, 'Image 2 - Optimized Projection')
    plot_utils.visualize_projection(image3, x3Data, x3_proj_opt, 'Image 3 - Optimized Projection')
