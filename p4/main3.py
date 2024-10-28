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

    theta_init = np.array([0.0, 0.0, 0.0])  # Inicialización de rotación en so(3)
    t_21_init = np.array([0.0, 0.0, 0.0])
    X1_init = X_w[:3, :]

    initial_params = np.hstack((theta_init, t_21_init, X1_init.flatten()))

    result = scOptim.least_squares(utils.resBundleProjection, initial_params,
                       args=(x1Data, x2Data, K_c, x1Data.shape[1]), method='lm')

    optimized_params = result.x
    theta_opt = optimized_params[:3]
    t_21_opt = optimized_params[3:6]
    X1_opt = optimized_params[6:].reshape(3, x1Data.shape[1])

    R_21_opt = utils.rotvec_to_rotmat(theta_opt)
    T_21_opt = np.eye(4)
    T_21_opt[:3, :3] = R_21_opt
    T_21_opt[:3, 3] = t_21_opt

    # Visualizar los puntos optimizados en las imágenes
    x1_proj_opt = utils.project_points(K_c, np.eye(4), X1_opt)  # Cámara 1 como identidad
    x2_proj_opt = utils.project_points(K_c, T_21_opt, X1_opt)

    image1 = cv2.imread('images/image1.png')
    image2 = cv2.imread('images/image2.png')

    # Graficar los puntos proyectados junto con los puntos observados
    plot_utils.visualize_projection(image1, x1Data, x1_proj_opt, 'Image 1 - Optimized Projection')
    plot_utils.visualize_projection(image2, x2Data, x2_proj_opt, 'Image 2 - Optimized Projection')
