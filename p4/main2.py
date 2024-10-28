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

    plot_utils.plot_3D_scene(T_wc1, T_wc2, T_wc3, X_w)

    image1 = cv2.imread('images/image1.png')
    image2 = cv2.imread('images/image2.png')
    image3 = cv2.imread('images/image3.png')
    
    kpCv1 = [cv2.KeyPoint(x1Data[0, k], x1Data[1, k], 1) for k in range(x1Data.shape[1])]
    kpCv2 = [cv2.KeyPoint(x2Data[0, k], x2Data[1, k], 1) for k in range(x2Data.shape[1])]
    kpCv3 = [cv2.KeyPoint(x3Data[0, k], x3Data[1, k], 1) for k in range(x3Data.shape[1])]

    matchesList12 = np.hstack((np.arange(x1Data.shape[1]).reshape(-1, 1),
                               np.arange(x2Data.shape[1]).reshape(-1, 1),
                               np.ones((x1Data.shape[1], 1))))
    dMatchesList12 = utils.indexMatrixToMatchesList(matchesList12)

    plot_utils.visualize_matches(image1, image2, kpCv1, kpCv2, dMatchesList12, "Matches between views 1 and 2")
    plot_utils.visualize_matches(image1, image3, kpCv1, kpCv3, dMatchesList12, "Matches between views 1 and 3")

    x1_p = utils.project_points(K_c, T_wc1, X_w)
    x2_p = utils.project_points(K_c, T_wc2, X_w)
    x3_p = utils.project_points(K_c, T_wc3, X_w)
   
   
    T_wc1_opt, T_wc2_opt, T_wc3_opt, X_w_opt = utils.run_bundle_adjustment(T_wc1, T_wc2, T_wc3, K_c, X_w, x1Data, x2Data, x3Data)
    
    # Step 6: Visualize Optimized Projection
    x1_p_opt = utils.project_points(K_c, T_wc1_opt, X_w_opt)
    x2_p_opt = utils.project_points(K_c, T_wc2_opt, X_w_opt)
    x3_p_opt = utils.project_points(K_c, T_wc3_opt, X_w_opt)

    plot_utils.visualize_projection(image1, x1Data, x1_p_opt, 'Image 1 - Optimized Projection')
    plot_utils.visualize_projection(image2, x2Data, x2_p_opt, 'Image 2 - Optimized Projection')
    plot_utils.visualize_projection(image3, x3Data, x3_p_opt, 'Image 3 - Optimized Projection')