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

work_dir = "/home/hsunekichi/Escritorio/Computer_Vision_Practicas/p4/"
#work_dir = ""

def load_data():
    T_wc1 = np.loadtxt(work_dir+"data/T_w_c1.txt")
    T_wc2 = np.loadtxt(work_dir+"data/T_w_c2.txt")
    T_wc3 = np.loadtxt(work_dir+"data/T_w_c3.txt")
    K_c = np.loadtxt(work_dir+"data/K_c.txt")
    X_w = np.loadtxt(work_dir+"data/X_w.txt")
    x1Data = np.loadtxt(work_dir+"data/x1Data.txt")
    x2Data = np.loadtxt(work_dir+"data/x2Data.txt")
    x3Data = np.loadtxt(work_dir+"data/x3Data.txt")
    return T_wc1, T_wc2, T_wc3, K_c, X_w, x1Data, x2Data, x3Data



if __name__ == "__main__":
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    _, _, T_wc3_ref, K_c, _, x1Data, x2Data, x3Data = load_data()

    #plot_utils.plot_3D_scene(T_wc1, T_wc2, T_wc3, X_w)
    
    im1_pth = work_dir+"images/image1.png"
    im2_pth = work_dir+"images/image2.png"
    im3_pth = work_dir+"images/image3.png"

    image1 = cv2.imread(im1_pth)
    image2 = cv2.imread(im2_pth)
    image3 = cv2.imread(im3_pth)

    # Construct the matches
    kpCv1 = []
    kpCv2 = []
    kpCv3 = []
    for kPoint in range(x1Data.shape[1]):
        kpCv1.append(cv2.KeyPoint(x1Data[0, kPoint], x1Data[1, kPoint],1))
        kpCv2.append(cv2.KeyPoint(x2Data[0, kPoint], x2Data[1, kPoint],1))
        kpCv3.append(cv2.KeyPoint(x3Data[0, kPoint], x3Data[1, kPoint],1))

    
    pts1, pts2, R12, t12 = utils.linearPoseEstimation(x1Data, x2Data, kpCv1, kpCv2, K_c)

    #x_coords = np.array(x_coords)
    #y_coords = np.array(y_coords)
    #x1 = np.vstack((x_coords, y_coords))
    #plot_utils.plot_epipolar_lines(F12, srcPts12, image2)


    T_wc1 = np.eye(4)   # se toma la primera cámara como referencia
    T_wc2 = utils.ensamble_T(R12, t12)

    P1 = utils.projectionMatrix(K_c, T_wc1) # K_c @ T_wc1[0:3, :]
    P2 = utils.projectionMatrix(K_c, T_wc2) # K_c @ T_wc2[0:3, :]

    X_w = utils.triangulate_points(P1, P2, pts1, pts2)

    x1 = x1Data
    x2 = x2Data
    x3 = x3Data

    T_opt, X_w_opt = utils.run_bundle_adjustmentFull([T_wc1, T_wc2], K_c, X_w, [x1, x2])

    T_wc1_opt = T_opt[0]
    T_wc2_opt = T_opt[1]

    # Step 6: Visualize Optimized Projection
    x1_p_opt = utils.project_points(K_c, T_wc1_opt, X_w)
    x2_p_opt = utils.project_points(K_c, T_wc2_opt, X_w)

    plot_utils.visualize_projection(image1, x1Data, x1_p_opt, "Image 1 - Optimized Projection")
    plot_utils.visualize_projection(image2, x2Data, x2_p_opt, "Image 2 - Optimized Projection")
