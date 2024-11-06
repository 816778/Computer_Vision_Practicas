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

work_dir = "/home/hsunekichi/Desktop/Computer_Vision_Practicas/p4/"
work_dir = ""

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

    _, _, _, K_c, _, x1Data, x2Data, x3Data = load_data()

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

    srcPts12, dstPts12, R12, t12 =  utils.create_match_lists(kpCv1, kpCv2, x1Data, x2Data, K_c)
    srcPts13, dstPts13, R13, t13 = utils.create_match_lists(kpCv1, kpCv3, x1Data, x3Data, K_c)

    T_wc1 = np.eye(4)
    T_wc2 = utils.ensamble_T(R12, t12)
    T_wc3 = utils.ensamble_T(R13, t13)
    P1 = K_c @ T_wc1[:3, :]
    P2 = K_c @ T_wc2[:3, :]

    X_w = utils.triangulate_points(P1, P2, srcPts12, dstPts12)

    x1 = x1Data
    x2 = x2Data
    x3 = x3Data

    T_wc1_opt, T_wc2_opt, T_wc3_opt, X_w_opt = utils.run_bundle_adjustment(T_wc1, T_wc2, T_wc3, K_c, X_w, x1, x2, x3)

    # Step 6: Visualize Optimized Projection
    x1_p_opt = utils.project_points(K_c, T_wc1_opt, X_w_opt)
    x2_p_opt = utils.project_points(K_c, T_wc2_opt, X_w_opt)
    x3_p_opt = utils.project_points(K_c, T_wc3_opt, X_w_opt)

    plot_utils.visualize_projection(image1, x1Data, x1_p_opt, "Image 1 - Optimized Projection")
    plot_utils.visualize_projection(image2, x2Data, x2_p_opt, "Image 2 - Optimized Projection")
    plot_utils.visualize_projection(image3, x3Data, x3_p_opt, "Image 3 - Optimized Projection")