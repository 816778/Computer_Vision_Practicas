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

    # Convertir a coordenadas homogéneas
    x1Data = np.vstack((x1Data, np.ones((1, x1Data.shape[1]))))
    x2Data = np.vstack((x2Data, np.ones((1, x2Data.shape[1]))))
    x3Data = np.vstack((x3Data, np.ones((1, x3Data.shape[1]))))

    return T_wc1, T_wc2, T_wc3, K_c, X_w, x1Data, x2Data, x3Data



if __name__ == "__main__":
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    T_wc1_ref, T_wc2_ref, T_wc3_ref, K_c, _, x1Data, x2Data, x3Data = load_data()

    #F2 = np.loadtxt(work_dir+"data/F_21.txt")
    #F3 = utils.calculate_fundamental_matrix(T_wc1_ref, T_wc3_ref, K_c)

    t3_ref = T_wc3_ref[0:3, 3]
    R3_ref = T_wc3_ref[0:3, 0:3] 

    t2_ref = T_wc2_ref[0:3, 3]
    R2_ref = T_wc2_ref[0:3, 0:3]

    #plot_utils.plot_3D_scene(T_wc1, T_wc2, T_wc3, X_w)
    
    im1_pth = work_dir+"images/image1.png"
    im2_pth = work_dir+"images/image2.png"
    im3_pth = work_dir+"images/image3.png"

    image1 = cv2.imread(im1_pth)
    image2 = cv2.imread(im2_pth)
    image3 = cv2.imread(im3_pth)
    
    R12, t12 = utils.linearPoseEstimation(x1Data, x2Data, K_c)
    R13, t13 = utils.linearPoseEstimation(x1Data, x3Data, K_c)

    T_wc1 = np.eye(4)   # se toma la primera cámara como referencia
    T_wc2 = utils.ensamble_T(R12, t12)
    T_wc3 = utils.ensamble_T(R13, t13)


    P1 = utils.projectionMatrix(K_c, T_wc1) # K_c @ T_wc1[0:3, :]
    P2 = utils.projectionMatrix(K_c, T_wc2) # K_c @ T_wc2[0:3, :]
    P3 = utils.projectionMatrix(K_c, T_wc3) # K_c @ T_wc3[0:3, :]
    
    X_w = utils.triangulate_points(P1, P2, x1Data, x2Data)

    x1_no_opt = utils.project_points(K_c, T_wc1, X_w)
    x2_no_opt = utils.project_points(K_c, T_wc2, X_w)
    x3_no_opt = utils.project_points(K_c, T_wc3, X_w)

    x1 = x1Data
    x2 = x2Data
    x3 = x3Data
    print(f"x1Data: {x1Data.shape}")
    print(f"X_w shape: {X_w.shape}")
    print(f"T_wc1: {T_wc1}")
    print(f"T_wc2: {T_wc2}")
    print(f"T_wc3: {T_wc3}")
    exit()
    T_opt, X_w_opt = utils.run_bundle_adjustment([T_wc1, T_wc2, T_wc3], K_c, X_w, [x1, x2, x3])

    T_wc1_opt, T_wc2_opt, T_wc3_opt = T_opt

    # Step 6: Visualize Optimized Projection
    x1_p_opt = utils.project_points(K_c, T_wc1_opt, X_w_opt)
    x2_p_opt = utils.project_points(K_c, T_wc2_opt, X_w_opt)
    x3_p_opt = utils.project_points(K_c, T_wc3_opt, X_w_opt)
    print(f"x1Data: {x1Data.shape}")
    print(f"X1[0]: {X_w_opt.T[0]}")


    plot_utils.visualize_projection_2(image1, x1Data, x1_no_opt, x1_p_opt, 'Image 1')
    plot_utils.visualize_projection_2(image2, x2Data, x2_no_opt, x2_p_opt, 'Image 2')
    plot_utils.visualize_projection_2(image3, x3Data, x3_no_opt, x3_p_opt, 'Image 3')

    fig3D = plt.figure(1)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plot_utils.drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    plot_utils.drawRefSystem(ax, T_wc1_opt, '-', 'C1')
    plot_utils.drawRefSystem(ax, T_wc2_opt, '-', 'C2')
    plot_utils.drawRefSystem(ax, T_wc3_opt, '-', 'C3')

    ax.scatter(X_w_opt[0, :], X_w_opt[1, :], X_w_opt[2, :], marker='.')

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()