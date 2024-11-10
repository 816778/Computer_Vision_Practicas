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
from scipy.linalg import expm, logm
import time

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

    _, _, _, K_c, _, x1Data, x2Data, x3Data = load_data()

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

    P1 = utils.projectionMatrix(K_c, T_wc1) 
    P2 = utils.projectionMatrix(K_c, T_wc2)
    P3 = utils.projectionMatrix(K_c, T_wc3)

    X_w = utils.triangulate_points(P1, P2, x1Data, x2Data)
    # = utils.triangulate_points(P1, P3, srcPts13, dstPts13)

    x1_no_opt = utils.project_points(K_c, T_wc1, X_w)
    x2_no_opt = utils.project_points(K_c, T_wc2, X_w)
    x3_no_opt = utils.project_points(K_c, T_wc3, X_w)
    
    ejecucion_R = False

    #######################
    if ejecucion_R:
        T_wc_list = [T_wc1, T_wc2, T_wc3]
        theta_t_list = utils.convert_T_wc_to_theta_t_list(T_wc_list)
        xData_list = [x1Data, x2Data, x3Data]
        theta_t_opt_list, X_w_opt = utils.resBundleProjection_2(theta_t_list, K_c, X_w, xData_list)

        projected_points_list = utils.project_points_multi_view(theta_t_opt_list, K_c, X_w_opt)
        x1_p_opt, x2_p_opt, x3_p_opt = projected_points_list

        plot_utils.visualize_projection_2(image1, x1Data, x1_no_opt, x1_p_opt, 'Image 1')
        plot_utils.visualize_projection_2(image2, x2Data, x2_no_opt, x2_p_opt, 'Image 2')
        plot_utils.visualize_projection_2(image3, x3Data, x3_no_opt, x3_p_opt, 'Image 3')

    else:
        using_2_views = False
        if using_2_views:
            T_wc1_opt, T_wc2_opt, X_w_opt = utils.run_bundle_adjustment_two_views(T_wc1, T_wc2, K_c, X_w, x1Data, x2Data)
            T_wc1_opt, T_wc3_opt, X_w_opt = utils.run_bundle_adjustment_two_views(T_wc1_opt, T_wc3, K_c, X_w_opt, x1Data, x3Data)
            print("TWO-VIEW")
        else:
            start_time = time.time()
            T_wc_opt_list, X_w_opt = utils.run_bundle_adjustmentFullT([T_wc1, T_wc2, T_wc3], K_c, X_w, [x1Data, x2Data, x3Data])
            end_time = time.time()  
            T_wc1_opt, T_wc2_opt, T_wc3_opt = T_wc_opt_list
            elapsed_time_2 = end_time - start_time
            print(f"Tiempo empleado en run_bundle_adjustmentFull: {elapsed_time_2:.2f} segundos")
                

        
        x1_p_opt = utils.project_points(K_c, T_wc1_opt, X_w_opt)
        x2_p_opt = utils.project_points(K_c, T_wc2_opt, X_w_opt)
        x3_p_opt = utils.project_points(K_c, T_wc3_opt, X_w_opt)
        
        plot_utils.visualize_projection_2(image1, x1Data, x1_no_opt, x1_p_opt, 'Image 1')
        plot_utils.visualize_projection_2(image2, x2Data, x2_no_opt, x2_p_opt, 'Image 2')
        plot_utils.visualize_projection_2(image3, x3Data, x3_no_opt, x3_p_opt, 'Image 3')


        #plot_utils.visualize_projection(image1, x1Data, x1_p_opt, "Image 1 - Optimized Projection")
        #plot_utils.visualize_projection(image2, x2Data, x2_p_opt, "Image 2 - Optimized Projection")
        #plot_utils.visualize_projection(image3, x3Data, x3_p_opt, "Image 3 - Optimized Projection")
    """
    T_wc2_wrong: 
    [[ 0.9902  0.1366 -0.0285 -0.0228]
    [-0.1393  0.9564 -0.2567  0.0079]
    [-0.0078  0.2582  0.9661  0.0258]
    [ 0.      0.      0.      1.    ]]

    T_wc2_opt: 
    [[ 0.2843 -1.1693  0.1616 -0.7112]
    [ 0.1871  1.108  -0.0215  0.1254]
    [-0.4835 -1.0109  0.4435  0.672 ]]

    Theta final: [-1.8359  0.3148  0.2362].
    t_final: [-0.7112  0.1254  0.672 ]
    """
    #########################################################################################

   