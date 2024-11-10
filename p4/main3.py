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
    # pts1_3, pts3, R13, t13 = utils.linearPoseEstimation(x1Data, x3Data, kpCv1, kpCv3, K_c)

    T_wc1 = np.eye(4)   # se toma la primera cámara como referencia
    T_wc2 = utils.ensamble_T(R12, t12)
    # T_wc3 = utils.ensamble_T(R13, t13)

    P1 = utils.projectionMatrix(K_c, T_wc1)
    P2 = utils.projectionMatrix(K_c, T_wc2)

    X_w = utils.triangulate_points(P1, P2, x1Data, x2Data)

    x1 = x1Data
    x2 = x2Data
    x3 = x3Data

    T_opt, X_w_opt = utils.run_bundle_adjustment([T_wc1, T_wc2], K_c, X_w, [x1, x2])
    T_wc1_opt, T_wc2_opt = T_opt

    if X_w_opt.shape[1] < 4:
        raise ValueError("Se requieren al menos 4 puntos para solvePnP con SOLVEPNP_EPNP")

    objectPoints = X_w_opt.T.astype(np.float64)
    imagePoints = np.ascontiguousarray(x3[0:2,:].T).reshape((x3.shape[1], 1, 2))
    print("objectPoints", objectPoints.shape)
    print("imagePoints", imagePoints.shape)

    distCoeffs = np.zeros((4, 1), dtype=np.float64)
    retval, rvec, tvec = cv2.solvePnP(
        objectPoints, 
        imagePoints, 
        K_c, 
        distCoeffs, 
        flags=cv2.SOLVEPNP_EPNP
    )
    
    #R = utils.rotvec_to_rotmat(rvec)
    R, _ = cv2.Rodrigues(rvec)
    print("Matriz de rotación R:", R)
    print("Vector de traslación t:", tvec)

    T_wc3 = np.eye(4)
    T_wc3[:3, :3] = R 
    T_wc3[:3, 3] = tvec.flatten()

    T_cw3 = T_wc3.copy()
    T_wc3 = np.linalg.inv(T_cw3)

    fig3D = plt.figure(1)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plot_utils.drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    plot_utils.drawRefSystem(ax, T_wc1_opt, '-', 'C1')
    plot_utils.drawRefSystem(ax, T_wc2_opt, '-', 'C2')
    plot_utils.drawRefSystem(ax, T_wc3, '-', 'C3')

    ax.scatter(X_w_opt[0, :], X_w_opt[1, :], X_w_opt[2, :], marker='.')

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()