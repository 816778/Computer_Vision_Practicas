import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as scAlg
import scipy as sc
import scipy.optimize as scOptim
import scipy.io as sio
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm

import utils.utils as utils
import utils.plot_utils as plot_utils

##########################################################################
# 
# FUNCIONES BASE
#
##########################################################################
def indexMatrixToMatchesList(matchesList):
    """
    Convert a numpy matrix of index in a list of DMatch OpenCv matches.
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0].astype('int'), _trainIdx=row[1].astype('int'), _distance=row[2]))
    return dMatchesList


def matchesListToIndexMatrix(dMatchesList):
    """
    Convert a list of DMatch OpenCv matches into a numpy matrix of index.

     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([np.int(dMatchesList[k].queryIdx), np.int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList


def project_points(K, T, X_w):
    if T.shape == (3, 4):
        T_hom = np.vstack([T, [0, 0, 0, 1]])
    else:
        T_hom = T

    if X_w.shape[0] == 3:
        X_w_hom = np.vstack([X_w, np.ones((1, X_w.shape[1]))])
    else:
        X_w_hom = X_w

    x_proj_hom = K @ np.eye(3, 4) @ np.linalg.inv(T_hom) @ X_w_hom
    x_proj = x_proj_hom / x_proj_hom[2, :]
    return x_proj



def residual_bundle_adjustment(params, K, x1Data, x2Data, x3Data):
    T_wc1 = params[:12].reshape(3, 4)
    T_wc2 = params[12:24].reshape(3, 4)
    T_wc3 = params[24:36].reshape(3, 4)
    X_w = params[36:].reshape(-1, 3).T
    
    x1_proj = project_points(K, T_wc1, X_w)
    x2_proj = project_points(K, T_wc2, X_w)
    x3_proj = project_points(K, T_wc3, X_w)
    
    residuals = np.hstack((
        (x1_proj[:2, :] - x1Data).flatten(),
        (x2_proj[:2, :] - x2Data).flatten(),
        (x3_proj[:2, :] - x3Data).flatten()
    ))

    return residuals

def run_bundle_adjustment(T_wc1, T_wc2, T_wc3, K, X_w, x1Data, x2Data, x3Data):
    if X_w.shape[0] == 4:
        X_w = (X_w[:3, :] / X_w[3, :])
    
    T_wc1_flat = T_wc1[:3, :].flatten()
    T_wc2_flat = T_wc2[:3, :].flatten()
    T_wc3_flat = T_wc3[:3, :].flatten()
    X_w_flat = X_w.T.flatten()

    initial_params = np.hstack((T_wc1_flat, T_wc2_flat, T_wc3_flat, X_w_flat))
    
    result = scOptim.least_squares(residual_bundle_adjustment, initial_params,
                                   args=(K, x1Data, x2Data, x3Data), method='lm')
    
    optimized_params = result.x
    T_wc1_opt = optimized_params[:12].reshape(3, 4)
    T_wc2_opt = optimized_params[12:24].reshape(3, 4)
    T_wc3_opt = optimized_params[24:36].reshape(3, 4)
    X_w_opt = optimized_params[36:].reshape(-1, 3).T
    
    return T_wc1_opt, T_wc2_opt, T_wc3_opt, X_w_opt


def skew_symmetric(v):
    """Genera la matriz skew-symmetric de un vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def rotvec_to_rotmat(theta):
    """Convierte un vector de rotación theta en una matriz de rotación."""
    angle = np.linalg.norm(theta)
    if angle == 0:
        return np.eye(3)  # Sin rotación
    axis = theta / angle
    K = skew_symmetric(axis)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R

def crossMatrix(x):
    """Genera la matriz skew-symmetric de un vector."""
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])



def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
    """
    Compute the residuals for bundle adjustment with two views.
    """
    # Extraer los parámetros de rotación y traslación para T_21
    theta = Op[:3]  # Los primeros tres elementos para la rotación en so(3)
    t_21 = Op[3:6]  # Siguientes tres elementos para la traslación

    # Convertir theta en una matriz de rotación R_21 usando scipy
    R_21 = rotvec_to_rotmat(theta)

    # Extraer puntos 3D en la primera referencia
    X1 = Op[6:].reshape(3, nPoints)

    # Construir la matriz de transformación T_21 de la segunda cámara con respecto a la primera
    T_21 = np.eye(4)
    T_21[:3, :3] = R_21
    T_21[:3, 3] = t_21

    # Proyección en la primera cámara
    x1_proj_hom = K_c @ X1
    x1_proj = x1_proj_hom[:2, :] / x1_proj_hom[2, :]  # Normalizar por coordenada z

    # Proyección en la segunda cámara (transformar X1 a la referencia 2)
    X2 = T_21[:3, :3] @ X1 + T_21[:3, 3:4]
    x2_proj_hom = K_c @ X2
    x2_proj = x2_proj_hom[:2, :] / x2_proj_hom[2, :]  # Normalizar por coordenada z

    # Calcular los residuales como diferencias entre proyectados y observados
    residuals_x1 = (x1_proj - x1Data[:2, :]).flatten()
    residuals_x2 = (x2_proj - x2Data[:2, :]).flatten()

    # Concatenar los residuales para ambas proyecciones
    residuals = np.hstack((residuals_x1, residuals_x2))
    
    return residuals


def resBundleProjectionThreeViews(Op, x1Data, x2Data, x3Data, K_c, nPoints):
    """
    -input:
    Op: Vector de optimización que incluye una parametrización para T_21 y T_31, 
        y los puntos 3D en el sistema de la cámara 1.
    x1Data: Puntos observados en la imagen 1 (coordenadas homogéneas 3xnPoints).
    x2Data: Puntos observados en la imagen 2 (coordenadas homogéneas 3xnPoints).
    x3Data: Puntos observados en la imagen 3 (coordenadas homogéneas 3xnPoints).
    K_c: Matriz intrínseca de calibración (3x3).
    nPoints: Número de puntos 3D.
    -output:
    res: Residuales (errores de proyección) entre los puntos observados y proyectados.
    """

    # Extrae rotación y traslación para T_21 y T_31
    theta_21 = Op[:3]       # Rotación para T_21 en so(3)
    t_21 = Op[3:6]          # Traslación para T_21
    theta_31 = Op[6:9]      # Rotación para T_31 en so(3)
    t_31 = Op[9:12]         # Traslación para T_31

    # Extrae los puntos 3D en la referencia de la primera cámara
    X1 = Op[12:].reshape(3, nPoints)

    # Convierte theta_21 y theta_31 en matrices de rotación usando la exponencial
    R_21 = expm(crossMatrix(theta_21))
    R_31 = expm(crossMatrix(theta_31))

    # Construye las matrices de transformación homogéneas T_21 y T_31
    T_21 = np.eye(4)
    T_21[:3, :3] = R_21
    T_21[:3, 3] = t_21

    T_31 = np.eye(4)
    T_31[:3, :3] = R_31
    T_31[:3, 3] = t_31

    # Proyección en la primera cámara
    x1_proj_hom = K_c @ X1
    x1_proj = x1_proj_hom[:2, :] / x1_proj_hom[2, :]  # Normalizar por coordenada z

    # Proyección en la segunda cámara (transformar X1 a la referencia 2)
    X2 = T_21[:3, :3] @ X1 + T_21[:3, 3:4]
    x2_proj_hom = K_c @ X2
    x2_proj = x2_proj_hom[:2, :] / x2_proj_hom[2, :]  # Normalizar por coordenada z

    # Proyección en la tercera cámara (transformar X1 a la referencia 3)
    X3 = T_31[:3, :3] @ X1 + T_31[:3, 3:4]
    x3_proj_hom = K_c @ X3
    x3_proj = x3_proj_hom[:2, :] / x3_proj_hom[2, :]  # Normalizar por coordenada z

    # Calcula los residuales entre los puntos proyectados y observados
    residuals_x1 = (x1_proj - x1Data[:2, :]).flatten()
    residuals_x2 = (x2_proj - x2Data[:2, :]).flatten()
    residuals_x3 = (x3_proj - x3Data[:2, :]).flatten()

    # Concatenar los residuales para las tres proyecciones
    residuals = np.hstack((residuals_x1, residuals_x2, residuals_x3))
    
    return residuals


