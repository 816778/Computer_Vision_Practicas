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
from scipy.linalg import expm, logm



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


def triangulate_multiview(P_list, puntos_por_imagen):
    """
    Triangula puntos 3D usando múltiples vistas.

    Args:
        P_list: Lista de matrices de cámara proyectivas (P = K * [R | t]).
        puntos_por_imagen: Lista de arrays (x, y) de puntos 2D observados en cada vista.

    Returns:
        X_w: Nube de puntos 3D reconstruida.
    """
    num_puntos = puntos_por_imagen[0].shape[1]
    num_vistas = len(P_list)
    X_w = []

    for i in range(num_puntos):
        A = []
        for j in range(num_vistas):
            coso = puntos_por_imagen[j][:, i]
            x, y = puntos_por_imagen[j][:, i]
            P = P_list[j]
            # Construir las ecuaciones lineales para la triangulación
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        A = np.array(A)
        # Resolver usando SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[-1]  # Homogeneizar
        X_w.append(X[:3])
    
    return np.array(X_w).T  # (3, N)


def crossMatrixInv(M):
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return x


def residual_bundle_adjustment(params, K, xData, nImages):

    X_w = params[nImages*6:].reshape(-1, 3).T
    residuals = np.array([])

    for i in range(nImages):
        t = params[i*6:(i*6)+3]
        th = params[(i*6)+3:(i*6)+6]
        R = expm(crossMatrix(th))

        T_wc = np.zeros((3, 4))
        T_wc[:3, :3] = R
        T_wc[:3, 3] = t

        x_proj = project_points(K, T_wc, X_w)

        residuals = np.hstack((residuals,
            ((x_proj[:2, :] - xData[i][:2, :])).flatten()
        ))

    print("Residuals: ", residuals.mean())

    return residuals



def crossMatrix(x):
    """Genera la matriz skew-symmetric de un vector."""
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def run_bundle_adjustment(T, K, X_w, xData):
    if X_w.shape[0] == 4:
        X_w = (X_w[:3, :] / X_w[3, :])
    
    nImages = len(T)

    T_flat = np.array([])
    for i in range(nImages):
        t = T[i][:3, 3]
        R = T[i][:3, :3]
        th = crossMatrixInv(logm(R))

        T_flat = np.hstack((T_flat, t, th))

    X_w_flat = X_w.T.flatten()

    initial_params = np.hstack((T_flat, X_w_flat))

    result = scOptim.least_squares(residual_bundle_adjustment, initial_params,
                                   args=(K, xData, nImages), method='lm')
    
    optimized_params = result.x

    T_opt = []
    for i in range(nImages):
        t = optimized_params[i*6:(i*6)+3]
        th = optimized_params[(i*6)+3:(i*6)+6]
        R = expm(crossMatrix(th))

        T_wc = np.zeros((3, 4))
        T_wc[:3, :3] = R
        T_wc[:3, 3] = t

        T_opt.append(T_wc)

    X_w_opt = optimized_params[nImages*6:].reshape(-1, 3).T

    return T_opt, X_w_opt



def triangulate_multiview_from_T(T_list, K, puntos_por_imagen):
    """
    Triangula puntos 3D usando múltiples vistas, a partir de T_wc.

    Args:
        T_list: Lista de matrices de transformaciones T_wc (4x4) para cada cámara.
        K: Matriz intrínseca de la cámara.
        puntos_por_imagen: Lista de arrays (x, y) de puntos 2D observados en cada vista.

    Returns:
        X_w: Nube de puntos 3D reconstruida.
    """
    num_puntos = puntos_por_imagen[0].shape[1]
    num_vistas = len(T_list)
    X_w = []

    # Construir las matrices proyectivas P a partir de T_wc y K
    P_list = []
    for T_wc in T_list:
        R = T_wc[:3, :3]
        t = T_wc[:3, 3]
        P = K @ np.hstack((R, t.reshape(-1, 1)))  # P = K * [R | t]
        P_list.append(P)

    for i in range(num_puntos):
        A = []
        for j in range(num_vistas):
            x, y = puntos_por_imagen[j][:, i]
            P = P_list[j]
            # Construir las ecuaciones lineales para la triangulación
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        A = np.array(A)
        # Resolver usando SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[-1]  # Homogeneizar
        X_w.append(X[:3])
    
    return np.array(X_w).T



def residual_bundle_adjustment_multiviwe(params, K, xData, nImages):

    X_w = params[nImages*6:].reshape(-1, 3).T
    residuals = np.array([])
    T_list = []
    for i in range(nImages):
        t = params[i*6:(i*6)+3]
        th = params[(i*6)+3:(i*6)+6]
        R = expm(crossMatrix(th))

        T_wc = np.zeros((3, 4))
        T_wc[:3, :3] = R
        T_wc[:3, 3] = t
        T_list.append(T_wc)

    x_proj = triangulate_multiview_from_T(T_list, K, T_wc, X_w)

    residuals = np.hstack((residuals,
        ((x_proj[:2, :] - xData[i][:2, :])).flatten()
    ))

    print("Residuals: ", residuals.mean())

    return residuals

