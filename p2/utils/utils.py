import numpy as np
import matplotlib.pyplot as plt


"""
FUNCIONES PARA LOS EJERCICIOS 1 Y 2
"""
def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c


def compute_projection_matrix(K, T_w_c):
    R_w_c = T_w_c[0:3, 0:3]  # La rotación es la parte superior izquierda (3x3)
    t_w_c = T_w_c[0:3, 3]    # La traslación es la última columna (3x1)
    
    # Rt = np.hstack((R_w_c, t_w_c.reshape(-1, 1)))
    Rt = ensamble_T(R_w_c, t_w_c)
    
    # K*[R|t] para obtener la matriz de proyección
    P = K @ Rt[0:3, :]
    return P


def project_points(P, X_w):
    """
    Proyecta puntos 3D en una imagen usando la matriz de proyección.
    - P: matriz de proyección 3x4 de la cámara.
    - X_w: puntos 3D en coordenadas del mundo (4xN).
    """
    x_proj = P @ X_w  # Proyectar puntos 3D a 2D
    x_proj /= x_proj[2]  # Normalizar las coordenadas homogéneas
    return x_proj[:2, :]  # Devolver coordenadas 2D