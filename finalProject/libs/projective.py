import numpy as np
import libs.geometric_math as math


def projectionMatrix(K, T_w_c):
    R_w_c = T_w_c[0:3, 0:3]  # La rotación: parte superior izquierda (3x3)
    t_w_c = T_w_c[0:3, 3]    # La traslación: última columna (3x1)
    
    T_w_c = math.ensamble_T(R_w_c, t_w_c)

    # Invert to get world from the camera
    T_c_w = np.linalg.inv(T_w_c)
    
    # K*[R|t] para obtener la matriz de proyección
    P = K @ T_c_w[0:3, :]
    return P


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



def triangulate_points(P1, P2, x1, x2):
    """
    Triangular puntos 3D a partir de las matrices de proyección P1 y P2 y las correspondencias x1 y x2.
    """
    if x1.shape[0] == 2:
        x1 = np.vstack((x1, np.ones((1, x1.shape[1]))))  
    if x2.shape[0] == 2:
        x2 = np.vstack((x2, np.ones((1, x2.shape[1])))) 

    num_points = x1.shape[1]
    X_homogeneous = np.zeros((4, num_points))

    # Triangular cada par de puntos
    for i in range(num_points):
        # Para cada punto, construir la matriz A para el sistema de ecuaciones
        A = np.array([
            x1[0, i] * P1[2, :] - P1[0, :],  # Ecuación para x1 en la primera cámara
            x1[1, i] * P1[2, :] - P1[1, :],  # Ecuación para y1 en la primera cámara
            x2[0, i] * P2[2, :] - P2[0, :],  # Ecuación para x2 en la segunda cámara
            x2[1, i] * P2[2, :] - P2[1, :]   # Ecuación para y2 en la segunda cámara
        ])

        # Resolver el sistema usando SVD
        _, _, Vt = np.linalg.svd(A)
        X_homogeneous[:, i] = Vt[-1]  # Última fila de Vt es la solución homogénea
    
    # Convertir a coordenadas 3D dividiendo por la cuarta coordenada
    X = X_homogeneous / X_homogeneous[3, :]
    return X[:3]