import utils.geometry as geometry
import numpy as np

def projectionMatrix(K, T_w_c):
    R_w_c = T_w_c[0:3, 0:3]  # La rotación: parte superior izquierda (3x3)
    t_w_c = T_w_c[0:3, 3]    # La traslación: última columna (3x1)
    
    T_w_c = geometry.ensamble_T(R_w_c, t_w_c)

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


def compute_reprojection_error(K, T_opt, X_w_opt, xData):
    total_error = 0.0
    total_points = 0

    for i, T_wc in enumerate(T_opt):
        x_proj = project_points(K, T_wc, X_w_opt)[:2, :]
        x_obs = xData[i][:2, :]  # Observaciones originales

        # Error euclidiano por punto
        errors = np.linalg.norm(x_proj - x_obs, axis=0)
        total_error += np.sum(errors)
        total_points += errors.size

    mean_error = total_error / total_points
    return mean_error

