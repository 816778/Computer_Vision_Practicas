import numpy as np
import geometric_math as math


def define_planes(v):
    """
    Define los planos Pi_sym y Pi_perp para un vector de dirección dado en 3D.
    
    Parámetros:
        v: Vector de dirección en 3D (forma (3,))
    
    Retorno:
        Pi_sym, Pi_perp: Los planos definidos por el vector v
    """
    vx, vy, vz = v
    Pi_sym = np.array([-vy, vx, 0, 0])  # Pi_sym según la imagen
    Pi_perp = np.array([-vx * vy, -vy**2, vx**2 + vy**2, 0])  # Pi_perp según la imagen
    return Pi_sym, Pi_perp


def triangulate_point_fisheye(directions1, directions2, T_wc1, T_wc2, T_c1c2):
    """
    Triangula un punto 3D usando la técnica de triangulación basada en planos.
    
    Parámetros:
        directions1: Direcciones en la cámara 1 (3,)
        directions2: Direcciones en la cámara 2 (3,)
        T_wc1: Transformación de la cámara 1 al sistema mundial (4, 4)
        T_wc2: Transformación de la cámara 2 al sistema mundial (4, 4)
    
    Retorno:
        Punto 3D en coordenadas homogéneas.
    """
    # Definir los planos en el sistema de la primera cámara
    Pi_sym_1, Pi_perp_1 = define_planes(directions1)
    Pi_sym_2, Pi_perp_2 = define_planes(directions2)

    # Transformar los planos al sistema de la segunda cámara
    Pi_sym_2_1 = T_c1c2.T @ Pi_sym_1
    Pi_perp_2_1 = T_c1c2.T @ Pi_perp_1

    # Construir la matriz A para el sistema AX = 0
    A = np.vstack((Pi_sym_2_1.T, Pi_perp_2_1.T, Pi_sym_2.T, Pi_perp_2.T))

    # Resolver AX = 0 usando SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X /= X[3]  # Normalizar para obtener coordenadas homogéneas

    return (T_wc2 @ X)[:3] # To world coordinates

def triangulate_points_fisheye(directions1, directions2, T_wc1, T_wc2, T_leftRight):
    """
    Triangula múltiples puntos 3D para un sistema estéreo.
    
    Parámetros:
        directions1: Direcciones en la cámara 1 (3, N)
        directions2: Direcciones en la cámara 2 (3, N)
        T_wc1: Transformación de la cámara 1 al sistema mundial (4, 4)
        T_wc2: Transformación de la cámara 2 al sistema mundial (4, 4)
    
    Retorno:
        Puntos 3D triangulados (3, N).
    """
    points_3d = []
    for v1, v2 in zip(directions1.T, directions2.T):
        X = triangulate_point_fisheye(v1, v2, T_wc1, T_wc2, T_leftRight)
        points_3d.append(X)

    return np.array(points_3d).T


def residual_bundle_adjustment_fisheye(params, K_1, K_2, D1_k_array, D2_k_array, xData, T_wc1, T_wc2, nImages):

    X_w = params[nImages*6:].reshape(-1, 3).T
    residuals = np.array([])

    for i in range(nImages):
        t = params[i*6:(i*6)+3]
        th = params[(i*6)+3:(i*6)+6]
        R = expm(math.crossMatrix(th))

        T_wc = np.eye(4)
        T_wc[:3, :3] = R
        T_wc[:3, 3] = t

        if i == 0 or i == 1:
            x_proj = math.project_points_fisheye(X_w, K_1, D1_k_array, T_wc1 @ T_wc)
        elif i == 2 or i == 3:
            x_proj = math.project_points_fisheye(X_w, K_2, D2_k_array, T_wc2 @ T_wc)
       

        residuals = np.hstack((residuals,
            ((x_proj[:2, :] - xData[i][:2, :])).flatten()
        ))

    print("Residuals: ", residuals.mean())

    return residuals


def run_bundle_adjustment_fisheye(T, K_1, K_2, D1_k_array, D2_k_array, X_w, xData, T_wc1, T_wc2):
    if X_w.shape[0] == 4:
        X_w = (X_w[:3, :] / X_w[3, :])
    
    nImages = len(T)

    T_flat = np.array([])
    for i in range(nImages):
        t = T[i][:3, 3]
        R = T[i][:3, :3]
        th = math.crossMatrixInv(logm(R))
        T_flat = np.hstack((T_flat, t, th))

    X_w_flat = X_w.T.flatten()

    initial_params = np.hstack((T_flat, X_w_flat))
    
    result = scOptim.least_squares(residual_bundle_adjustment_fisheye, initial_params,
                                   args=(K_1, K_2, D1_k_array, D2_k_array, xData, T_wc1, T_wc2, nImages), method='lm')
    
    optimized_params = result.x

    T_opt = []
    for i in range(nImages):
        t = optimized_params[i*6:(i*6)+3]
        th = optimized_params[(i*6)+3:(i*6)+6]
        R = expm(math.crossMatrix(th))

        T_wc = np.zeros((3, 4))
        T_wc[:3, :3] = R
        T_wc[:3, 3] = t
        T_opt.append(T_wc)

    X_w_opt = optimized_params[nImages*6:].reshape(-1, 3).T

    return T_opt, X_w_opt



