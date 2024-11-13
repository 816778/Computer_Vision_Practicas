import numpy as np
import scipy.optimize as scOptim
from scipy.linalg import expm, logm
import libs.geometric_math as math
import cv2

import libs.projective as projective
import libs.geometric_math as math
import libs.matching as matching



def is_valid_camera(R, t, K1, K2, pts1, pts2):
    """
    Verifica si una solución (R, t) genera puntos 3D válidos (delante de ambas cámaras).
    Params:
        R (np.ndarray): Matriz de rotación.
        t (np.ndarray): Vector de traslación.
        K1 (np.ndarray): Matriz intrínseca de la primera cámara.
        K2 (np.ndarray): Matriz intrínseca de la segunda cámara.
        pts1 (np.ndarray): Puntos en la primera imagen.
        pts2 (np.ndarray): Puntos en la segunda imagen.
    
    Returns:
        bool: True si los puntos están delante de ambas cámaras, False en caso contrario.
    """
    # Construcción de las matrices de proyección para ambas cámaras
    #P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Cámara 1 en el origen
    #P2 = K2 @ np.hstack((R, t.reshape(3, 1)))  # Cámara 2 con rotación y traslación
    
    P1 = projective.projectionMatrix(K1, np.eye(4))
    P2 = projective.projectionMatrix(K2, math.ensamble_T(R, t))

    # Triangular los puntos 3D
    pts_3d = projective.triangulate_points(P1, P2, pts1, pts2)
    if pts_3d.shape[0] == 3:
        pts_3d = pts_3d.T

    # Verificar que los puntos tengan la coordenada Z positiva (delante de ambas cámaras)
    pts_cam1 = pts_3d[:, 2]  # Coordenada Z en la cámara 1
    pts_cam2 = (R @ pts_3d.T + t.reshape(-1, 1))[2, :]  # Coordenada Z en la cámara 2

    return np.all(pts_cam1 > 0) and np.all(pts_cam2 > 0)



def select_correct_pose(R1, R2, t, K1, K2, pts1, pts2):
    """
    Selecciona la solución correcta de rotación y traslación que da como resultado puntos 3D
    válidos, es decir, que están delante de ambas cámaras.

    Params:
        R1 (np.ndarray): Primera matriz de rotación.
        R2 (np.ndarray): Segunda matriz de rotación.
        t (np.ndarray): Vector de traslación.
        K (np.ndarray): Matriz intrínseca de la cámara.
        pts1 (np.ndarray): Puntos en la primera imagen.
        pts2 (np.ndarray): Puntos en la segunda imagen.

    Returns:
        R_correct (np.ndarray): La rotación correcta.
        t_correct (np.ndarray): La traslación correcta.
    """
    T_21_solutions = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    for i, (R, t_vec) in enumerate(T_21_solutions):
        if is_valid_camera(R, t_vec, K1, K2, pts1, pts2):
            print(f"Solución correcta: R{(i//2)+1}, t{'+' if i % 2 == 0 else '-'}")
            return R, t_vec

    # Si ninguna solución es válida, se puede manejar un caso por defecto.
    raise ValueError("No se encontró una solución válida.")



def fundamental_8point(x1, x2):
    """
    Estima la matriz fundamental usando el método de los ocho puntos.
    
    Args:
        x1: Puntos en la primera imagen, tamaño (3, N).
        x2: Puntos en la segunda imagen, tamaño (3, N).
        
    Returns:
        F: Matriz fundamental estimada de tamaño (3, 3).
    """
    assert x1.shape[1] >= 8, "Necesitas al menos 8 puntos para aplicar el método de los 8 puntos"
    
    # Construir la matriz A
    A = np.zeros((x1.shape[1], 9))

    A[:, 0] = x2[0, :] * x1[0, :]
    A[:, 1] = x2[0, :] * x1[1, :]
    A[:, 2] = x2[0, :]
    A[:, 3] = x2[1, :] * x1[0, :]
    A[:, 4] = x2[1, :] * x1[1, :]
    A[:, 5] = x2[1, :]
    A[:, 6] = x1[0, :]
    A[:, 7] = x1[1, :]
    A[:, 8] = 1
    
    # Resolver Af = 0 usando SVD
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Aplicar la restricción de rango 2
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # Forzar el último valor singular a 0
    F = U @ np.diag(S) @ Vt

    return F



def fundamental_matrix_ransac(srcPts, dstPts, num_iterations, threshold):
    best_F = None
    best_inliers_count = 0

    for _ in range(num_iterations):
        # Seleccionar 8 puntos aleatorios
        sample_idx = np.random.choice(srcPts.shape[1], 8, replace=False)
        src = srcPts[:, sample_idx]
        dst = dstPts[:, sample_idx]

        # Calcular la matriz fundamental con 8 puntos seleccionados
        F = fundamental_8point(src, dst)

        lines = F @ srcPts
        lines /= np.sqrt(lines[0]**2 + lines[1]**2)
        errors = math.multi_line_point_distance(lines, dstPts)

        # Calcular el número de inliers
        nInliers = np.sum(errors < threshold)

        # Actualizar la mejor matriz fundamental
        if nInliers > best_inliers_count:
            best_F = F
            best_inliers_count = nInliers

    return best_F, best_inliers_count


def residual_bundle_adjustment(params, K, xData, nImages, verbose):

    X_w = params[nImages*6:].reshape(-1, 3).T
    residuals = np.array([])

    for i in range(nImages):
        t = params[i*6:(i*6)+3]
        th = params[(i*6)+3:(i*6)+6]
        R = expm(math.crossMatrix(th))

        T_wc = np.zeros((3, 4))
        T_wc[:3, :3] = R
        T_wc[:3, 3] = t

        x_proj = projective.project_points(K, T_wc, X_w)

        residuals = np.hstack((residuals,
            ((x_proj[:2, :] - xData[i][:2, :])).flatten()
        ))

    if verbose:
        print("Residuals: ", residuals.mean())

    return residuals


def run_bundle_adjustment(T, K, X_w, xData, verbose=False):
    
    nImages = len(T)
    if X_w.shape[0] == 4:
        X_w = (X_w[:3, :] / X_w[3, :])

    T_flat = np.array([])
    for i in range(nImages):
        t = T[i][:3, 3]
        R = T[i][:3, :3]
        th = math.crossMatrixInv(logm(R))

        T_flat = np.hstack((T_flat, t, th))

    X_w_flat = X_w.T.flatten()
    initial_params = np.hstack((T_flat, X_w_flat))

    result = scOptim.least_squares(residual_bundle_adjustment, initial_params,
                                   args=(K, xData, nImages, verbose), method='lm')
    
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


def decompose_essential_matrix(E):
    """
    Descompone la matriz esencial E en las cuatro posibles soluciones de R (rotación) y t (traslación).
    """
    U, _, Vt = np.linalg.svd(E)
    
    # Comprobar que U y Vt son matrices de rotación
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    # Definir W para la descomposición
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    # Dos posibles rotaciones
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Dos posibles traslaciones (normalizada)
    t = U[:, 2]  # vector de traslación
    return R1, R2, t, -t




# Calcular la matriz fundamental
def compute_fundamental_matrix(T_w_c1, T_w_c2, K_c):
    """
    Calcula la matriz fundamental F a partir de las poses y la matriz intrínseca.
    """

    # Calcular la rotación y traslación relativas entre las dos cámaras
    T_c2_c1 = np.linalg.inv(T_w_c2) @ T_w_c1
    R_c2_c1 = T_c2_c1[:3, :3] 
    t_c2_c1 = T_c2_c1[:3, 3]  

    # Calcular la matriz de esencialidad
    E = math.skew(t_c2_c1) @ R_c2_c1

    # Calcular la matriz fundamental
    F = np.linalg.inv(K_c).T @ E @ np.linalg.inv(K_c)
    return F



def linearPoseEstimation(srcPts, dstPts, K_c):
    """
    Crea listas de coincidencias y convierte los puntos en coordenadas homogéneas.
    """

    matches = np.hstack((srcPts.T, dstPts.T))
    F, nInliers = fundamental_matrix_ransac(srcPts, dstPts, 1000, 1)
    #F = fundamental_8point(srcPts, dstPts)

    print("Fundamental matrix estimated with", nInliers, "inliers")


    #F, mask = cv2.findFundamentalMat(srcPts.T, dstPts.T, cv2.FM_8POINT)

    E = K_c.T @ F @ K_c

    # Descomponer la matriz esencial E en 4 posibles soluciones
    R21_1, R21_2, t21, _ = decompose_essential_matrix(E)

    # Seleccionar la solución correcta triangulando los puntos 3D
    R21, t21 = select_correct_pose(R21_1, R21_2, t21, K_c, K_c, srcPts, dstPts)

    T_c2c1 = math.ensamble_T(R21, t21)
    T_c1c2 = np.linalg.inv(T_c2c1)

    return T_c1c2


def bundlePoseEstimation(x1, x2, K_c, verbose=False):
    """
    Estima la rotación y traslación entre dos cámaras usando la técnica de bundle adjustment.

    Args:
    - srcPts: Puntos observados en la primera imagen (2D).
    - dstPts: Puntos observados en la segunda imagen (2D).
    - K_c: Matriz intrínseca de la cámara.

    Returns:
    - R12: Matriz de rotación entre la primera y segunda cámara.
    - t12: Vector de traslación entre la primera y segunda cámara.
    """

    T_wc1 = np.eye(4)  # se toma la primera cámara como referencia
    T_c1c2 = linearPoseEstimation(x1, x2, K_c)
    
    P1 = projective.projectionMatrix(K_c, T_wc1) # K_c @ T_wc1[0:3, :]
    P2 = projective.projectionMatrix(K_c, T_c1c2) # K_c @ T_wc2[0:3, :]

    X_w = projective.triangulate_points(P1, P2, x1, x2)
    T_opt, X_w_opt = run_bundle_adjustment([T_wc1, T_c1c2], K_c, X_w, [x1, x2], verbose)

    return T_opt[0], T_opt[1], X_w_opt
    #return T_wc1, T_c1c2, X_w


def add_new_view(prevImage, newImage, prevSpData, X_w, model, device, K_c):

    spData23, spData3, mask23 = matching.match_superglue(prevImage, newImage, model, device, prevSpData)
    _, x3_2 = matching.tensors_to_matches(spData23, spData3)

    if X_w.shape[1] < 4:
        raise ValueError("Se requieren al menos 4 puntos para solvePnP con SOLVEPNP_EPNP")

    objectPoints = X_w.T[mask23].astype(np.float64)
    imagePoints = np.ascontiguousarray(x3_2[0:2,:].T).reshape((x3_2.shape[1], 1, 2))
    #imagePoints = x3_2.T

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

    return T_wc3

def add_new_points(prevImage, newImage, Pprev, Pnew, model, device):
        
    # Add new 3d points from camera 3
    spData1, spData2, _ = matching.match_superglue(prevImage, newImage, model, device)
    x1, x2 = matching.tensors_to_matches(spData1, spData2)

    X_w = projective.triangulate_points(Pprev, Pnew, x1, x2)

    return X_w




