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
        matchesList.append([np.int32(dMatchesList[k].queryIdx), np.int32(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList


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




def is_valid_solution(R, t, K1, K2, pts1, pts2):
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
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Cámara 1 en el origen
    P2 = K2 @ np.hstack((R, t.reshape(3, 1)))  # Cámara 2 con rotación y traslación
    
    # Triangular los puntos 3D
    pts_3d = triangulate_points(P1, P2, pts1, pts2)
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
        if is_valid_solution(R, t_vec, K1, K2, pts1, pts2):
            print(f"Solución correcta: R{(i//2)+1}, t{'+' if i % 2 == 0 else '-'}")
            return R, t_vec

    # Si ninguna solución es válida, se puede manejar un caso por defecto.
    raise ValueError("No se encontró una solución válida.")


def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return np.linalg.inv(T_w_c)


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


def project_points_exp_rot(K, T_w_c, rot, X_w):
    if T_w_c.shape == (3, 4):
        T_w_c = np.vstack([T_w_c, [0, 0, 0, 1]])

    if X_w.shape[0] == 3:
        X_w = np.vstack([X_w, np.ones((1, X_w.shape[1]))])

    # Rt = np.linalg.inv(T_w_c)
    
    R_w_c = expm(crossMatrix(rot))

    T = ensamble_T(R_w_c, T_w_c)    
    x_proj = project_points(K, T, X_w)
    
    return x_proj


def crossMatrixInv(M):
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return x

def crossMatrix(x):
    M = np.array([[0, -x[2], x[1]],
    [x[2], 0, -x[0]],
    [-x[1], x[0], 0]], dtype="object")
    return M

def skew(t):
    """Devuelve la matriz antisimétrica de la traslación t."""
    return np.array([[0, -t[2], t[1]],
                     [t[2], 0, -t[0]],
                     [-t[1], t[0], 0]])


                     
def estimate_fundamental_8point(x1, x2):
    """
    Estima la matriz fundamental usando el método de los ocho puntos.
    
    Args:
        x1: Puntos en la primera imagen, tamaño (3, N).
        x2: Puntos en la segunda imagen, tamaño (3, N).
        
    Returns:
        F: Matriz fundamental estimada de tamaño (3, 3).
    """
    assert x1.shape[1] >= 8, "Necesitas al menos 8 puntos para aplicar el método de los 8 puntos"
    
    # Normalización de las coordenadas
    x1 = x1 / x1[2, :]
    x2 = x2 / x2[2, :]

    # Construir la matriz A
    A = []
    for i in range(x1.shape[1]):
        x1_i = x1[0, i]
        y1_i = x1[1, i]
        x2_i = x2[0, i]
        y2_i = x2[1, i]
        A.append([x2_i * x1_i, x2_i * y1_i, x2_i,
                  y2_i * x1_i, y2_i * y1_i, y2_i,
                  x1_i, y1_i, 1])
    
    A = np.array(A)

    # Resolver Af = 0 usando SVD
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Aplicar la restricción de rango 2
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # Forzar el último valor singular a 0
    F = U @ np.diag(S) @ Vt

    return F



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



def residual_bundle_adjustmentNormalRot(params, K, x1Data, x2Data, x3Data):

    T_wc1 = params[:12].reshape(3, 4)
    T_wc2 = params[12:24].reshape(3, 4)
    T_wc3 = params[24:36].reshape(3, 4)
    X_w = params[36:].reshape(-1, 3).T
    
    x1_proj = project_points(K, T_wc1, X_w)
    x2_proj = project_points(K, T_wc2, X_w)
    x3_proj = project_points(K, T_wc3, X_w)
    
    residuals = np.hstack((
        ((x1_proj[:2, :] - x1Data)*(x1_proj[:2, :] - x1Data)).flatten(),
        ((x2_proj[:2, :] - x2Data)*(x2_proj[:2, :] - x2Data)).flatten(),
        ((x3_proj[:2, :] - x3Data)*(x3_proj[:2, :] - x3Data)).flatten()
    ))

    print("Residuals: ", residuals.mean())

    return residuals

def residual_bundle_adjustment(params, K, x1Data, x2Data, x3Data):
    t1 = params[:3]
    r1 = params[3:6]
    t2 = params[6:9]
    r2 = params[9:12]
    t3 = params[12:15]
    r3 = params[15:18]

    X_w = params[18:].reshape(-1, 3).T
    
    x1_proj = project_points_exp_rot(K, t1, r1, X_w)
    x2_proj = project_points_exp_rot(K, t2, r2, X_w)
    x3_proj = project_points_exp_rot(K, t3, r3, X_w)
    
    residuals = np.hstack((
        ((x1_proj[:2, :] - x1Data)*(x1_proj[:2, :] - x1Data)).flatten(),
        ((x2_proj[:2, :] - x2Data)*(x2_proj[:2, :] - x2Data)).flatten(),
        ((x3_proj[:2, :] - x3Data)*(x3_proj[:2, :] - x3Data)).flatten()
    ))

    print("Residuals: ", residuals.mean())

    return residuals


def run_bundle_adjustmentFull(T_wc1, T_wc2, T_wc3, K, X_w, x1Data, x2Data, x3Data):
    
    if X_w.shape[0] == 4:
        X_w = (X_w[:3, :] / X_w[3, :])
    
    T_wc1_flat = T_wc1[:3, :].flatten()
    T_wc2_flat = T_wc2[:3, :].flatten()
    T_wc3_flat = T_wc3[:3, :].flatten()
    X_w_flat = X_w.T.flatten()

    initial_params = np.hstack((T_wc1_flat, T_wc2_flat, T_wc3_flat, X_w_flat))
    
    result = scOptim.least_squares(residual_bundle_adjustmentNormalRot, initial_params,
                                   args=(K, x1Data, x2Data, x3Data), method='lm')
    
    optimized_params = result.x
    T_wc1_opt = optimized_params[:12].reshape(3, 4)
    T_wc2_opt = optimized_params[12:24].reshape(3, 4)
    T_wc3_opt = optimized_params[24:36].reshape(3, 4)
    X_w_opt = optimized_params[36:].reshape(-1, 3).T
    
    return T_wc1_opt, T_wc2_opt, T_wc3_opt, X_w_opt



def run_bundle_adjustment(T_wc1, T_wc2, T_wc3, K, X_w, x1Data, x2Data, x3Data):
    if X_w.shape[0] == 4:
        X_w = (X_w[:3, :] / X_w[3, :])
    
    t1 = T_wc1[:3, 3]
    R1 = T_wc1[:3, :3]
    th1 = crossMatrixInv(sc.linalg.logm(R1))

    t2 = T_wc2[:3, 3]
    R2 = T_wc2[:3, :3]
    th2 = crossMatrixInv(sc.linalg.logm(R2))

    t3 = T_wc3[:3, 3]
    R3 = T_wc3[:3, :3]
    th3 = crossMatrixInv(sc.linalg.logm(R3))

    X_w_flat = X_w.T.flatten()

    initial_params = np.hstack((t1, th1, t2, th2, t3, th3, X_w_flat))
    
    result = scOptim.least_squares(residual_bundle_adjustment, initial_params,
                                   args=(K, x1Data, x2Data, x3Data), method='lm')
    
    optimized_params = result.x

    t1_opt = optimized_params[:3]
    th1_opt = optimized_params[3:6]
    t2_opt = optimized_params[6:9]
    th2_opt = optimized_params[9:12]
    t3_opt = optimized_params[12:15]
    th3_opt = optimized_params[15:18]
    X_w_opt = optimized_params[18:].reshape(-1, 3).T

    R1_opt = expm(crossMatrix(th1_opt))
    R2_opt = expm(crossMatrix(th2_opt))
    R3_opt = expm(crossMatrix(th3_opt))

    T_wc1_opt = np.eye(4)
    T_wc1_opt[:3, :3] = R1_opt
    T_wc1_opt[:3, 3] = t1_opt

    T_wc2_opt = np.eye(4)
    T_wc2_opt[:3, :3] = R2_opt
    T_wc2_opt[:3, 3] = t2_opt

    T_wc3_opt = np.eye(4)
    T_wc3_opt[:3, :3] = R3_opt
    T_wc3_opt[:3, 3] = t3_opt
    
    return T_wc1_opt, T_wc2_opt, T_wc3_opt, X_w_opt



def rotvec_to_rotmat(rvec):
    """Convierte un vector de rotación theta en una matriz de rotación."""
    theta = np.linalg.norm(rvec)
    if theta == 0:
        return np.eye(3)
    
    axis = rvec / theta
    skew_rvec = crossMatrix(axis)
    R = np.eye(3) + np.sin(theta) * skew_rvec + (1 - np.cos(theta)) * (skew_rvec @ skew_rvec)
    return R.astype(np.float64)

def crossMatrix(x):
    """Genera la matriz skew-symmetric de un vector."""
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])



def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
    """
    -input:
        Op: Optimization parameters: this must include aparamtrization for T_21 (reference 1 seen from reference 2) in a proper way and for X1 (3D points in ref 1)
        x1Data: (3xnPoints) 2D points on image 1 (homogeneous coordinates)
        x2Data: (3xnPoints) 2D points on image 2 (homogeneous coordinates)
        K_c: (3x3) Intrinsic calibration matrix nPoints: Number of points
    -output:
        res: residuals from the error between the 2D matched points and the projected points from the 3D points (2 equations/residuals per 2D point)
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


def resBundleProjectionThreeViews(Op, K_c, x1Data, x2Data, x3Data):
    """
    -input:
    Op: Vector de optimización que incluye una parametrización para T_21 y T_31, 
        y los puntos 3D en el sistema de la cámara 1.
    x1Data: Puntos observados en la imagen 1 (coordenadas homogéneas 3xnPoints).
    x2Data: Puntos observados en la imagen 2 (coordenadas homogéneas 3xnPoints).
    x3Data: Puntos observados en la imagen 3 (coordenadas homogéneas 3xnPoints).
    K_c: Matriz intrínseca de calibración (3x3).
    -output:
    res: Residuales (errores de proyección) entre los puntos observados y proyectados.
    """

    nPoints = x1Data.shape[1]

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






def visualize_matches(path_image_1, path_image_2, maxDist, distRatio, draw=True):
    image1 = cv2.imread(path_image_1)
    image2 = cv2.imread(path_image_2)

    # Feature extraction
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    matchesList = matchWith2NDRR(descriptors1, descriptors2, distRatio=distRatio, maxDist=maxDist)
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)

    # Dibujar los primeros 100 emparejamientos
    img_matched = cv2.drawMatches(
        image1, keypoints1, image2, keypoints2, dMatchesList[:100], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    if draw:
        plt.imshow(img_matched, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Emparejamientos con distRatio = {distRatio}")
        plt.subplots_adjust(
            top=0.985,     # Border for top
            bottom=0.015,  # Border for bottom
            left=0.028,    # Border for left
            right=0.992,   # Border for right
        )
        plt.draw()
        plt.waitforbuttonpress()
    
    return dMatchesList, keypoints1, keypoints2  # Retornar las coincidencias para análisis posterior



def matchWith2NDRR(desc1, desc2, distRatio, maxDist=100):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.
    -input:
        desc1: descriptors from image 1 nDesc x 128
        desc2: descriptors from image 2 nDesc x 128
        distRatio:
    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        # Ordena las distancias y selecciona los índices de los dos más cercanos
        indexSort = np.argsort(dist)
        d1 = dist[indexSort[0]]  # Distancia al vecino más cercano
        d2 = dist[indexSort[1]] # Distancia al segundo vecino más cercano

        if d1 < d2 * distRatio and d1 < maxDist:
            matches.append([kDesc1, indexSort[0], d1])
    
    return np.array(matches)


def ransac_fundamental_matrix(matches, num_iterations, threshold):
    best_F = None
    best_inliers_count = 0
    best_inliers = []

    for _ in range(num_iterations):
        # Seleccionar 8 puntos aleatorios
        sample_idx = np.random.choice(matches.shape[0], 8, replace=False)
        src_points = matches[sample_idx, :3].T
        dst_points = matches[sample_idx, 3:6].T

        # Calcular la matriz fundamental con 8 puntos seleccionados
        F = estimate_fundamental_8point(src_points, dst_points)

        # Calcular el error de transferencia para todos los emparejamientos
        inliers = []
        for i in range(matches.shape[0]):
            x1 = np.append(matches[i, :2], 1)  # Punto en la primera imagen (homogéneo)
            x2 = np.append(matches[i, 2:4], 1) # Punto en la segunda imagen (homogéneo)

            l2 = F @ x1  # Línea epipolar en la segunda imagen
            l2 /= np.sqrt(l2[0]**2 + l2[1]**2)  # Normalizar la línea
            error = line_point_distance(l2, x2[:2])

            if error < threshold:
                inliers.append(i)

        # Actualizar la mejor matriz fundamental
        if len(inliers) > best_inliers_count:
            best_F = F
            best_inliers_count = len(inliers)
            best_inliers = inliers

    return best_F, best_inliers


def line_point_distance(line, point):
    """
    Calcula la distancia entre una línea y un punto.
    
    Args:
        line: Coeficientes de la línea, tamaño (3,).
        point: Coordenadas del punto, tamaño (2,).
        
    Returns:
        float: Distancia entre la línea y el punto.
    """

    a, b, c = line
    x, y = point
    return abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)



def do_matches(path_image_1='images/image1.png', path_image_2='images/image2.png', option=0, draw=False):
    if option == 0:
        path = './results/image1_image2_matches.npz'
        npz = np.load(path)

        keypoints1 = npz['keypoints0'] 
        keypoints2 = npz['keypoints1']  
        matches = npz['matches']

        valid_matches_idx = matches > -1
        keypoints1_matched = keypoints1[valid_matches_idx]
        keypoints2_matched = keypoints2[matches[valid_matches_idx]]

        keypoints_cv1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in keypoints1_matched]
        keypoints_cv2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in keypoints2_matched]

        # Crear objetos DMatch con índices secuenciales
        dMatchesList = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(keypoints_cv1))]


        # Convierte los emparejamientos a coordenadas (x, y)
        srcPts = keypoints1_matched  # Ya es un array con coordenadas (x, y)
        dstPts = keypoints2_matched
        x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
        x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))

        matched_points = np.hstack((x1, x2))
        matched_points = np.hstack((srcPts, dstPts))

    elif option == 1:
        distRatio = 0.75
        maxDist = 500
        dMatchesList, keypoints1, keypoints2 = visualize_matches(path_image_1, path_image_2, maxDist, distRatio, draw=draw)

        print("Total de keypoints en la primera imagen:", len(keypoints1))
        print("Total de keypoints en la segunda imagen:", len(keypoints2))

        # Convierte los emparejamientos a coordenadas (x, y)
        srcPts = np.float32([keypoints1[m.queryIdx].pt for m in dMatchesList]).T
        dstPts = np.float32([keypoints2[m.trainIdx].pt for m in dMatchesList]).T
        x1 = np.vstack((srcPts, np.ones((1, srcPts.shape[1])))).T
        x2 = np.vstack((dstPts, np.ones((1, dstPts.shape[1])))).T

        matched_points = np.hstack((x1, x2))
        #matched_points = np.hstack((srcPts, dstPts))

    else:
        matched_points = None

    return matched_points, srcPts, dstPts

####################################################################################


def create_match_lists(kpCv1, kpCv_other, x1Data, x_otherData, K_c):
    """
    Crea listas de coincidencias y convierte los puntos en coordenadas homogéneas.

    Args:
    - kpCv1: Lista de puntos clave (KeyPoint) en la primera imagen.
    - kpCv_other: Lista de puntos clave (KeyPoint) en la segunda/tercera imagen.
    - x1Data: Puntos observados en la primera imagen (2D).
    - x_otherData: Puntos observados en la segunda/tercera imagen (2D).

    Returns:
    - srcPts_hom: Puntos fuente en coordenadas homogéneas.
    - dstPts_hom: Puntos destino en coordenadas homogéneas.
    """
    # Crear lista de coincidencias (matches)
    matchesList = np.hstack((
        np.reshape(np.arange(0, x1Data.shape[1]), (x_otherData.shape[1], 1)),
        np.reshape(np.arange(0, x1Data.shape[1]), (x1Data.shape[1], 1)),
        np.ones((x1Data.shape[1], 1))
    ))

    dMatchesList = utils.indexMatrixToMatchesList(matchesList)

    # Extraer puntos coincidentes en las dos vistas
    srcPts = np.float32([kpCv1[m.queryIdx].pt for m in dMatchesList])
    dstPts = np.float32([kpCv_other[m.trainIdx].pt for m in dMatchesList])

    # Convertir a coordenadas homogéneas
    srcPts = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    dstPts = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))

    F = estimate_fundamental_8point(srcPts, dstPts)

    E = K_c.T @ F @ K_c

    # Descomponer la matriz esencial E en 4 posibles soluciones
    R12_1, R12_2, t12, _ = decompose_essential_matrix(E)

    # Seleccionar la solución correcta triangulando los puntos 3D
    R12, t12 = select_correct_pose(R12_1, R12_2, t12, K_c, K_c, srcPts, dstPts)

    return srcPts, dstPts, R12, t12



def resBundleProjection_2(Op, x1Data, x2Data, K_c, nPoints):
    """
    -input:
        Op: Optimization parameters: this must include aparamtrization for T_21 (reference 1 seen from reference 2) in a proper way and for X1 (3D points in ref 1)
        x1Data: (3xnPoints) 2D points on image 1 (homogeneous coordinates)
        x2Data: (3xnPoints) 2D points on image 2 (homogeneous coordinates)
        K_c: (3x3) Intrinsic calibration matrix 
        nPoints: Number of points
    -output:
        res: residuals from the error between the 2D matched points and the projected points from the 3D points (2 equations/residuals per 2D point)
    """
    theta = Op[:3]
    T = Op[3:6].reshape(3, 1)
    X1 = Op[6:].reshape(nPoints, 3).T  # (3xnPoints)
    
    R = expm(crossMatrix(theta)) 

    X2 = R @ X1 + T  # X2 en coordenadas de la cámara 2

    x1_proj = K_c @ X1  # Proyección en la cámara 1
    x2_proj = K_c @ X2  # Proyección en la cámara 2

    # Convertir a coordenadas inhomogéneas 
    x1_proj = x1_proj[:2, :] / x1_proj[2, :]
    x2_proj = x2_proj[:2, :] / x2_proj[2, :]

    # Calcular los residuales (diferencias entre puntos observados y proyectados)
    res1 = (x1_proj - x1Data) ** 2  
    res2 = (x2_proj - x2Data) ** 2
    
    # Concatenar los residuales en un solo vector
    res = np.hstack((res1.ravel(), res2.ravel()))  # Cada punto contribuye con 2 residuales (x, y)

    print("Residuals: ", res.mean())

    return res



def extract_theta_and_t_from_T(T_wc):
    """
    Extrae los parámetros theta (rotación) y t (traslación) de una matriz de transformación T_wc.
    
    Args:
    - T_wc: Matriz de transformación 4x4 de la cámara en el mundo.
    
    Returns:
    - theta: Vector de rotación de 3 elementos en so(3).
    - t: Vector de traslación de 3 elementos.
    """
    # Extraer la rotación y traslación de T_wc
    R_wc = T_wc[:3, :3]  # Matriz de rotación 3x3
    t_wc = T_wc[:3, 3]   # Vector de traslación 3x1
    
    # Convertir la matriz de rotación R_wc a la representación theta (so(3))
    theta_matrix = logm(R_wc)  # Calcular la matriz logarítmica de R_wc
    theta = crossMatrixInv(theta_matrix)
    
    return theta, t_wc


def construct_T_from_theta_and_t(theta, t):
    """
    Construye la matriz de transformación T_wc a partir de theta (rotación) y t (traslación).
    
    Args:
    - theta: Vector de rotación de 3 elementos en so(3).
    - t: Vector de traslación de 3 elementos.
    
    Returns:
    - T_wc: Matriz de transformación 4x4 de la cámara en el mundo.
    """
    # Convertir theta (en so(3)) a la matriz de rotación R usando la exponencial de matrices
    R_wc = expm(crossMatrix(theta)) 

    # Crear la matriz de transformación T_wc de 4x4
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc  # Colocar la matriz de rotación en T_wc
    T_wc[:3, 3] = t.flatten()  # Colocar el vector de traslación en T_wc

    return T_wc



def residual_bundle_adjustment_multiple_views(params, K, x_data_list, num_cameras, num_points):
    """
    Calcula los residuales para un ajuste de bundle con múltiples vistas.
    
    - params: vector unidimensional con parámetros de rotación y traslación para las cámaras,
              y coordenadas de los puntos 3D optimizados.
    - K: matriz intrínseca de la cámara.
    - x_data_list: lista de puntos 2D observados en cada cámara (ground truth).
    - num_cameras: número de cámaras (vistas).
    - num_points: número de puntos 3D.
    """
    
    # Extraer los parámetros de cada cámara y los puntos 3D
    # Cámara 1 no se optimiza, por lo que no tiene parámetros en 'params'
    
    camera_params = []
    index = 0
    
    # Cámara 2: optimización con grados de libertad restringidos si es necesario
    T_wc2 = params[index:index + 6]  # 3 para rotación, 3 para traslación
    index += 6
    camera_params.append(T_wc2)

    # Cámaras adicionales (3 en adelante): 6 grados de libertad cada una
    for _ in range(2, num_cameras):  # Desde la cámara 3 en adelante
        T_wci = params[index:index + 6]  # 3 para rotación, 3 para traslación
        index += 6
        camera_params.append(T_wci)
    
    # Puntos 3D en el espacio mundial
    X_w = params[index:].reshape((3, num_points))

    # Inicializar lista de residuales
    residuals = []

    # Proyectar puntos y calcular residuales para cada cámara
    for i, (T_wci, x_data) in enumerate(zip(camera_params, x_data_list)):
        # Convertir el vector de rotación y traslación a una matriz de transformación
        R = rotvec_to_rotmat(T_wci[:3])  # rotación
        t = T_wci[3:].reshape((3, 1))    # traslación
        
        T_wci_matrix = np.eye(4)
        T_wci_matrix[:3, :3] = R
        T_wci_matrix[:3, 3] = t.flatten()

        # Proyectar puntos 3D en la imagen de la cámara i+1
        x_proj = project_points(K, T_wci_matrix, X_w)

        # Calcular el error cuadrático de reproyección
        reprojection_error = (x_proj[:2, :] - x_data) ** 2
        residuals.append(reprojection_error.flatten())

    # Concatenar todos los errores de reproyección en un vector unidimensional
    return np.hstack(residuals)
