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
import time


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


                     

# Calcular la matriz fundamental
def calculate_fundamental_matrix(T_w_c1, T_w_c2, K_c):
    """
    Calcula la matriz fundamental F a partir de las poses y la matriz intrínseca.
    """

    # Calcular la rotación y traslación relativas entre las dos cámaras
    T_c2_c1 = np.linalg.inv(T_w_c2) @ T_w_c1
    R_c2_c1 = T_c2_c1[:3, :3] 
    t_c2_c1 = T_c2_c1[:3, 3]  

    # Calcular la matriz de esencialidad
    E = skew(t_c2_c1) @ R_c2_c1

    # Calcular la matriz fundamental
    F = np.linalg.inv(K_c).T @ E @ np.linalg.inv(K_c)
    return F


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


def kannala_brandt_projection(X, K, D):
    """
    Proyección de Kannala-Brandt (3D a 2D).
    
    Parámetros:
        X: Arreglo de puntos 3D de forma (3, n)
        K: Matriz intrínseca de la cámara (3, 3)
        D: Array de coeficientes de distorsión (k1, k2, k3, k4)
        
    Retorno:
        Coordenadas 2D proyectadas en el plano de imagen en forma de arreglo (2, n)
    """
    # Extracción de parámetros intrínsecos
    alpha_x, alpha_y = K[0, 0], K[1, 1]
    c_x, c_y = K[0, 2], K[1, 2]

    x, y, z = X[0, :], X[1, :], X[2, :]
    
    R = np.sqrt(x**2 + y**2)
    theta = np.arctan2(R, z)
    
    d_theta = theta + D[0] * theta**3 + D[1] * theta**5 + D[2] * theta**7 + D[3] * theta**9

    phi = np.arctan2(y, x)
    
    u = alpha_x * d_theta * np.cos(phi) + c_x
    v = alpha_y * d_theta * np.sin(phi) + c_y
    
    return np.vstack((u, v))



def kannala_brandt_unprojection_newton(u, K, D, tol=1e-6, max_iter=10):
    """
    Desproyección de Kannala-Brandt (2D a 3D).
    
    Parámetros:
        u: Coordenadas 2D en el plano de imagen (2, n)
        K: Matriz intrínseca de la cámara (3, 3)
        D: Array de coeficientes de distorsión (k1, k2, k3, k4)
        tol: Tolerancia para el método iterativo (Newton-Raphson)
        max_iter: Máximo número de iteraciones para el método iterativo
        
    Retorno:
        Direcciones en el espacio 3D como un arreglo de (3, n)
    """
    # Extraer parámetros intrínsecos
    alpha_x, alpha_y = K[0, 0], K[1, 1]
    c_x, c_y = K[0, 2], K[1, 2]

    # Coordenadas normalizadas en el plano de la cámara
    x_c = (u[0, :] - c_x) / alpha_x
    y_c = (u[1, :] - c_y) / alpha_y

    # Cálculo de r y phi
    r = np.sqrt(x_c**2 + y_c**2)
    phi = np.arctan2(y_c, x_c)
    
    # Inicializar theta con el valor de r como aproximación inicial
    theta = r
    
    # Método iterativo (Newton-Raphson) para resolver r = d(theta)
    for _ in range(max_iter):
        # Cálculo de d(theta) y su derivada d'(theta)
        d_theta = theta + D[0] * theta**3 + D[1] * theta**5 + D[2] * theta**7 + D[3] * theta**9
        d_theta_prime = 1 + 3 * D[0] * theta**2 + 5 * D[1] * theta**4 + 7 * D[2] * theta**6 + 9 * D[3] * theta**8
        
        # Actualizar theta usando Newton-Raphson
        theta = theta - (d_theta - r) / d_theta_prime
        
        # Verificar convergencia
        if np.max(np.abs(d_theta - r)) < tol:
            break

    # Calcular la dirección en el espacio 3D
    v_x = np.sin(theta) * np.cos(phi)
    v_y = np.sin(theta) * np.sin(phi)
    v_z = np.cos(theta)
    
    return np.vstack((v_x, v_y, v_z))


def kannala_brandt_unprojection_roots(u, K, D):
    """
    Desproyección de Kannala-Brandt (2D a 3D) usando raíces de un polinomio de noveno grado.
    
    Parámetros:
        u: Coordenadas 2D en el plano de imagen (2, n)
        K: Matriz intrínseca de la cámara (3, 3)
        D: Array de coeficientes de distorsión [k1, k2, k3, k4]
        
    Retorno:
        Direcciones en el espacio 3D como un arreglo de (3, n)
    """
    # Extraer parámetros intrínsecos
    K_inv = np.linalg.inv(K)
    
    if u.shape[0] == 2:
        u = np.vstack((u, np.ones(u.shape[1])))

    # Coordenadas normalizadas en el plano de la cámara
    x_c = K_inv @ u

    # Cálculo de r y phi
    r = np.sqrt((x_c[0]**2 + x_c[1]**2) / x_c[2]**2)

    phi = np.arctan2(x_c[1], x_c[0])
    
    # Coeficientes de distorsión

    k1, k2, k3, k4, _ = D

    # Array para almacenar los valores de theta para cada punto
    theta_values = []
    
    # Resolver el polinomio para cada valor de r
    for radius in r:
        # Construir los coeficientes del polinomio en función del valor actual de r
        poly_coeffs = [k4, 0, k3, 0, k2, 0, k1, 0, 1, -radius]  # Coeficientes para el polinomio en theta

        # Resolver el polinomio para theta usando np.roots
        roots = np.roots(poly_coeffs)

        # Filtrar solo las raíces reales
        real_roots = roots[np.isreal(roots)].real

        # Seleccionar la raíz real positiva más cercana a radius como theta
        if len(real_roots) > 0:
            theta = real_roots[np.argmin(np.abs(real_roots - radius))]
        else:
            theta = 0  # Si no hay raíces reales, usamos 0 como fallback
        theta_values.append(theta)
    
    theta_values = np.array(theta_values)

    # Calcular la dirección en el espacio 3D
    v_x = np.sin(theta_values) * np.cos(phi)
    v_y = np.sin(theta_values) * np.sin(phi)
    v_z = np.cos(theta_values)
    
    return np.vstack((v_x, v_y, v_z))



##################################################################
## TRIANGULATE POINTS
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


def triangulate_point(directions1, directions2, T_wc1, T_wc2, T_c1c2):
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

def triangulate_points(directions1, directions2, T_wc1, T_wc2, T_leftRight):
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
        X = triangulate_point(v1, v2, T_wc1, T_wc2, T_leftRight)
        points_3d.append(X)

    return np.array(points_3d).T



##################################################################
# BUNDLE ADJUSTMENT
##################################################################
def project_points_fisheye(points_3D, K, D, T_wc):
    """
    Proyecta un punto 3D usando el modelo de lente ojo de pez Kannala-Brandt.
    
    Parámetros:
        point_3D: Coordenada del punto en el espacio.
        K: Matriz intrínseca de la cámara.
        D: Coeficientes de distorsión.
        T_wc: Transformación de la cámara en el sistema de referencia.

    Retorno:
        Punto proyectado en 2D en la imagen de la cámara.
    """
    # Transformar el punto al sistema de la cámara
    # print("Shape point_3D: ", point_3D.shape)
    # print("Shape T_wc: ", T_wc.shape)
    

    num_points = points_3D.shape[1]
    points_3D_hom = np.vstack((points_3D, np.ones((1, num_points))))
    points_cam_hom = T_wc @ points_3D_hom 
    points_cam = points_cam_hom[:3, :] / points_cam_hom[3, :]

    projected_2D = kannala_brandt_projection(points_cam, K, D)
    
    return projected_2D


def residual_bundle_adjustment_fisheye(params, K_1, K_2, D1_k_array, D2_k_array, xData, T_wc1, T_wc2, nImages):

    X_w = params[nImages*6:].reshape(-1, 3).T
    residuals = np.array([])

    for i in range(nImages):
        t = params[i*6:(i*6)+3]
        th = params[(i*6)+3:(i*6)+6]
        R = expm(crossMatrix(th))

        T_wc = np.eye(4)
        T_wc[:3, :3] = R
        T_wc[:3, 3] = t

        if i == 0 or i == 1:
            x_proj = project_points_fisheye(X_w, K_1, D1_k_array, T_wc1 @ T_wc)
        elif i == 2 or i == 3:
            x_proj = project_points_fisheye(X_w, K_2, D2_k_array, T_wc2 @ T_wc)
       

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
        th = crossMatrixInv(logm(R))
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
        R = expm(crossMatrix(th))

        T_wc = np.zeros((3, 4))
        T_wc[:3, :3] = R
        T_wc[:3, 3] = t
        T_opt.append(T_wc)

    X_w_opt = optimized_params[nImages*6:].reshape(-1, 3).T

    return T_opt, X_w_opt
