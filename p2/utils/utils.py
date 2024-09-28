import numpy as np
import matplotlib.pyplot as plt
import cv2

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
    # Extraer rotación y traslación
    R_w_c1 = T_w_c1[:3, :3]  
    t_w_c1 = T_w_c1[:3, 3]  
    R_w_c2 = T_w_c2[:3, :3]  
    t_w_c2 = T_w_c2[:3, 3] 

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
    
    # Asegurarse de que U y Vt son matrices de rotación
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

def triangulate_points(P1, P2, x1, x2):
    """
    Triangular puntos 3D a partir de las matrices de proyección P1 y P2 y las correspondencias x1 y x2.
    """
    # Triangular los puntos
    X_homogeneous = cv2.triangulatePoints(P1, P2, x1[:2], x2[:2])

    # Convertir a coordenadas 3D
    X = X_homogeneous / X_homogeneous[3]
    return X[:3]


def select_correct_pose(R1, R2, t, K1, K2, x1, x2):
    """
    Selecciona la correcta entre las cuatro posibles soluciones triangulando los puntos 3D y verificando
    que estén delante de las cámaras.
    """
    # Matriz de proyección de la primera cámara
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # Posibles matrices de proyección para la segunda cámara
    P2_options = [
        K2 @ np.hstack((R1, t.reshape(3, 1))),
        K2 @ np.hstack((R1, -t.reshape(3, 1))),
        K2 @ np.hstack((R2, t.reshape(3, 1))),
        K2 @ np.hstack((R2, -t.reshape(3, 1)))
    ]

    # Para cada opción, triangula los puntos y verifica si están delante de las cámaras
    for i, P2 in enumerate(P2_options):
        X = triangulate_points(P1, P2, x1, x2)

        # Verificar si los puntos triangulados están delante de ambas cámaras (Z > 0)
        if np.all(X[2, :] > 0):
            print(f"Solución correcta encontrada: Opción {i + 1}")
            return P2

    print("No se encontró una solución válida.")
    return None