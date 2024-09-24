import numpy as np
import matplotlib.pyplot as plt


"""
FUNCIONES PARA LOS EJERCICIOS 1 Y 2
"""
def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensambla una matriz SE(3) con la matriz de rotación y el vector de traslación.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    
    # Return inverse
    return np.linalg.inv(T_w_c)

def get_projection_matrix(K, R, t):
    """Calcula la matriz de proyección P = K * [R | t]"""
    T = ensamble_T(R, t)
    P = K @ T[:3, :]  # Matriz de proyección
    return P

def project_points(P, points_3D_hom, return_homogeneous=False):
    projected_points_hom = P @ points_3D_hom.T  # Proyección
    projected_points_hom_normalized = projected_points_hom / projected_points_hom[2]
    # Convertir de homogéneo a cartesiano
    if return_homogeneous:
        # Devolver las 3 coordenadas homogéneas normalizadas (con la tercera componente igual a 1)
        return projected_points_hom_normalized.T
    else:
        # Devolver solo las 2 primeras coordenadas normalizadas (formato cartesiano)
        projected_points_2D = projected_points_hom_normalized[:2]
        return projected_points_2D.T

def compute_line(p1, p2):
    """Calcula la recta que pasa por dos puntos"""
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = p1[0]*p2[1] - p2[0]*p1[1]
    return a, b, c

def compute_intersection(line1, line2):
    """Calcula la intersección de dos rectas"""
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1*b2 - a2*b1
    x = (b1*c2 - b2*c1) / det
    y = (a2*c1 - a1*c2) / det
    return x, y




"""
FUNCIONES PARA LOS EJERCICIOS 3 Y 4
"""
def distance_point_to_plane(point, a, b, c, d):
    x, y, z = point
    return np.abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)


