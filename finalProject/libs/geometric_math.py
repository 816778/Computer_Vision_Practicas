import numpy as np



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

        

def line_point_distance(line, point):
    """
    Calcula la distancia entre una línea y un punto.
    
    Args:
        line: Coeficientes de la línea, tamaño (3,).
        point: Coordenadas homogéneas del punto, tamaño (3,).
        
    Returns:
        float: Distancia entre la línea y el punto.
    """

    a, b, c = line
    x, y, h = point
    return abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)

def multi_line_point_distance(lines, points):
    """
    Calcula la distancia entre múltiples líneas y puntos.
    
    Args:
        lines: Coeficientes de las líneas, tamaño (3, n).
        points: Coordenadas homogéneas de los puntos, tamaño (3, n).
        
    Returns:
        float: Distancia entre las líneas y los puntos.
    """

    d = np.abs(np.sum(lines * points, axis=0) / np.sqrt(np.sum(lines[:2, :]**2, axis=0)))

    return d

def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c

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



def kannala_brandt_unprojection(u, K, D):
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

