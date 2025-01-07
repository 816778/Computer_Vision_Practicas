import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm, logm

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


def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return np.linalg.inv(T_w_c)


def projectionMatrix(K, T_w_c):
    R_w_c = T_w_c[0:3, 0:3]  # La rotación: parte superior izquierda (3x3)
    t_w_c = T_w_c[0:3, 3]    # La traslación: última columna (3x1)
    
    Rt = ensamble_T(R_w_c, t_w_c)
    # Rt = np.linalg.inv(T_w_c)
    
    # K*[R|t] para obtener la matriz de proyección
    P = K @ Rt[0:3, :]
    return P



def print_projected_with_homography(H, path_image_1, path_image_2, matches, title='Matches projected using Homography'):
    # Load the images
    image1 = cv2.imread(path_image_1)
    image2 = cv2.imread(path_image_2)

    # Extract the matching points from matches (the first 2 columns are from image1, the last 2 are from image2)
    keypoints1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in matches[:, :2]]
    keypoints2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in matches[:, 2:4]]

    # Project points from image1 to image2 using the homography matrix
    projected_points = cv2.perspectiveTransform(np.array([matches[:, :2]]), H)
    projected_points = projected_points[0]  # Get rid of the extra dimension

    # Convert the projected points into cv2.KeyPoint format to visualize with drawMatches
    projected_keypoints = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in projected_points]

    # Create matches between the original keypoints in image1 and the projected points in image2
    dMatchesList = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(keypoints1))]

    # Draw matches between keypoints1 (from image1) and projected_keypoints (projected points in image2)
    img_matches = cv2.drawMatches(
        image1, keypoints1, image2, projected_keypoints, dMatchesList, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Show the result
    plt.imshow(img_matches, cmap='gray')
    plt.title(title)
    plt.subplots_adjust(
        top=0.985,     # Border for top
        bottom=0.015,  # Border for bottom
        left=0.028,    # Border for left
        right=0.992,   # Border for right
    )

    plt.show()

def compute_homography(x1, x2):
    """
    Calcula la homografía que mapea los puntos x1 a x2.
    :param x1: Puntos en la primera imagen, de forma (2xN) o (3xN) en coordenadas homogéneas.
    :param x2: Puntos en la segunda imagen, de forma (2xN) o (3xN) en coordenadas homogéneas.
    :return: Matriz de homografía 3x3 que mapea x1 a x2.
    """

    assert x1.shape[1] == x2.shape[1], "x1 y x2 deben tener el mismo número de puntos."
    assert x1.shape[0] == 2 or x1.shape[0] == 3, "Los puntos x1 deben tener forma 2xN o 3xN."
    assert x2.shape[0] == 2 or x2.shape[0] == 3, "Los puntos x2 deben tener forma 2xN o 3xN."

    # Asegurar que los puntos están en coordenadas homogéneas (3xN)
    if x1.shape[0] == 2:
        x1 = np.vstack((x1, np.ones((1, x1.shape[1]))))  # Agregar una fila de 1s
    if x2.shape[0] == 2:
        x2 = np.vstack((x2, np.ones((1, x2.shape[1]))))  # Agregar una fila de 1s

    N = x1.shape[1]  # Número de puntos
    A = []

    # Para cada correspondencia de puntos (x1, x2), construimos el sistema de ecuaciones
    for i in range(N):
        X1 = x1[:, i]
        X2 = x2[:, i]
        x_1, y_1, _ = X1
        x_2, y_2, _ = X2

        # Dos filas de la matriz A por cada correspondencia de puntos
        A.append([-x_1, -y_1, -1, 0, 0, 0, x_2 * x_1, x_2 * y_1, x_2])
        A.append([0, 0, 0, -x_1, -y_1, -1, y_2 * x_1, y_2 * y_1, y_2])

    A = np.array(A)

    # Resolver el sistema de ecuaciones usando SVD (Descomposición en Valores Singulares)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)  # Última fila de Vt es la solución

    # Normalizar H para que H[2, 2] sea 1
    H = H / H[2, 2]

    return H


def ransac_homography(matches, num_iterations, threshold, display_interval=20000):
    best_inliers_count = 0
    best_homography = None
    best_inliers = None
    n_points = 4  # Mínimo número de puntos para calcular la homografía

    print(f"Shape of matches: {matches.shape}")

    for i in range(num_iterations):
        # Seleccionar 4 puntos aleatorios
        sample_idx = np.random.choice(matches.shape[0], n_points, replace=False)
        src_points = matches[sample_idx, :2].T
        dst_points = matches[sample_idx, 2:4].T

        # Calcular homografía con los 4 puntos seleccionados
        H = compute_homography(src_points, dst_points)

        if H is None:
            continue

        # Transformar puntos usando la homografía
        projected_points = cv2.perspectiveTransform(np.array([matches[:, :2]]), H)
        errors = np.sqrt(np.sum((matches[:, 2:4] - projected_points[0]) ** 2, axis=1))

        # Calcular inliers
        inliers = errors < threshold
        inliers_count = np.sum(inliers)

        if i == display_interval:
            print_projected_with_homography(H, 'images/image1.png', 'images/image1.png', matches, title='Homography random')

        # Mostrar los 4 puntos de la hipótesis actual
        #if i % 20 == 0:  # Mostrar cada 20 iteraciones
        #    display_matches(matches, inliers, src_points, dst_points, H, title=f"Iteration {i}")

        # Verificar si es la mejor hipótesis
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_homography = H
            best_inliers = inliers

    # Visualizar la mejor hipótesis
    #display_matches(matches, best_inliers, matches[best_inliers, :2], matches[best_inliers, 2:4], best_homography, title="Best Hypothesis")


    return best_homography, best_inliers



def estimate_fundamental_8point(x1, x2):
    """
    Estima la matriz fundamental usando el método de los ocho puntos.
    
    Args:
        x1: Puntos en la primera imagen, tamaño (3, N).
        x2: Puntos en la segunda imagen, tamaño (3, N).
        
    Returns:
        F: Matriz fundamental estimada de tamaño (3, 3).
    """

    if x1.shape[0] == 2:
        x1 = np.vstack((x1, np.ones((1, x1.shape[1]))))
    if x2.shape[0] == 2:
        x2 = np.vstack((x2, np.ones((1, x2.shape[1]))))
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


def ransac_fundamental_matrix_optimized(matches, num_iterations, threshold):
    """
    Estima la matriz fundamental utilizando RANSAC.

    Parameters:
        matches (np.ndarray): Matriz con correspondencias [x1, y1, x2, y2].
        num_iterations (int): Número de iteraciones de RANSAC.
        threshold (float): Umbral de error para considerar inliers.

    Returns:
        best_F (np.ndarray): Matriz fundamental estimada.
        best_inliers (list): Índices de los inliers.
    """
    best_F = None
    best_inliers_count = 0
    best_inliers = []

    for _ in range(num_iterations):
        # Seleccionar 8 puntos aleatorios
        sample_idx = np.random.choice(matches.shape[0], 8, replace=False)
        src_points = matches[sample_idx, :2]
        dst_points = matches[sample_idx, 2:4]

        # Calcular la matriz fundamental con los 8 puntos seleccionados
        F = estimate_fundamental_8point(src_points.T, dst_points.T)

        if F is None:
            continue

        # Calcular el error de transferencia para todos los emparejamientos
        x1 = np.hstack((matches[:, :2], np.ones((matches.shape[0], 1))))  # Puntos homogéneos en la primera imagen
        x2 = np.hstack((matches[:, 2:4], np.ones((matches.shape[0], 1))))  # Puntos homogéneos en la segunda imagen

        # Líneas epipolares en la segunda imagen
        l2 = F @ x1.T
        l2 /= np.sqrt(l2[0, :]**2 + l2[1, :]**2)  # Normalizar las líneas

        # Cálculo del error de transferencia
        errors = np.abs(np.sum(l2.T * x2, axis=1))  # Distancia punto-línea

        # Determinar los inliers
        inliers = np.where(errors < threshold)[0]
        inliers_count = len(inliers)

        # Actualizar la mejor matriz fundamental si es necesario
        if inliers_count > best_inliers_count:
            best_F = F
            best_inliers_count = inliers_count
            best_inliers = inliers

    return best_F, best_inliers


def ransac_fundamental_matrix(matches, num_iterations, threshold):
    best_F = None
    best_inliers_count = 0
    best_inliers = []

    for _ in range(num_iterations):
        # Seleccionar 8 puntos aleatorios
        sample_idx = np.random.choice(matches.shape[0], 8, replace=False)
        src_points = matches[sample_idx, :2].T
        dst_points = matches[sample_idx, 2:4].T

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




def line_points_distance(line, points):
    """
    Calcula la distancia entre una línea y un punto.
    
    Args:
        line: Coeficientes de la línea, tamaño (3,).
        point: Coordenadas del punto, tamaño (2,).
        
    Returns:
        float: Distancia entre la línea y el punto.
    """

    d = []

    for point in points:
        a, b, c = line
        x, y = point
        d.append(abs(a*x + b*y + c) / np.sqrt(a**2 + b**2))

    # Convert list of lists to list
    d = [item for sublist in d for item in sublist]

    return d



def matchEpipolar(x1, x2, F, minDist=100, ratio=0.8):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.
    -input:
        x1: Interest points in image 1 
        x2: Interest points in image 2
    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    #print(x1)

    matches = np.array([])
    for p1 in range(x1.shape[0]):
        x = np.array([x1[p1, 0], x1[p1, 1], 1]).reshape(3, 1)

        # Compute the epipolar line
        l = F @ x

        # Compute the distance from the points in 2 to the line
        d = line_points_distance(l, x2)

        # Sort by distance
        indexSort = np.argsort(d)

        if d[indexSort[0]] < minDist and (ratio < 0 or (ratio > 0 and d[indexSort[0]] < ratio * d[indexSort[1]])):
            # Add pair x1, x2 to matches
            match = np.array([x1[p1, 0], x1[p1, 1], x2[indexSort[0], 0], x2[indexSort[0], 1]])
            matches = np.append(matches, match)

    # Add epipolar matches
    matches = matches.reshape(-1, 4)
    
    return matches


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



def triangulate_multiview(P_list, puntos_por_imagen):
    """
    Triangula puntos 3D usando múltiples vistas.

    Args:
        P_list: Lista de matrices de cámara proyectivas (P = K * [R | t]).
        puntos_por_imagen: Lista de arrays (x, y) de puntos 2D observados en cada vista.

    Returns:
        X_w: Nube de puntos 3D reconstruida.
    """
    num_puntos = puntos_por_imagen[0].shape[1]
    num_vistas = len(P_list)
    X_w = []

    for i in range(num_puntos):
        A = []
        for j in range(num_vistas):
            coso = puntos_por_imagen[j][:, i]
            x, y = puntos_por_imagen[j][:, i]
            P = P_list[j]
            # Construir las ecuaciones lineales para la triangulación
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        A = np.array(A)
        # Resolver usando SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[-1]  # Homogeneizar
        X_w.append(X[:3])
    
    return np.array(X_w).T  # (3, N)


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
    # P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    # P2 = np.hstack((R, t.reshape(3, 1)))
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


def linearPoseEstimation(F, srcPts, dstPts, K1, K2):
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
    
    #matches = np.hstack((srcPts.T, dstPts.T))
    E = np.dot(K1.T, np.dot(F, K2)) 


    # Descomponer la matriz esencial E en 4 posibles soluciones
    R12_1, R12_2, t12, _ = decompose_essential_matrix(E)

    # Seleccionar la solución correcta triangulando los puntos 3D
    R12, t12 = select_correct_pose(R12_1, R12_2, t12, K1, K2, srcPts, dstPts)

    return R12, t12


def linearPoseEstimation_withoutF(srcPts, dstPts, K_c):
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
    
    #matches = np.hstack((srcPts.T, dstPts.T))
    #F, _ = ransac_fundamental_matrix(matches, 1000, 1.5)
    F = estimate_fundamental_8point(srcPts, dstPts)

    #F, mask = cv2.findFundamentalMat(srcPts.T, dstPts.T, cv2.FM_8POINT)

    E = K_c.T @ F @ K_c

    # Descomponer la matriz esencial E en 4 posibles soluciones
    R12_1, R12_2, t12, _ = decompose_essential_matrix(E)

    # Seleccionar la solución correcta triangulando los puntos 3D
    R12, t12 = select_correct_pose(R12_1, R12_2, t12, K_c, K_c, srcPts, dstPts)

    return R12, t12

