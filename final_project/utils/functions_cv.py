import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from collections import defaultdict




def match_features(reference_image_path, new_images):
    """
    Realiza el emparejamiento de puntos entre una imagen de referencia y un conjunto de imágenes nuevas.

    :param reference_image_path: Ruta de la imagen de referencia.
    :param new_images: Lista de rutas de imágenes nuevas.
    :return: Diccionario con correspondencias en formato {x12, x2, x13, x3, ...}.
    """
    # Leer imagen de referencia
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    # Inicializar SIFT
    sift = cv2.SIFT_create()

    # Detectar y describir características en la imagen de referencia
    keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_image, None)

    # Inicializar emparejador de características (BFMatcher)
    bf = cv2.BFMatcher()

    # Diccionario para guardar emparejamientos
    matches_dict = {}

    # Iterar sobre las imágenes nuevas
    for idx, img_path in enumerate(new_images, start=2):
        # Leer nueva imagen
        new_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Detectar y describir características en la nueva imagen
        keypoints_new, descriptors_new = sift.detectAndCompute(new_image, None)

        # Emparejar descriptores entre la imagen de referencia y la nueva imagen
        matches = bf.knnMatch(descriptors_ref, descriptors_new, k=2)

        # Aplicar la proporción de Lowe para filtrar buenos emparejamientos
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Extraer puntos emparejados (x_ref y x_new)
        x_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches])
        x_new = np.float32([keypoints_new[m.trainIdx].pt for m in good_matches])

        # Guardar resultados en el diccionario
        matches_dict[f"x1{idx}"] = x_ref
        matches_dict[f"x{idx}"] = x_new

    return matches_dict



def emparejamiento_sift_multiple_2(new_images, mtx):
    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []
    matches_list = []

    for img_path in new_images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)


    matcher = cv2.BFMatcher()
    for i in range(len(new_images) - 1):
        matches = matcher.knnMatch(descriptors_list[i], descriptors_list[i + 1], k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        matches_list.append(good_matches)

    # visualizar_emparejamientos(new_images, keypoints_list, matches_list)
    
    poses = [(np.eye(3), np.zeros((3, 1)))]
    for i in range(len(matches_list)):
        pts1 = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches_list[i]])
        pts2 = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in matches_list[i]])
        
        E, mask = cv2.findEssentialMat(pts1, pts2, mtx, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, mtx)
        prev_R, prev_t = poses[-1]
        poses.append((prev_R @ R, prev_t + prev_R @ t))
    
    points_3d = []
    for i in range(len(poses) - 1):
        R1, t1 = poses[i]
        R2, t2 = poses[i + 1]
        P1 = mtx @ np.hstack((R1, t1))
        P2 = mtx @ np.hstack((R2, t2))
        pts1 = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches_list[i]])
        pts2 = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in matches_list[i]])
        pts_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts_4d /= pts_4d[3]
        points_3d.append(pts_4d[:3].T)

    return poses, points_3d


def visualize_matches(reference_image_path, new_images, matches_dict):
    """
    Visualiza los emparejamientos entre la imagen de referencia y las imágenes nuevas.

    :param reference_image_path: Ruta de la imagen de referencia.
    :param new_images: Lista de rutas de imágenes nuevas.
    :param matches_dict: Diccionario con correspondencias generadas por match_features.
    """
    # Leer imagen de referencia
    reference_image = cv2.imread(reference_image_path)

    for idx, img_path in enumerate(new_images, start=2):
        # Leer nueva imagen
        new_image = cv2.imread(img_path)

        # Convertir a escala de grises para detectar keypoints
        reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

        # Inicializar SIFT y detectar puntos clave
        sift = cv2.SIFT_create()
        keypoints_ref = sift.detect(reference_gray, None)
        keypoints_new = sift.detect(new_gray, None)

        # Dibujar emparejamientos
        matches_to_draw = []
        x_ref = matches_dict[f"x1{idx}"]
        x_new = matches_dict[f"x{idx}"]
        for i in range(len(x_ref)):
            matches_to_draw.append(cv2.DMatch(i, i, 0))

        combined_image = cv2.drawMatches(
            reference_image, [cv2.KeyPoint(x[0], x[1], 1) for x in x_ref],
            new_image, [cv2.KeyPoint(x[0], x[1], 1) for x in x_new],
            matches_to_draw, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Mostrar imagen combinada
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Emparejamientos: Referencia ↔ Imagen {idx}")
        plt.axis('off')
        plt.show()


def visualizar_emparejamientos(new_images, keypoints_list, matches_list):
    """
    Visualiza los emparejamientos de puntos clave entre imágenes consecutivas.
    
    Args:
        new_images (list): Lista de rutas a las imágenes.
        keypoints_list (list): Lista de puntos clave detectados en cada imagen.
        matches_list (list): Lista de emparejamientos entre imágenes consecutivas.
    """
    for i in range(len(matches_list)):
        # Cargar las imágenes consecutivas
        img1 = cv2.imread(new_images[i], cv2.IMREAD_COLOR)
        img2 = cv2.imread(new_images[i + 1], cv2.IMREAD_COLOR)

        # Dibujar emparejamientos
        img_matches = cv2.drawMatches(
            cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), keypoints_list[i],
            cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), keypoints_list[i + 1],
            matches_list[i], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Mostrar las imágenes emparejadas
        plt.figure(figsize=(15, 10))
        plt.title(f"Emparejamientos entre Imagen {i} e Imagen {i + 1}")
        plt.imshow(img_matches)
        plt.axis("off")
        plt.show()



##############################################################################################
## AUXILIAR FUNCTIONS
##############################################################################################

def essential_matrix(K1, K2, F):
    E = K1.T @ F @ K2
    U, S, Vt = np.linalg.svd(E)
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = np.dot(U, np.dot(W, Vt))
    R2 = np.dot(U, np.dot(W.T, Vt))
    t = U[:, 2]
    return E, R1, R2, t




def triangulate_point(projection_matrices, image_points):
    """
    Triangulate a 3D point from multiple 2D image points and their camera projection matrices.

    Parameters:
    - projection_matrices (list of np.ndarray): List of 3x4 camera projection matrices (P_i).
    - image_points (list of tuple): List of 2D coordinates (x, y) observed in the images.

    Returns:
    - np.ndarray: The triangulated 3D point in homogeneous coordinates (X, Y, Z).
    """
    if len(projection_matrices) != len(image_points):
        raise ValueError("Number of projection matrices must match the number of image points.")

    # Construct the A matrix for the linear system AX = 0
    A = []
    for P, (x, y) in zip(projection_matrices, image_points):
        A.append(x * P[2, :] - P[0, :])  # Equation for x
        A.append(y * P[2, :] - P[1, :])  # Equation for y
    
    A = np.array(A)

    # Solve the linear system using Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[-1]  # The last row of Vt is the solution

    # Convert from homogeneous to 3D coordinates
    X = X_homogeneous[:-1] / X_homogeneous[-1]
    return X



def emparejamiento_sift_multiple_3(new_images, mtx, verbose=False):
    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []
    matches_list = []

    for img_path in new_images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)


    matcher = cv2.BFMatcher()
    for i in range(len(new_images) - 1):
        matches = matcher.knnMatch(descriptors_list[i], descriptors_list[i + 1], k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        matches_list.append(good_matches)

    ransac_matches = []  
    for i in range(len(matches_list)):
        # Obtener puntos clave de las dos imágenes actuales
        kp1 = np.array([keypoints_list[i][m.queryIdx].pt for m in matches_list[i]])
        kp2 = np.array([keypoints_list[i + 1][m.trainIdx].pt for m in matches_list[i]])

        # Convertir a formato float32 para OpenCV
        kp1 = np.float32(kp1)
        kp2 = np.float32(kp2)

        # Estimar la matriz fundamental usando RANSAC
        # También puede ser cv2.findHomography si estás interesado en la homografía
        F, mask = cv2.findFundamentalMat(kp1, kp2, method=cv2.FM_RANSAC, ransacReprojThreshold=1.0)

        # Filtrar emparejamientos usando la máscara
        ransac_good_matches = [matches_list[i][j] for j in range(len(mask)) if mask[j]]
        ransac_matches.append(ransac_good_matches)

    if verbose:
        visualizar_emparejamientos(new_images, keypoints_list, matches_list)
        visualizar_emparejamientos(new_images, keypoints_list, ransac_matches)
    
    poses = [(np.eye(3), np.zeros((3, 1)))]
    for i in range(len(matches_list)):
        pts1 = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches_list[i]])
        pts2 = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in matches_list[i]])
        
        E, mask = cv2.findEssentialMat(pts1, pts2, mtx, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, mtx)
        prev_R, prev_t = poses[-1]
        poses.append((prev_R @ R, prev_t + prev_R @ t))

    return poses, keypoints_list, matches_list



def emparejamiento_supergluemultiple(path_superglue, path_images, path_images_order, resize_dim=None, verbose=False, threshold=0.6):
    keypoints_list = []
    global_points = defaultdict(list)
    point_id = 0
    common_indices = defaultdict(set)

    for idx, (path_sg, (path_img1, path_img2)) in enumerate(zip(path_superglue, path_images)):
        print(f"Emparejando {path_img1} y {path_img2}...")
        
        npz = np.load(path_sg)
        keypoints1 = npz['keypoints0'] 
        keypoints2 = npz['keypoints1']  
        matches = npz['matches'] # también está match_confidence
    
        confidence = npz['match_confidence']
        valid_matches_idx = np.logical_and(matches > -1, confidence > threshold)
        # valid_matches_idx = matches > -1
        keypoints1_matched = keypoints1[valid_matches_idx]
        keypoints2_matched = keypoints2[matches[valid_matches_idx]]

        if idx == 0:
            keypoints_list.append(keypoints1)  # Imagen de referencia

        keypoints_list.append(keypoints2)

        if verbose:
            # Cargar imágenes
            img1 = cv2.imread(path_img1)
            img2 = cv2.imread(path_img2)

            # Opcional: Redimensionar las imágenes
            if resize_dim:
                img1 = cv2.resize(img1, resize_dim)
                img2 = cv2.resize(img2, resize_dim)

            keypoints_cv1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in keypoints1_matched]
            keypoints_cv2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in keypoints2_matched]
            matches_cv = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(keypoints_cv1))]
            img_matches = cv2.drawMatches(img1, keypoints_cv1, img2, keypoints_cv2, matches_cv, None,
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            plt.figure(figsize=(12, 6))
            plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
            plt.title(f"Matches entre {path_img1} y {path_img2}")
            plt.axis('off')
            plt.show()

        for i, kp1_idx in enumerate(np.where(valid_matches_idx)[0]):
            kp2_idx = matches[valid_matches_idx][i]
            global_points[kp1_idx].append((idx + 1, kp2_idx))
        
        if idx == 0:
            common_indices[0] = set(np.where(valid_matches_idx)[0])
        else:
            common_indices[0] = common_indices[0].intersection(np.where(valid_matches_idx)[0])

    # Actualizar índices comunes entre todas las imágenes
    common_coordinates = []
    for common_point_idx in common_indices[0]:
        coords = []
        coords.append(keypoints_list[0][common_point_idx])  # Coordenadas de la referencia
        for img_idx, pt_idx in global_points[common_point_idx]:
            coords.append(keypoints_list[img_idx][pt_idx])  # Coordenadas en las imágenes comparadas
        common_coordinates.append(coords)

    if verbose:
        for i, img_path in enumerate(path_images_order):
            img = cv2.imread(img_path)
            if resize_dim:
                    img = cv2.resize(img, resize_dim)
            for coord in common_coordinates:
                if i < len(coord):
                    x, y = coord[i]
                    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.imshow(f"Image {i}", img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    return common_coordinates



def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return np.linalg.inv(T_w_c)



def estimar_posiciones_camaras(common_coordinates, K):
    """
    Estima las posiciones de las cámaras a partir de los puntos clave comunes.

    Args:
        common_coordinates (list): Lista de puntos clave comunes entre todas las imágenes.
                                  Cada elemento debe ser una lista con las coordenadas de una misma característica en cada imagen.
        K (np.ndarray): Matriz intrínseca de la cámara.

    Returns:
        list: Lista de matrices de proyección para cada cámara.
        list: Lista de matrices de transformación homogénea (T_wc) para cada cámara.
    """
    # Separar las coordenadas comunes en diferentes imágenes
    puntos_por_imagen = [[] for _ in range(len(common_coordinates[0]))]

    for coord_list in common_coordinates:
        for i, punto in enumerate(coord_list):
            puntos_por_imagen[i].append(punto)

    # Convertir las listas en matrices numpy
    puntos_por_imagen = [np.array(puntos) for puntos in puntos_por_imagen]

    # Asumir que la cámara de referencia tiene la matriz de proyección P1 = K[I|0]
    P1 = np.hstack((K, np.zeros((3, 1))))
    T_wc1 = np.eye(4)

    proyecciones = [P1]  # Lista para almacenar las matrices de proyección
    T_list = [T_wc1]     # Lista para almacenar las transformaciones homogéneas

    # Calcular las matrices esenciales y descomponerlas para encontrar las posiciones de las cámaras
    for i in range(1, len(puntos_por_imagen)):
        x_ref = puntos_por_imagen[0]  # Puntos en la imagen de referencia
        x_curr = puntos_por_imagen[i]  # Puntos en la imagen actual

        # Calcular la matriz fundamental usando los puntos de correspondencia
        F, mask = cv2.findFundamentalMat(x_ref, x_curr, cv2.FM_RANSAC)

        # Calcular la matriz esencial
        E = K.T @ F @ K

        # Descomponer la matriz esencial en R y t
        _, R, t, _ = cv2.recoverPose(E, x_ref, x_curr, K)

        # Construir la matriz de transformación homogénea T_wc
        T_wc = ensamble_T(R, t.ravel())

        # Construir la matriz de proyección para la cámara actual
        P_curr = np.hstack((R, t))
        P_curr = K @ P_curr

        T_list.append(T_wc)
        proyecciones.append(P_curr)

    return proyecciones, T_list



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




def emparejamiento_sift_multiple_4(new_images, mtx, verbose=False):
    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []

    for img_path in new_images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
        print(f"Image {img_path} keypoints: {len(keypoints)}")

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    global_points = defaultdict(list)
    point_id = 0

    matcher = cv2.BFMatcher()
    for i in range(len(descriptors_list) - 1):
        desc1, desc2 = descriptors_list[0], descriptors_list[i + 1]
        kp1, kp2 = keypoints_list[0], keypoints_list[i + 1]
        
        matches = bf.knnMatch(desc1, desc2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 3, 0.99)
        inlier_matches = [good_matches[j] for j in range(len(good_matches)) if mask[j]]
        
        for match in inlier_matches:
            print(match)
            exit()
            query_idx = match.queryIdx
            train_idx = match.trainIdx

            # Buscar si el punto de la imagen i ya tiene un ID
            found = False
            for key, track in global_points.items():
                if track[-1][0] == i and track[-1][1] == query_idx:  # Match en la imagen actual
                    # Agregar el punto de la imagen i+1 al mismo track
                    track.append((i + 1, train_idx))
                    found = True
                    break
            if not found:
                global_points[point_id].append((i, query_idx))
                global_points[point_id].append((i + 1, train_idx))
                point_id += 1
    num_images = len(new_images)
    common_points = [key for key, track in global_points.items() if len(set([t[0] for t in track])) == num_images]
    

    common_coordinates = []
    for key in common_points:
        coords = []
        for img_idx, pt_idx in global_points[key]:
            coords.append(keypoints_list[img_idx][pt_idx].pt)
        common_coordinates.append(coords)

    """
    for i, img_path in enumerate(new_images):
        img = cv2.imread(img_path)
        for coord in common_coordinates:
            if i < len(coord):
                x, y = coord[i]
                cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.imshow(f"Image {i}", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    print(f"Common points: {len(common_points)}")
    
    """
    poses = [(np.eye(3), np.zeros((3, 1)))]
    for i in range(len(new_images)):
        
        pts1, pts2 = [], []
        for coord_list in common_coordinates:
            pts1 = coord_list[0]
            pts2 = coord_list[i+1]

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        print(f"pts1: {pts1.shape}")    
        
        E, mask = cv2.findEssentialMat(pts1, pts2, mtx, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, mtx)
        prev_R, prev_t = poses[-1]
        poses.append((prev_R @ R, prev_t + prev_R @ t))
        """
    poses = [(np.eye(3), np.zeros((3, 1)))]
    
    for i in range(len(new_images)-1):
        pts1, pts2 = [], []
        for coord_list in common_coordinates:
            pts1.append(coord_list[0]) 
            pts2.append(coord_list[i+1])

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
 

        E, mask = cv2.findEssentialMat(pts1, pts2, mtx, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, mtx)
        prev_R, prev_t = poses[-1]
        poses.append((prev_R @ R, prev_t + prev_R @ t))

    return poses, keypoints_list, common_coordinates



##############################################################################################
## AUXILIAR FUNCTIONS
##############################################################################################
def load_data_3d_points(path="data/bundle_adjustment_results.npz"):
    data = np.load(path, allow_pickle=True)
    cameras = data["cameras"].item() 
    X_w_opt= data["points_3D"]   
    puntos_0 = data['puntos_0']
    puntos_1 = data['puntos_1']
    puntos_2 = data['puntos_2']
    puntos_3 = data['puntos_3']

    T_wc1 = cameras["C1"]  
    T_wc2 = cameras["C2"] 
    T_wc3 = cameras["C3"]
    T_wc4 = cameras["C4"]
    T_list = [T_wc1, T_wc2, T_wc3, T_wc4]
    Point_list = [puntos_0, puntos_1, puntos_2, puntos_3]

    print("Puntos 3D cargados:", X_w_opt.shape)
    print("Puntos 2d cargados:", puntos_0.shape, puntos_1.shape, puntos_2.shape, puntos_3.shape)
    return T_list, X_w_opt, Point_list


def load_data_old_points(path="data/bundle_adjustment_results_old.npz"):
    data = np.load(path, allow_pickle=True)
    puntos_0 = data['puntos_0']
    puntos_1 = data['puntos_1']
    puntos_2 = data['puntos_2']
    puntos_3 = data['puntos_3']
    puntos_4 = data['puntos_4']
    print("Puntos 2d cargados match con old photo:", puntos_0.shape, puntos_1.shape, puntos_2.shape, puntos_3.shape, puntos_4.shape)
    Point_list = [puntos_0, puntos_1, puntos_2, puntos_3, puntos_4]

    return Point_list



def aprox_K(img_width=680, img_height=341):
    """
    Aproxima la matriz intrínseca de la cámara K a partir de la resolución de la imagen.
    """
    # Parámetros intrínsecos de la cámara
    focal_length = 1.0  # Focal length in pixels
    principal_point = (img_width / 2, img_height / 2)  # Principal point (u0, v0)

    # Matriz intrínseca de la cámara
    K = np.array([
        [focal_length, 0, principal_point[0]],
        [0, focal_length, principal_point[1]],
        [0, 0, 1]
    ])

    return K