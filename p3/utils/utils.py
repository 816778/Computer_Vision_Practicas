import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

##########################################################################
# 
# FUNCIONES BASE
#
##########################################################################

def indexMatrixToMatchesList(matchesList):
    """
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return dMatchesList

def matchesListToIndexMatrix(dMatchesList):
    """
     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([int(dMatchesList[k].queryIdx), int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList


def matchWith2NDRR(desc1, desc2, distRatio, minDist=100):
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

        if d1 < d2 * distRatio and d1 < minDist:
            matches.append([kDesc1, indexSort[0], d1])
    
    return matches


def matchWith2NDRR_0(desc1, desc2, distRatio, minDist):
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
        indexSort = np.argsort(dist)
        if (dist[indexSort[0]] < minDist):
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])
    return matches


def visualize_matches_with_threshold(path_image_1, path_image_2, minDist, distRatio):
    image1 = cv2.imread(path_image_1)
    image2 = cv2.imread(path_image_2)

    # Feature extraction
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    matchesList = matchWith2NDRR_0(descriptors1, descriptors2, distRatio=distRatio, minDist=minDist)
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)
    
    # Dibujar los primeros 100 emparejamientos
    img_matched = cv2.drawMatches(
        image1, keypoints1, image2, keypoints2, dMatchesList[:100], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Mostrar el resultado
    plt.imshow(img_matched, cmap='gray', vmin=0, vmax=255)
    plt.title(f"Emparejamientos con minDist = {minDist}")
    plt.draw()
    plt.waitforbuttonpress()
    
    return dMatchesList, keypoints1, keypoints2



def visualize_matches(path_image_1, path_image_2, distRatio):
    image1 = cv2.imread(path_image_1)
    image2 = cv2.imread(path_image_2)

    # Feature extraction
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    matchesList = matchWith2NDRR(descriptors1, descriptors2, distRatio=distRatio)
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)
    
    # Dibujar los primeros 100 emparejamientos
    img_matched = cv2.drawMatches(
        image1, keypoints1, image2, keypoints2, dMatchesList[:100], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Mostrar el resultado
    plt.imshow(img_matched, cmap='gray', vmin=0, vmax=255)
    plt.title(f"Emparejamientos con distRatio = {distRatio}")
    plt.draw()
    plt.waitforbuttonpress()
    
    return dMatchesList, keypoints1, keypoints2  # Retornar las coincidencias para análisis posterior


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


def display_matches(matches, inliers, src_points, dst_points, H, title="Matches"):
    # Visualizar coincidencias usando drawMatches
    img1 = cv2.imread('images/image1.png')
    img2 = cv2.imread('images/image2.png')
    
    inlier_points1 = matches[inliers, :2]  # Forma (n_inliers, 2)
    inlier_points2 = matches[inliers, 2:4]  # Forma (n_inliers, 2)

    # Crear KeyPoints para los puntos inliers
    keypoints1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in inlier_points1]
    keypoints2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in inlier_points2]

    # Crear matches de acuerdo al índice secuencial
    matches_inliers = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(keypoints1))]

    # Usar drawMatches con las nuevas listas de KeyPoints y Matches
    img_matches = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, matches_inliers, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    plt.figure(figsize=(10, 5))
    plt.imshow(img_matches)
    plt.title(title)
    plt.show()


def ransac_homography(matches, num_iterations, threshold):
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

        # Mostrar los 4 puntos de la hipótesis actual
        if i % 10 == 0:  # Mostrar cada 10 iteraciones
            display_matches(matches, inliers, src_points, dst_points, H, title=f"Iteration {i}")

        # Verificar si es la mejor hipótesis
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_homography = H
            best_inliers = inliers

    # Visualizar la mejor hipótesis
    display_matches(matches, best_inliers, matches[best_inliers, :2], matches[best_inliers, 2:4], best_homography, title="Best Hypothesis")
    return best_homography, best_inliers_count


