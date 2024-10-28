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
    
    return matches

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
        if (dist[indexSort[0]] < minDist and dist[indexSort[0]] < distRatio * dist[indexSort[1]]):
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
    plt.subplots_adjust(
        top=0.985,     # Border for top
        bottom=0.015,  # Border for bottom
        left=0.028,    # Border for left
        right=0.992,   # Border for right
    )
    plt.draw()
    plt.waitforbuttonpress()
    
    return dMatchesList, keypoints1, keypoints2


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


def print_projected_with_homography_2(H, path_image_1, path_image_2, matches):

    # Project points with homography
    projected_points = H @ np.vstack((matches[:, :2].T, np.ones((1, matches.shape[0]))))
    projected_points = projected_points / projected_points[2, :]
    projected_points = projected_points[:2, :].T

    # Display the projected points over images
    img1 = cv2.imread(path_image_1)
    img2 = cv2.imread(path_image_2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(img1)
    ax[0].plot(matches[:, 0], matches[:, 1], 'ro', ms=3)
    #ax[0].plot(projected_points[:, 0], projected_points[:, 1], 'bo')
    ax[0].set_title("Image 1")

    ax[1].imshow(img2)
    ax[1].plot(matches[:, 2], matches[:, 3], 'ro', ms=3)
    ax[1].plot(projected_points[:, 0], projected_points[:, 1], 'bo', ms=3)
    ax[1].set_title("Image 2")

    plt.subplots_adjust(
        top=0.985,     # Border for top
        bottom=0.015,  # Border for bottom
        left=0.028,    # Border for left
        right=0.992,   # Border for right
    )

    plt.legend(["Matched points", "Projected points"])
    plt.show()

def visualize_matches(path_image_1, path_image_2, distRatio, maxDist, draw=True):
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


    return best_homography, best_inliers_count



def do_matches(option=0, path_image_1='images/image1.png', path_image_2='images/image2.png', draw=False):
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
        dMatchesList, keypoints1, keypoints2 = visualize_matches(path_image_1, path_image_2, distRatio, maxDist, draw=draw)

        print("Total de keypoints en la primera imagen:", len(keypoints1))
        print("Total de keypoints en la segunda imagen:", len(keypoints2))

        # Convierte los emparejamientos a coordenadas (x, y)
        srcPts = np.float32([keypoints1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
        dstPts = np.float32([keypoints2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
        x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
        x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))

        matched_points = np.hstack((x1, x2))
        matched_points = np.hstack((srcPts, dstPts))

    else:
        matched_points = None

    return matched_points, srcPts, dstPts


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