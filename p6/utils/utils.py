import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import random
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as scAlg
import scipy as sc
import scipy.optimize as scOptim
import scipy.io as sio
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm, logm
import time

import numpy as np
from scipy.ndimage import map_coordinates

def read_image(filename: str, ):
    """
    Read image using opencv converting from BGR to RGB
    :param filename: name of the image
    :return: np matrix with the image
    """
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img



def read_flo_file(filename, verbose=False):
    """
    Read from .flo optical flow file (Middlebury format)
    :param flow_file: name of the flow file
    :return: optical flow data in matrix

    adapted from https://github.com/liruoteng/OpticalFlowToolkit/

    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        raise TypeError('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        if verbose:
            print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=int(2 * w * h))
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d


def generate_wheel(size):
    """
     Generate wheel optical flow for visualizing colors
     :param size: size of the image
     :return: flow: optical flow for visualizing colors
     """
    rMax = size / 2
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    u = x - size / 2
    v = y - size / 2
    r = np.sqrt(u ** 2 + v ** 2)
    u[r > rMax] = 0
    v[r > rMax] = 0
    flow = np.dstack((u, v))

    return flow



def normalized_cross_correlation(patch: np.array, search_area: np.array) -> np.array:
    """
    Estimate normalized cross-correlation (NCC) values for a patch in a search area.

    Parameters:
        patch (np.array): Template patch, size (h, w).
        search_area (np.array): Area to search, larger than the patch.

    Returns:
        np.array: NCC values for the search area, adjusted for valid computation.
    """
    i0 = patch
    margin_y = int(patch.shape[0]/2)
    margin_x = int(patch.shape[1]/2)

    # Pre-compute mean and normalization factor for the patch
    i0_mean = np.mean(i0) 
    i0_diff = i0 - i0_mean
    i0_norm = np.sqrt(np.sum(i0_diff ** 2))

    # Resultant NCC matrix (adjusted to exclude invalid edges)
    result = np.zeros(search_area.shape, dtype=np.float)

    # Iterate over all valid positions in the search area
    for i in range(margin_y, search_area.shape[0] - margin_y):
        for j in range(margin_x, search_area.shape[1] - margin_x):
            
            i1 = search_area[i-margin_x:i + margin_x + 1, j-margin_y:j + margin_y + 1]

            # Compute mean and normalization factor for the sub-area
            i1_mean = np.mean(i1)
            i1_diff = i1 - i1_mean
            i1_norm = np.sqrt(np.sum(i1_diff ** 2))

            # Compute numerator of the NCC formula
            numerator = np.sum(i0_diff * i1_diff)

            # Compute NCC value (only if both norms are non-zero)
            if i0_norm != 0 and i1_norm != 0:
                result[i, j] = numerator / (i0_norm * i1_norm)
            else:
                result[i, j] = 0.0  # Set to 0 if normalization factors are invalid

    # Return valid part of the result (excluding margins)
    return result



def seed_estimation_NCC_single_point(img1_gray, img2_gray, i_img, j_img, patch_half_size: int = 5, searching_area_size: int = 100):

    # Attention!! we are not checking the padding
    patch = img1_gray[i_img - patch_half_size:i_img + patch_half_size + 1, j_img - patch_half_size:j_img + patch_half_size + 1]

    i_ini_sa = i_img - int(searching_area_size / 2)
    i_end_sa = i_img + int(searching_area_size / 2) + 1
    j_ini_sa = j_img - int(searching_area_size / 2)
    j_end_sa = j_img + int(searching_area_size / 2) + 1

    search_area = img2_gray[i_ini_sa:i_end_sa, j_ini_sa:j_end_sa]
    result = normalized_cross_correlation(patch, search_area)

    iMax, jMax = np.where(result == np.amax(result))

    i_flow = i_ini_sa + iMax[0] - i_img
    j_flow = j_ini_sa + jMax[0] - j_img

    return i_flow, j_flow # filas, columnas

def numerical_gradient(img_int: np.array, point: np.array)->np.array:
    """
    https://es.wikipedia.org/wiki/Interpolaci%C3%B3n_bilineal
    :param img:image to interpolate
    :param point: [[y0,x0],[y1,x1], ... [yn,xn]]
    :return: Ix_y = [[Ix_0,Iy_0],[Ix_1,Iy_1], ... [Ix_n,Iy_n]]
    """

    a = np.zeros((point.shape[0], 2), dtype= np.float)
    filter = np.array([-1, 0, 1], dtype=np.float)
    point_int = point.astype(np.int)
    img = img_int.astype(np.float)

    for i in range(0,point.shape[0]):
        py = img[point_int[i,0]-1:point_int[i,0]+2,point_int[i,1]].astype(np.float)
        px = img[point_int[i,0],point_int[i,1]-1:point_int[i,1]+2].astype(np.float)
        a[i, 0] = 1/2*np.dot(filter,px)
        a[i, 1] = 1/2*np.dot(filter,py)

    return a

def int_bilineal(img: np.array, point: np.array)->np.array:
    """
    https://es.wikipedia.org/wiki/Interpolaci%C3%B3n_bilineal
    Vq = scipy.ndimage.map_coordinates(img.astype(np.float), [point[:, 0].ravel(), point[:, 1].ravel()], order=1, mode='nearest').reshape((point.shape[0],))

    :param img:image to interpolate
    :param point: point subpixel
    point = [[y0,x0],[y1,x1], ... [yn,xn]]
    :return: [gray0,gray1, .... grayn]
    """
    h, w = img.shape
    
    A = np.zeros((point.shape[0], 2, 2), dtype= np.float)
    point_lu = point.astype(np.int)
    point_ru = np.copy(point_lu)
    point_ru[:,1] = point_ru[:,1] + 1
    point_ld = np.copy(point_lu)
    point_ld[:, 0] = point_ld[:, 0] + 1
    point_rd = np.copy(point_lu)
    point_rd[:, 0] = point_rd[:, 0] + 1
    point_rd[:, 1] = point_rd[:, 1] + 1

    A[:, 0, 0] = img[point_lu[:,0],point_lu[:,1]]
    A[:, 0, 1] = img[point_ru[:,0],point_ru[:,1]]
    A[:, 1, 0] = img[point_ld[:,0],point_ld[:,1]]
    A[:, 1, 1] = img[point_rd[:,0],point_rd[:,1]]
    l_u = np.zeros((point.shape[0],1,2),dtype= np.float)
    l_u[:, 0, 0] = -((point[:,0]-point_lu[:,0])-1)
    l_u[:, 0, 1] = point[:,0]-point_lu[:,0]

    r_u = np.zeros((point.shape[0],2,1),dtype= np.float)
    r_u[:, 0, 0] = -((point[:,1]-point_lu[:,1])-1)
    r_u[:, 1, 0] = point[:, 1]-point_lu[:,1]
    grays = l_u @ A @ r_u

    return grays.reshape((point.shape[0],))



def lucas_kanade_refinement(img1, img2, points, initial_flows, patch_half_size=5, epsilon=1e-2, max_iterations=100, det_threshold=1e-5):
    """
    Refinar el flujo óptico inicial utilizando el método Lucas-Kanade.

    Parámetros:
        img1 (np.array): Primera imagen en escala de grises.
        img2 (np.array): Segunda imagen en escala de grises.
        points (np.array): Puntos seleccionados (x, y) en la primera imagen.
        initial_flows (np.array): Flujos iniciales (dx, dy) calculados con NCC.
        patch_half_size (int): Radio del parche centrado en el punto.
        epsilon (float): Umbral para detener las iteraciones.
        max_iterations (int): Máximo número de iteraciones por punto.

    Retorno:
        refined_flows (np.array): Flujos refinados (dx, dy) después del refinamiento.
    """
    # Gradientes de la primera imagen
    Ix, Iy = np.gradient(img1)

    refined_flows = np.zeros_like(initial_flows)

    for idx, (x, y) in enumerate(points):
        u = initial_flows[idx].copy()

        # Extraer parche centrado en el punto en img1
        x_start, x_end = int(x - patch_half_size), int(x + patch_half_size + 1)
        y_start, y_end = int(y - patch_half_size), int(y + patch_half_size + 1)
        
        Ix_patch = Ix[y_start:y_end, x_start:x_end].flatten()
        Iy_patch = Iy[y_start:y_end, x_start:x_end].flatten()
        I0_patch = img1[y_start:y_end, x_start:x_end].flatten()

        # Matriz A
        Ix2 = np.sum(Ix_patch ** 2)
        Iy2 = np.sum(Iy_patch ** 2)
        IxIy = np.sum(Ix_patch * Iy_patch)
        
        A = np.array([
            [Ix2, IxIy],
            [IxIy, Iy2]
        ])

        if np.linalg.det(A) < det_threshold:
            refined_flows[idx] = u
            continue  # Pasar al siguiente punto si A no es invertible


        for i in range(max_iterations):
            
            # Generar coordenadas desplazadas para el parche
            x_coords, y_coords = np.meshgrid(
                np.arange(x_start, x_end) + u[0],
                np.arange(y_start, y_end) + u[1]
            )

            points_to_interpolate = np.vstack((y_coords.ravel(), x_coords.ravel())).T
            
            I1_patch = int_bilineal(img2, points_to_interpolate)
            It = I1_patch - I0_patch

            b = -np.array([
                np.sum(Iy_patch * It),
                np.sum(Ix_patch * It)
            ]).T

            inv_A = np.linalg.inv(A)
            delta_u = inv_A @ b
            # delta_u = np.linalg.solve(A, b)
            u += delta_u

            if i == max_iterations - 1:
                print(f"No converge en punto ({x}, {y}), usando flujo inicial.")
                u = initial_flows[idx]
                break

            # print(f"u: {u}, delta_u: {delta_u}, norm: {np.linalg.norm(delta_u)}")

            if np.linalg.norm(delta_u) < epsilon:
                break

        refined_flows[idx] = u  

    return refined_flows



def lucas_kanade_refinement_region(img1, img2, points, initial_flows, region, patch_half_size=5, epsilon=1e-2, max_iterations=100, det_threshold=1e-5):
    x_min, y_min, x_max, y_max = region

    # Recortar imágenes a la región de interés
    img1 = img1[y_min:y_max, x_min:x_max]
    img2 = img2[y_min:y_max, x_min:x_max]

    Ix, Iy = np.gradient(img1)

    refined_flows = np.zeros_like(initial_flows)

    for idx, (x, y) in enumerate(points):
        u = initial_flows[idx]

        # Extraer parche centrado en el punto en img1
        x_start, x_end = int(x - patch_half_size), int(x + patch_half_size + 1)
        y_start, y_end = int(y - patch_half_size), int(y + patch_half_size + 1)

        if x_start < 0 or y_start < 0 or x_end > img1.shape[1] or y_end > img1.shape[0]:
            print(f"Image shape: {img1.shape}")
            print(f"x_start, x_end: {x_start}, {x_end}")
            print(f"y_start, y_end: {y_start}, {y_end}")
            print(f"Parche fuera de límites para punto ({x}, {y})")
            continue

        Ix_patch = Ix[y_start:y_end, x_start:x_end].flatten()
        Iy_patch = Iy[y_start:y_end, x_start:x_end].flatten()
        I0_patch = img1[y_start:y_end, x_start:x_end].flatten()

        # Matriz A
        Ix2 = np.sum(Ix_patch ** 2)
        Iy2 = np.sum(Iy_patch ** 2)
        IxIy = np.sum(Ix_patch * Iy_patch)
        
        A = np.array([
            [Ix2, IxIy],
            [IxIy, Iy2]
        ])

        if np.linalg.det(A) < det_threshold:
            refined_flows[idx] = u
            print(f"Matriz A no invertible para punto ({x}, {y})")
            continue  


        for i in range(max_iterations):
            
            # Generar coordenadas desplazadas para el parche
            x_coords, y_coords = np.meshgrid(
                np.arange(x_start, x_end) + u[0],
                np.arange(y_start, y_end) + u[1]
            )

            points_to_interpolate = np.vstack((y_coords.ravel(), x_coords.ravel())).T
            
            I1_patch = int_bilineal(img2, points_to_interpolate)
            It = I1_patch - I0_patch

            b = -np.array([
                np.sum(Iy_patch * It),
                np.sum(Ix_patch * It)
            ]).T

            inv_A = np.linalg.inv(A)
            delta_u = inv_A @ b
            u += delta_u

            # print(f"delta_u: {delta_u}")
            # print(f"u: {np.linalg.norm(delta_u)}")

            if np.linalg.norm(delta_u) < epsilon:
                break

        refined_flows[idx] = u

    return refined_flows



def compute_seed_flow(img1_gray, img2_gray, points_selected, template_size_half, searching_area_size):
    """
    Calcula el flujo inicial usando NCC para los puntos seleccionados.
    """
    seed_optical_flow_sparse = np.zeros(points_selected.shape)
    for k in range(points_selected.shape[0]):
        i_flow, j_flow = seed_estimation_NCC_single_point(
            img1_gray, img2_gray,
            points_selected[k, 1], points_selected[k, 0],
            template_size_half, searching_area_size
        )
        seed_optical_flow_sparse[k, :] = np.hstack((j_flow, i_flow))

    return seed_optical_flow_sparse


def compute_sparse_flow_errors(seed_optical_flow_sparse, flow_gt):
    """
    Calcula el error del flujo óptico inicial basado en NCC.
    """
    error_sparse_ncc = seed_optical_flow_sparse - flow_gt
    error_sparse_norm_ncc = np.sqrt(np.sum(error_sparse_ncc ** 2, axis=1))

    return error_sparse_ncc, error_sparse_norm_ncc


def convert_to_dense_flow(points_selected, sparse_flows, image_shape):
    """
    Convierte un flujo óptico disperso a una representación densa.

    Parameters:
        points_selected (np.array): Coordenadas de los puntos seleccionados (n, 2).
        sparse_flows (np.array): Flujos calculados en puntos seleccionados (n, 2).
        image_shape (tuple): Forma de la imagen (alto, ancho).

    Returns:
        np.array: Flujo denso (alto, ancho, 2).
    """
    dense_flow = np.zeros((image_shape[0], image_shape[1], 2), dtype=np.float32)
    for idx, (x, y) in enumerate(points_selected):
        dense_flow[y, x] = sparse_flows[idx]
    return dense_flow


def compute_dense_flow_error(flow_gt, flow_est, unknownFlowThresh=1e9):
    """
    Calcula el error entre el flujo ground truth y el flujo estimado.

    Parameters:
        flow_gt_dense (np.array): Flujo ground truth (alto, ancho, 2).
        flow_est_dense (np.array): Flujo estimado (alto, ancho, 2).

    Returns:
        np.array: Norma del error por píxel (alto, ancho).
    """
    binUnknownFlow = flow_gt > unknownFlowThresh
    flow_error = flow_est - flow_gt
    flow_error[binUnknownFlow] = 0
    error_norm = np.sqrt(np.sum(flow_error ** 2, axis=-1))
    return error_norm



def select_new_points(img1, region, num_points=10, border_margin=20, use_random_points=True):
    x_min, y_min, x_max, y_max = region
    roi = img1[y_min:y_max, x_min:x_max]

    if use_random_points:
        corners = cv.goodFeaturesToTrack(roi, maxCorners=num_points, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            corners = np.int0(corners)

            # Ajustar las coordenadas relativas a la región al marco global
            points = []
            global_points = []
            for corner in corners:
                x, y = corner.ravel()
                x_global, y_global = x + x_min, y + y_min 
                
                # Filtrar puntos cerca de los bordes de la región
                if (x > border_margin and x < (roi.shape[1] - border_margin) and
                    y > border_margin and y < (roi.shape[0] - border_margin)):
                    global_points.append([x_global, y_global])
                    points.append([x, y])

            points = np.array(points[:num_points])
            global_points = np.array(global_points[:num_points])
    else:
        height, width = roi.shape
        points = []
        global_points = []

        for y in range(border_margin, height - border_margin):
            for x in range(border_margin, width - border_margin):
                points.append([x, y])
                global_points.append([x + x_min, y + y_min])

        points = np.array(points)
        global_points = np.array(global_points)

    return points, global_points