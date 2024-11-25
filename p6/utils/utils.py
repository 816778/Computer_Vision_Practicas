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
from scipy.ndimage import gaussian_filter, map_coordinates

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



def normalized_cross_correlation_2(patch: np.array, search_area: np.array) -> np.array:
    """
    Estimate normalized cross correlation values for a patch in a searching area.
    """
    # Complete the function
    i0 = patch
    # ....
    result = np.zeros(search_area.shape, dtype=np.float)
    margin_y = int(patch.shape[0]/2)
    margin_x = int(patch.shape[1]/2)

    for i in range(margin_y, search_area.shape[0] - margin_y):
        for j in range(margin_x, search_area.shape[1] - margin_x):
            i1 = search_area[i-margin_x:i + margin_x + 1, j-margin_y:j + margin_y + 1]
            # Implement the correlation
            # ...
            # result[i, j] = ...
    return result



import numpy as np

def normalized_cross_correlation(patch: np.array, search_area: np.array) -> np.array:
    """
    Estimate normalized cross-correlation (NCC) values for a patch in a search area.

    Parameters:
        patch (np.array): The patch (template) to be matched, size (h, w).
        search_area (np.array): The area in which the patch is searched, larger than the patch.

    Returns:
        np.array: NCC values for the search area, with dimensions adjusted to account for valid NCC computation.
    """
    # Patch size
    patch_h, patch_w = patch.shape

    # Margins for the patch (half-size of patch dimensions)
    margin_y = patch_h // 2
    margin_x = patch_w // 2

    ncc_result = np.zeros(search_area.shape, dtype=np.float)
    patch_mean = np.mean(patch)
    patch_norm = np.sqrt(np.sum((patch - patch_mean) ** 2))

    # Iterate over all valid positions in the search area
    for i in range(margin_y, search_area.shape[0] - margin_y):
        for j in range(margin_x, search_area.shape[1] - margin_x):

            sub_area = search_area[i - margin_y:i + margin_y + 1, j - margin_x:j + margin_x + 1]
            sub_area_mean = np.mean(sub_area)
            sub_area_norm = np.sqrt(np.sum((sub_area - sub_area_mean) ** 2))
            numerator = np.sum((patch - patch_mean) * (sub_area - sub_area_mean))
            if patch_norm > 0 and sub_area_norm > 0:
                ncc_result[i, j] = numerator / (patch_norm * sub_area_norm)
            else:
                ncc_result[i, j] = 0.0  # Set to 0 if normalization factors are invalid

    return ncc_result[margin_y:-margin_y, margin_x:-margin_x]



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

    return i_flow, j_flow


def lucas_kanade_refinement(img1, img2, points, initial_flows, patch_half_size=5, epsilon=1e-2, max_iterations=10):
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
    Ix = gaussian_filter(img1, sigma=1, order=[0, 1])  # Gradiente en x
    Iy = gaussian_filter(img1, sigma=1, order=[1, 0])  # Gradiente en y

    refined_flows = np.zeros_like(initial_flows)

    for idx, (x, y) in enumerate(points):
        u, v = initial_flows[idx]

        # Extraer parche centrado en el punto en img1
        x_start, x_end = int(x - patch_half_size), int(x + patch_half_size + 1)
        y_start, y_end = int(y - patch_half_size), int(y + patch_half_size + 1)

        Ix_patch = Ix[y_start:y_end, x_start:x_end].flatten()
        Iy_patch = Iy[y_start:y_end, x_start:x_end].flatten()
        I1_patch = img1[y_start:y_end, x_start:x_end].flatten()

        # Matriz A
        A = np.array([
            [np.sum(Ix_patch * Ix_patch), np.sum(Ix_patch * Iy_patch)],
            [np.sum(Ix_patch * Iy_patch), np.sum(Iy_patch * Iy_patch)]
        ])

        if np.linalg.det(A) < 1e-5:
            refined_flows[idx] = [u, v]
            continue  # Pasar al siguiente punto si A no es invertible

        for _ in range(max_iterations):
            x_coords, y_coords = np.meshgrid(
                np.arange(x_start, x_end) + u,
                np.arange(y_start, y_end) + v
            )
            I2_patch = map_coordinates(img2, [y_coords.ravel(), x_coords.ravel()], order=1)

            e = I2_patch - I1_patch

            b = np.array([
                np.sum(Ix_patch * e),
                np.sum(Iy_patch * e)
            ])

            delta_u = np.linalg.solve(A, b)

            u += delta_u[0]
            v += delta_u[1]

            if np.linalg.norm(delta_u) < epsilon:
                break

        refined_flows[idx] = [u, v]

    return refined_flows
