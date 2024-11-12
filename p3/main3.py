#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Line RANSAC fitting
#
# Date: 28 September 2020
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.linalg as scAlg
import utils.utils as utils
import cv2
import utils.plot_utils as plot_utils

IMAGE_PATH = 'images/'

if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)
    path_image_1 = 'images/image1.png'
    path_image_2 = 'images/image2.png'


    P = 0.99  # Probabilidad de éxito deseada
    e = 0.5   # Estimación de la fracción de outliers
    s = 8     # Número mínimo de puntos para una homografía

    num_iterations = int(np.log(1 - P) / np.log(1 - (1 - e) ** s))
    threshold = 4 # En píxeles

    num_iterations = 1000

    img1 = cv2.cvtColor(cv2.imread(IMAGE_PATH + 'image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(IMAGE_PATH + 'image2.png'), cv2.COLOR_BGR2RGB)

    print("Número de iteraciones:", num_iterations)
    print("Umbral de error:", threshold)

    matched_points_superglue, _, _ = utils.do_matches()
    matched_points_nndr_sift, _, _ = utils.do_matches(option=1)

    matched_points_all = [
        (matched_points_nndr_sift, "NNDR SIFT Epipolar Lines"),
        (matched_points_superglue, "SuperGlue Epipolar Lines")
    ]

    for matched_points, match_title in matched_points_all:
        H, _ = utils.ransac_homography(matched_points, num_iterations, threshold)
        F, inliers = utils.ransac_fundamental_matrix(matched_points, num_iterations, threshold)
        print(f"Matriz fundamental F found with {len(inliers)} inliers")

        plot_utils.createPlot(IMAGE_PATH + 'image1.png')

        x_coords = [368, 101, 200, 500, 363]
        y_coords = [390, 452, 247, 370, 112]

        assert len(x_coords) == len(y_coords), "Las longitudes de x_coords y y_coords no coinciden"
        labels = [str(i+1) for i in range(len(x_coords))]
        plot_utils.show_points_on_image(x_coords, y_coords, labels, block=False)

        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        x1 = np.vstack((x_coords, y_coords))
        plot_utils.plot_epipolar_lines(F, H, x1, img2, match_title)






