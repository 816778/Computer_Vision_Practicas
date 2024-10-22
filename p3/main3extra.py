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

    matched_points, kp1, kp2 = utils.do_matches(option=1)
    H, _ = utils.ransac_homography(matched_points, num_iterations, threshold)
    F, inliers = utils.ransac_fundamental_matrix(matched_points, num_iterations, threshold)
    
    # Add extra matches
    new_matches = utils.matchEpipolar(kp1, kp2, F, 10)
    print(new_matches.shape)
    #matched_points = np.vstack((matched_points, new_matches))

    plot_utils.createPlot(IMAGE_PATH + 'image1.png')
    
    # Plot matches in both images
    plot_utils.show_points_on_image(matched_points[:, 0], matched_points[:, 1], labels=None, block=False)

    plot_utils.createPlot(IMAGE_PATH + 'image2.png')
    
    # Dibujar los puntos en la imagen
    plt.scatter(matched_points[:, 2], matched_points[:, 3], color='yellow', s=100, marker='x', label='Puntos')
    plt.scatter(new_matches[:, 2], new_matches[:, 3], color='red', s=100, marker='x', label='Puntos')

    # Mostrar el resultado
    plt.title('Puntos proyectados en la imagen')
    plt.axis('off')  # Ocultar los ejes
    # Plot and continue
    plt.show(block=True)    







