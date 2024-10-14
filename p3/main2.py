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

def drawLine(l,strFormat,lWidth):
    """
    Draw a line
    -input:
      l: image line in homogenous coordinates
      strFormat: line format
      lWidth: line width
    -output: None
    """
    # p_l_y is the intersection of the line with the axis Y (x=0)
    p_l_y = np.array([0, -l[2] / l[1]])
    # p_l_x is the intersection point of the line with the axis X (y=0)
    p_l_x = np.array([-l[2] / l[0], 0])
    # Draw the line segment p_l_x to  p_l_y
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], strFormat, linewidth=lWidth)


if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)
    path_image_1 = 'images/image1.png'
    path_image_2 = 'images/image2.png'


    P = 0.99  # Probabilidad de éxito deseada
    e = 0.5   # Estimación de la fracción de outliers
    s = 4     # Número mínimo de puntos para una homografía

    num_iterations = int(np.log(1 - P) / np.log(1 - (1 - e) ** s))
    threshold = 4 # En píxeles

    print("Número de iteraciones:", num_iterations)
    print("Umbral de error:", threshold)

    distRatio = 0.8
    dMatchesList, keypoints1, keypoints2 = utils.visualize_matches(path_image_1, path_image_2, distRatio)

    print("Total de keypoints en la primera imagen:", len(keypoints1))
    print("Total de keypoints en la segunda imagen:", len(keypoints2))

    # Convierte los emparejamientos a coordenadas (x, y)
    srcPts = np.float32([keypoints1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
    dstPts = np.float32([keypoints2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
    x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))

    matched_points = np.hstack((x1, x2))
    matched_points = np.hstack((srcPts, dstPts))

    best_H, inliers_count = utils.ransac_homography(matched_points, num_iterations, threshold)