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
    num_iterations = 1000

    print("Número de iteraciones:", num_iterations)
    print("Umbral de error:", threshold)

    matched_points_superglue, _, _ = utils.do_matches()
    matched_points_nndr_sift, _, _ = utils.do_matches(option=1)

    matched_points_all = [
        (matched_points_nndr_sift, "NNDR SIFT Matches Homography"),
        (matched_points_superglue, "SuperGlue Matches Homography")
    ]
    
    for matched_points, match_title in matched_points_all:
      # Compute homography using RANSAC
      best_H, inliers_count = utils.ransac_homography(matched_points, num_iterations, threshold)

      # Call the function with the title and matched points
      utils.print_projected_with_homography(best_H, path_image_1, path_image_2, matched_points, title=match_title)
      utils.print_projected_with_homography_2(best_H, path_image_1, path_image_2, matched_points)
      print("#############################################################################")

