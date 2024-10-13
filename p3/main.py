import utils.utils as utils
import numpy as np
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    # Images path
    timestamp1 = '1403715282262142976'
    timestamp2 = '1403715413262142976'

    path_image_1 = 'images/image1.png'
    path_image_2 = 'images/image2.png'

    distRatio = 0.8
    for minDist in [50, 100, 200, 500, 1000]:
        break
        dMatchesList, keypoints1, keypoints2 = utils.visualize_matches_with_threshold(path_image_1, path_image_2, minDist, distRatio)

    for distRatio in [1, 0.8, 0.5, 0.25]:
        break
        dMatchesList, keypoints1, keypoints2 = utils.visualize_matches(path_image_1, path_image_2, distRatio)
    exit()
    # Conversion from DMatches to Python list
    matchesList = utils.matchesListToIndexMatrix(dMatchesList)

    # Matched points in numpy from list of DMatches
    srcPts = np.float32([keypoints1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
    dstPts = np.float32([keypoints2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)

    # Matched points in homogeneous coordinates
    x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))
