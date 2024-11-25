#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 4
#
# Title: Optical Flow
#
# Date: 22 November 2020
#
#####################################################################################
#
# Authors: Jose Lamarca, Jesus Bermudez, Richard Elvira, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import utils.utils as utils
import utils.plot_utils as plot_utils

IMAGES_FOLDER = "images/"
DATA_FOLDER = "data/"



if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    img1 = utils.read_image(IMAGES_FOLDER + "frame10.png")
    img2 = utils.read_image(IMAGES_FOLDER + "frame11.png")

    img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    # List of sparse points
    points_selected = np.loadtxt(DATA_FOLDER + 'points_selected.txt')
    points_selected = points_selected.astype(int)

    template_size_half = 5
    searching_area_size: int = 15

    seed_optical_flow_sparse = np.zeros((points_selected.shape))
    for k in range(0,points_selected.shape[0]):
        i_flow, j_flow = utils.seed_estimation_NCC_single_point(img1_gray, img2_gray, points_selected[k,1], points_selected[k,0], template_size_half, searching_area_size)
        seed_optical_flow_sparse[k,:] = np.hstack((j_flow,i_flow))

    print(seed_optical_flow_sparse)

   


    # Plot results for sparse optical flow
    unknownFlowThresh = 1e9
    flow_12 = utils.read_flo_file(DATA_FOLDER + "flow10.flo", verbose=True)
    binUnknownFlow = flow_12 > unknownFlowThresh

    flow_gt = flow_12[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)].astype(float)

    error_sparse_ncc = seed_optical_flow_sparse - flow_gt
    error_sparse_norm_ncc = np.sqrt(np.sum(error_sparse_ncc ** 2, axis=1))

    flow_gt_sparse_lk = flow_12[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)]
    flow_gt_sparse_ncc = flow_12[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)]

    plot_utils.visualize_sparse_flow(img1, points_selected, seed_optical_flow_sparse, error_sparse_ncc, error_sparse_norm_ncc, title='NCC')

    refined_flows = utils.lucas_kanade_refinement(
        img1_gray, img2_gray,
        points_selected,
        seed_optical_flow_sparse,
        patch_half_size=template_size_half,
        epsilon=1e-3,
        max_iterations=1000
    )

    print("Flujos refinados:\n", refined_flows)

    error_sparse_lk = refined_flows - flow_gt
    error_sparse_norm_lk = np.sqrt(np.sum(error_sparse_lk ** 2, axis=1))

    plot_utils.visualize_sparse_flow(img1, points_selected, refined_flows, error_sparse_lk, error_sparse_norm_lk)

