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


def load_images_and_points():
    """
    Carga las imágenes y los puntos seleccionados.
    """
    img1 = utils.read_image(IMAGES_FOLDER + "frame10.png")
    img2 = utils.read_image(IMAGES_FOLDER + "frame11.png")

    img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    points_selected = np.loadtxt(DATA_FOLDER + 'points_selected.txt').astype(int)

    flow_12 = utils.read_flo_file(DATA_FOLDER + "flow10.flo", verbose=True)

    return img1, img2, img1_gray, img2_gray, points_selected, flow_12


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    DO_PLOT = False

    img1, img2, img1_gray, img2_gray, points_selected, flow_12 = load_images_and_points()
    flow_gt = flow_12[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)].astype(float)

    template_size_half = 5
    searching_area_size: int = 15
    unknownFlowThresh = 1e9

    seed_optical_flow_sparse = utils.compute_seed_flow(
        img1_gray, img2_gray,
        points_selected,
        template_size_half,
        searching_area_size
    )
    print("Flujo inicial (NCC):\n", seed_optical_flow_sparse)
    error_sparse_ncc, error_sparse_norm_ncc = utils.compute_sparse_flow_errors(seed_optical_flow_sparse, flow_gt)

   
    refined_flows = utils.lucas_kanade_refinement(
        img1_gray, img2_gray,
        points_selected,
        seed_optical_flow_sparse,
        patch_half_size=template_size_half,
        epsilon=1e-3,
        max_iterations=1000
    )
    print("Flujos refinados:\n", refined_flows)

    error_sparse_lk, error_sparse_norm_lk = utils.compute_sparse_flow_errors(refined_flows, flow_gt)

    if DO_PLOT:
        plot_utils.visualize_sparse_flow(img1, points_selected, seed_optical_flow_sparse, 
                                        error_sparse_ncc, error_sparse_norm_ncc, title="NCC")

        plot_utils.visualize_sparse_flow(img1, points_selected,refined_flows,
            error_sparse_lk, error_sparse_norm_lk, title="Lucas-Kanade")

    ##########################################################################################
    # OPTIONAL
    ##########################################################################################
    flow_gt_dense = utils.convert_to_dense_flow(points_selected, flow_gt, img1_gray.shape)
    flow_refined_dense = utils.convert_to_dense_flow(points_selected, refined_flows, img1_gray.shape)

    # Calcular el error denso
    flow_error_dense = utils.compute_dense_flow_error(flow_gt_dense, flow_refined_dense)

    print("flow_gt_dense shape:", flow_gt_dense.shape)
    print("flow_refined_dense shape:", flow_refined_dense.shape)
    print("flow_error_dense shape:", flow_error_dense.shape)

    plot_utils.visualize_dense_flow(img1_gray, img2_gray, flow_gt_dense, flow_refined_dense, flow_error_dense)