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

    DO_PLOT = True

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
        epsilon=1e-6,
        max_iterations=1000
    )
    print("flow_gt:\n", flow_gt)
    print("Flujos refinados:\n", refined_flows)

    error_sparse_lk, error_sparse_norm_lk = utils.compute_sparse_flow_errors(refined_flows, flow_gt)
    print("error_sparse_norm_lk:", error_sparse_norm_lk)
    if DO_PLOT:
        # plot_utils.visualize_sparse_flow_2(img1, points_selected,seed_optical_flow_sparse, error_sparse_ncc, error_sparse_norm_ncc, flow_est_sparse_norm_ncc, title="NCC")
        # plot_utils.visualize_sparse_flow_2(img1, points_selected,refined_flows, error_sparse_lk, error_sparse_norm_lk, flow_est_sparse_norm, title="Lucas-Kanade")
        # plot_utils.visualize_sparse_flow(img1, points_selected, seed_optical_flow_sparse, error_sparse_ncc, error_sparse_norm_ncc, title="NCC")
        plot_utils.visualize_sparse_flow(img1, points_selected,refined_flows, error_sparse_lk, error_sparse_norm_lk, title="Lucas-Kanade")

    exit()
    ##########################################################################################
    # OPTIONAL
    ##########################################################################################
    print("\n\n##################################################################")
    print("# OPTIONAL")
    print("##################################################################\n")
    region = (x_min, y_min, x_max, y_max) = (50, 140, 200, 210)
    # plot_utils.visualite_points_region(img1, region, points_selected)
    img1_sub = img1[region[1]:region[3], region[0]:region[2]]
    img2_sub = img2[region[1]:region[3], region[0]:region[2]]

    points_selected, global_points = utils.select_new_points(img1_gray, region, use_random_points=True)
    if DO_PLOT:
        plot_utils.visualite_points_region(img1, region, global_points)
        plot_utils.visualite_points_region(img1_sub, region, points_selected, is_global=False)

    seed_optical_flow_sparse = utils.compute_seed_flow(
        img1_gray, img2_gray,
        points_selected,
        template_size_half,
        searching_area_size
    )
    print("Flujo inicial (NCC):\n", seed_optical_flow_sparse)
    # seed_optical_flow_sparse = -1 * np.ones((8, 2))

    refined_flows = utils.lucas_kanade_refinement_region(
        img1=img1_gray,
        img2=img2_gray,
        points=points_selected,
        initial_flows=seed_optical_flow_sparse,
        region=region,
        patch_half_size=5,
        epsilon=1e-2,
        max_iterations=10,
        det_threshold=1e-5
    )
    flow_gt = flow_12[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)].astype(float)
    print("flow_gt:\n", flow_gt)
    print("Flujos refinados:\n", refined_flows)

    # error_sparse_lk, error_sparse_norm_lk = utils.compute_sparse_flow_errors(refined_flows, flow_gt)
    binUnknownFlow = flow_12 > unknownFlowThresh
    flow_est = utils.convert_to_dense_flow(points_selected, refined_flows, img1_gray.shape)

    flow_12_sub = flow_12[region[1]:region[3], region[0]:region[2]]
    flow_est_sub = flow_est[region[1]:region[3], region[0]:region[2]]
    binUnknownFlow_sub = binUnknownFlow[region[1]:region[3], region[0]:region[2]]

    if DO_PLOT:
        plot_utils.visualize_dense_flow(img1_sub, img2_sub, flow_12_sub, flow_est_sub, binUnknownFlow_sub)


    ##########################################################################################
    # OPTIONAL 2
    ##########################################################################################
    tvl1 = cv.optflow.createOptFlow_DualTVL1()
    flow_tvl1 = tvl1.calc(img1_gray, img2_gray, None)
    binUnknownFlow = flow_12 > unknownFlowThresh
    plot_utils.visualize_dense_flow(img1, img2, flow_12, flow_tvl1, binUnknownFlow)
    print("Flujo de referencia (GT):", flow_gt)
    print("Flujo estimado (TV-L1):", flow_tvl1)

    flow_farneback = cv.calcOpticalFlowFarneback(img1_gray, img2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    plot_utils.visualize_dense_flow(img1, img2, flow_12, flow_farneback, binUnknownFlow)
    
