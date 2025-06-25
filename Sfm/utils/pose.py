import numpy as np
import cv2

import utils.geometry as geometry
import utils.plot_utils as plot_utils


def estimate_baseline_pose_from_superglue(data, K, params, draw_matches_bool=False, idx_file=0, fund_method=cv2.FM_RANSAC, outlier_thres=1.0, fund_prob=0.999):
    keypoints0 = data['keypoints0']  # Nx2
    keypoints1 = data['keypoints1']  # Mx2
    matches = data['matches']        # length N0
    confidence = data.get('match_confidence', np.ones_like(matches, dtype=np.float32))

    matched_idx0 = np.where(matches >= 0)[0]
    matched_idx1 = matches[matched_idx0]

    pts0 = keypoints0[matched_idx0]
    pts1 = keypoints1[matched_idx1]
    conf = confidence[matched_idx0]

    if draw_matches_bool:
        print(f"Matches encontrados: {len(pts0)}")
        print(f'Path images: {params["path_images"][idx_file]}')
        resize_dim = tuple(params['resize_dim']) if params['resize_dim'] else None
        plot_utils.draw_matches(pts0, pts1, params['path_images'][idx_file][0], params['path_images'][idx_file][1], resize_dim=resize_dim)

    if len(pts0) < 8:
        print("No hay suficientes matches válidos.")
        return None, None, None

    F, mask = cv2.findFundamentalMat(pts0, pts1, method=fund_method, ransacReprojThreshold=outlier_thres, confidence=fund_prob)
    if F is None:
        print("Falló la estimación de la matriz fundamental.")
        return None, None, None

    mask = mask.ravel().astype(bool)

    # matriz esencial
    E = K.T @ F @ K

    # rotación y traslación
    retval, R21, t21, _ = cv2.recoverPose(E, pts0[mask], pts1[mask], K)
    if retval < 8:
        print(f"recoverPose devolvió pocos inliers: {retval}")
        return None, None, None

    T_c2c1 = geometry.ensamble_T(R21, t21.ravel())
    T_c2c1 = np.linalg.inv(T_c2c1) 

    R, t = R21, t21
    return R, t, T_c2c1, {
        "F": F,
        "E": E,
        "pts0_inliers": pts0[mask],
        "pts1_inliers": pts1[mask],
        "conf_inliers": conf[mask],
        "match_indices": (matched_idx0[mask], matched_idx1[mask])
    }
