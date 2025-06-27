import numpy as np
import os
import cv2

from utils import projection
from utils import pose

def triangulate_points(P1, P2, x1, x2):
    """
    Triangular puntos 3D a partir de las matrices de proyección P1 y P2 y las correspondencias x1 y x2.
    """
    if x1.shape[0] == 2:
        x1 = np.vstack((x1, np.ones((1, x1.shape[1]))))  
    if x2.shape[0] == 2:
        x2 = np.vstack((x2, np.ones((1, x2.shape[1])))) 

    num_points = x1.shape[1]
    X_homogeneous = np.zeros((4, num_points))

    # Triangular cada par de puntos
    for i in range(num_points):
        # Para cada punto, construir la matriz A para el sistema de ecuaciones
        A = np.array([
            x1[0, i] * P1[2, :] - P1[0, :],  # Ecuación para x1 en la primera cámara
            x1[1, i] * P1[2, :] - P1[1, :],  # Ecuación para y1 en la primera cámara
            x2[0, i] * P2[2, :] - P2[0, :],  # Ecuación para x2 en la segunda cámara
            x2[1, i] * P2[2, :] - P2[1, :]   # Ecuación para y2 en la segunda cámara
        ])

        # Resolver el sistema usando SVD
        _, _, Vt = np.linalg.svd(A)
        X_homogeneous[:, i] = Vt[-1]  # Última fila de Vt es la solución homogénea
    
    # Convertir a coordenadas 3D dividiendo por la cuarta coordenada
    X = X_homogeneous / X_homogeneous[3, :]
    return X[:3]

def triangulate_from_superglue_pair(data, K, params, state, idx_file=0):
    # Paso 1: Estimar pose
    R, t, T_c2c1, extra = pose.estimate_baseline_pose_from_superglue(data, K, params, idx_file=idx_file)
    if R is None:
        return None, None

    pts0 = extra['pts0_inliers'] # Nx2
    pts1 = extra['pts1_inliers'] # Nx2
    idx0 = extra['match_indices'][0]  # índices en keypoints0
    idx1 = extra['match_indices'][1]  # índices en keypoints1

    T_wc1 = np.eye(4)   # se toma la primera cámara como referencia
    T_wc2 = T_c2c1 @ T_wc1  

    P1 = projection.projectionMatrix(K, T_wc1)
    P2 = projection.projectionMatrix(K, T_wc2)
    
    x1 = np.vstack((pts0.T, np.ones((1, pts0.T.shape[1]))))  # Homogeneizar
    x2 = np.vstack((pts1.T, np.ones((1, pts0.T.shape[1]))))  # Homogeneizar
    X = triangulate_points(P1, P2, x1, x2)  # 3xN

    img1_name, img2_name = params['path_images'][idx_file]
    img1_name = os.path.splitext(os.path.basename(img1_name))[0]
    img2_name = os.path.splitext(os.path.basename(img2_name))[0]
    
    state.cargar(
        R1=np.eye(3), t1=np.zeros(3),  # Cámara 1 en origen
        R2=R, t2=t,
        idx1=idx0, idx2=idx1,
        puntos_3d=X.T,
        num_kp1=data['keypoints0'].shape[0],
        num_kp2=data['keypoints1'].shape[0],
        img1_name=img1_name,
        img2_name=img2_name,
        T_wc1=T_wc1,
        T_wc2=T_wc2
    )

    return X, x1, x2, T_wc1, T_wc2, state # Devolver como Nx3


def triangulate_pairs(T_wc1, T_wc2, K, pts0, pts1, idx_prev_tri, idx_new_tri, idx_file=0, fund_method=cv2.FM_RANSAC, outlier_thres=1.0, fund_prob=0.999):

    F, msk = cv2.findFundamentalMat(
        pts0, pts1,
        method=fund_method,
        ransacReprojThreshold=outlier_thres,
        confidence=fund_prob
    )

    if F is None:
        return
    
    
    msk = msk.ravel().astype(bool)
    pts_prev, pts_new = pts0[msk], pts1[msk]
    idx_prev_tri, idx_new_tri = idx_prev_tri[msk], idx_new_tri[msk]
    
    P1 = projection.projectionMatrix(K, T_wc1)
    P2 = projection.projectionMatrix(K, T_wc2)

    x1 = np.vstack((pts_prev.T, np.ones((1, pts_prev.T.shape[1]))))  # Homogeneizar
    x2 = np.vstack((pts_new.T, np.ones((1, pts_new.T.shape[1]))))  # Homogeneizar
    X = triangulate_points(P1, P2, x1, x2) # 3xN

    return X.T, idx_prev_tri, idx_new_tri # Nx3


def triagulate_new_image(params, name_image, state, K, verbose=False):
    n_kp_new = state.image_data[name_image]['ref'].shape[0]
    ref_new  = state.image_data[name_image]['ref']
    assert ref_new.shape == (n_kp_new,)

    R2, t2 = state.image_data[name_image]['R'], state.image_data[name_image]['t'].reshape(3, 1)
    T_wc2 = state.image_data[name_image]['t_wc']

    all_superglue = params['path_superglue']

    superglue_for_view = [
        p for p in all_superglue
        if name_image in os.path.basename(p)            
    ]

    for file_path in superglue_for_view:
        filename = os.path.basename(file_path) 
        name_a, name_b, _ = filename.split('_', 2)
        other_image = name_a if name_b == name_image else name_b
        if other_image not in state.image_data:
            continue

        if verbose:
            print(f"[INFO] {name_image} is matched with {other_image}: {file_path}")

        data = np.load(file_path)
        keypoints0, keypoints1, matches = data['keypoints0'], data['keypoints1'], data['matches']
    
        if name_a == name_image: 
            kp_new, kp_prev = keypoints0, keypoints1
            idx_new = np.where(matches >= 0)[0]
            idx_other = matches[idx_new]
        else:
            kp_new, kp_prev = keypoints1, keypoints0
            idx_other = np.where(matches >= 0)[0]
            idx_new = matches[idx_other]

        ref_prev = state.image_data[other_image]['ref']
        R1, t1   = state.image_data[other_image]['R'], state.image_data[other_image]['t'].reshape(3, 1)
        T_wc1 = state.image_data[other_image]['t_wc']

        mask_prev_has3d = ref_prev[idx_other] >= 0

        idx_prev_copy   = idx_other[mask_prev_has3d]
        idx_new_copy    = idx_new[mask_prev_has3d]

        ref_ids_to_copy = ref_prev[idx_prev_copy]
        ref_new[idx_new_copy] = ref_ids_to_copy

        tri_mask = ~mask_prev_has3d & (ref_new[idx_new] < 0)
        if np.count_nonzero(tri_mask) < 8:
            continue  # insuficiente para F

        idx_prev_tri = idx_other[tri_mask]
        idx_new_tri  = idx_new [tri_mask]

        pts_prev = kp_prev[idx_prev_tri, :2]  # (N,2)
        pts_new  = kp_new [idx_new_tri , :2]

        pts3d, idx_prev_tri, idx_new_tri = triangulate_pairs(
            T_wc1=T_wc1,
            T_wc2=T_wc2,
            K=K,
            pts0=pts_prev,
            pts1=pts_new,
            idx_prev_tri=idx_prev_tri,
            idx_new_tri=idx_new_tri,
        )

        start = len(state.point_cloud)
        state.point_cloud = np.vstack([state.point_cloud, pts3d])

        ref_prev[idx_prev_tri] = np.arange(len(pts3d)) + start
        ref_new [idx_new_tri ] = np.arange(len(pts3d)) + start

        state.image_data[other_image]['ref'] = ref_prev 
        state.image_data[name_image]['ref'] = ref_new

        return state




def calculate_new_pose(params, name_image, state, K, name=None, verbose=False):
    all_superglue = params['path_superglue']

    superglue_for_view = [
        p for p in all_superglue
        if name_image in os.path.basename(p)            
    ]

    pts3D, pts2D = [], []
    corr = {}
    max_kp = 0

    for f in superglue_for_view:
        with np.load(f) as d:
            name_a, name_b, _ = os.path.basename(f).split('_', 2)
            if name_a == name_image:
                max_kp = max(max_kp, d['keypoints0'].shape[0])
            else:
                max_kp = max(max_kp, d['keypoints1'].shape[0])

    new_image_index = -np.ones(max_kp, dtype=np.int32)

    for file_path in superglue_for_view:
        filename = os.path.basename(file_path) 
        name_a, name_b, _ = filename.split('_', 2)
        other_image = name_a if name_b == name_image else name_b

        if verbose:
            print(f"[INFO] {name_image} is matched with {other_image}: {file_path}")

        if other_image not in state.image_data:
            continue

        
        ref_kp_indices = state.image_data[other_image]['ref']
        kp_to_3d = {i: kp for i, kp in enumerate(ref_kp_indices) if kp >= 0}

        data = np.load(file_path)
        keypoints0, keypoints1, matches = data['keypoints0'], data['keypoints1'], data['matches']
    
        if name_a == name_image: 
            kp_new = keypoints0 
            idx_new = np.where(matches >= 0)[0]
            idx_other = matches[idx_new]
        else:
            kp_new = keypoints1 
            idx_other = np.where(matches >= 0)[0]
            idx_new = matches[idx_other]

        # kp_new: new image keypoints
        for i_new, i_other in zip(idx_new, idx_other):
            point3d_id = kp_to_3d.get(i_other, -1)       # -1 si no existe
            if point3d_id >= 0:
                new_image_index[i_new] = point3d_id
                if point3d_id not in corr:
                    corr[point3d_id] = kp_new[i_new]


    pts3D = state.point_cloud[list(corr.keys())].astype(np.float32)   # (M, 3)
    pts2D = np.vstack(list(corr.values())).astype(np.float32)
    state.add_view_refs(name_image, new_image_index)

    success, Rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=pts3D[:, np.newaxis],  # puntos 3D
        imagePoints=pts2D[:, np.newaxis],   # puntos 2D
        cameraMatrix=K,                    # matriz intrínseca de la cámara
        distCoeffs=None,                   # distorsión (None si no hay)
        flags=cv2.SOLVEPNP_ITERATIVE       # o EPNP, DLS, etc., según config
    )

    if not success:
        print(f"[ERROR] PnP failed for image {name}. Skipping.")
        return None, None, state

    R, _ = cv2.Rodrigues(Rvec)
    T_wc = np.eye(4)
    T_wc[:3, :3] = R 
    T_wc[:3, 3] = tvec.flatten()

    T_cw = T_wc.copy()
    T_wc = np.linalg.inv(T_cw)

    state.add_r_t(name_image, R, tvec.flatten(), T_wc)
    
    return pts3D, pts2D, state, T_wc, state
   

def dlt_old_camera(state, params, name_image_old):
    all_superglue = params['path_old_superglue']

    superglue_for_view = [
        p for p in all_superglue
        if name_image_old in os.path.basename(p)            
    ]

    xs_2d, Xs_3d = [], []
    seen_ids = set()

    for file_path in superglue_for_view:
        filename = os.path.basename(file_path) 
        name_a, name_b, _ = filename.split('_', 2)
        other_image = name_a if name_b == name_image_old else name_b

        data = np.load(file_path)
        keypoints0, keypoints1, matches = data['keypoints0'], data['keypoints1'], data['matches']
    
        if name_a == name_image_old: 
            kp_old, kp_prev = keypoints0, keypoints1
            idx_new = np.where(matches >= 0)[0]
            idx_other = matches[idx_new]
        else:
            kp_old, kp_prev = keypoints1, keypoints0
            idx_other = np.where(matches >= 0)[0]
            idx_new = matches[idx_other]

        ref_prev = state.image_data[other_image]['ref']
        R1, t1   = state.image_data[other_image]['R'], state.image_data[other_image]['t'].reshape(3, 1)
        T_wc1 = state.image_data[other_image]['t_wc']

        ids_3d_other = ref_prev[idx_other]       # shape (N_matches,)

        # 2) descartar ids repetidos
        mask_new = [pid not in seen_ids for pid in ids_3d_other]
        xs_old   = kp_old[idx_new][mask_new]     # shape (M,2)
        ids_new  = ids_3d_other[mask_new]

        # 3) acumular y marcar como vistos
        Xs_3d.extend(state.point_cloud[ids_new])    # (X,Y,Z)
        xs_2d.extend(xs_old)
        seen_ids.update(ids_new)

    xs_2d = np.asarray(xs_2d, dtype=np.float64)
    Xs_3d = np.asarray(Xs_3d, dtype=np.float64)

    if len(Xs_3d) < 6:
        raise RuntimeError("Necesitas ≥ 6 correspondencias 2D-3D para la DLT")

    Xs_h = np.hstack([Xs_3d, np.ones((Xs_3d.shape[0],1))])   # (N,4)
    xs_h = np.hstack([xs_2d, np.ones((xs_2d.shape[0],1))])   # (N,3)
    
    def estimate_P(sample_idx):
        A = []
        for i in sample_idx:
            X = Xs_h[i]
            u,v,_ = xs_h[i]
            A.append(np.hstack([np.zeros(4), -X, v*X]))
            A.append(np.hstack([ X, np.zeros(4), -u*X]))
        A = np.asarray(A)
        _,_,Vt = np.linalg.svd(A)
        P = Vt[-1].reshape(3,4)
        return P / P[-1,-1]

    # 2. error reproyección
    def reproj_err(P, idx):
        X = Xs_h[idx].T                          # (4,N)
        x_est = (P @ X).T                        # (N,3)

        w = x_est[:, 2]
        valid = np.abs(w) > 1e-6                 # evitar división por cero

        x_est_valid = x_est[valid]
        xs_valid = xs_2d[idx][valid]

        x_proj = x_est_valid[:, :2] / x_est_valid[:, 2, np.newaxis]

        return np.linalg.norm(x_proj - xs_valid, axis=1)


    # 3. RANSAC
    best_P, best_inliers = None, []
    rng = np.random.default_rng(0)
    thr = 3.0                                   # px
    iters = 2000
    for _ in range(iters):
        sample = rng.choice(len(Xs_3d), 6, replace=False)
        P_try = estimate_P(sample)
        err   = reproj_err(P_try, np.arange(len(Xs_3d)))
        inl   = np.where(err < thr)[0]
        if len(inl) > len(best_inliers):
            best_inliers, best_P = inl, P_try
            if len(inl) > 0.8*len(Xs_3d): break

    # 4. refinar con todos los inliers
    P_refined = estimate_P(best_inliers)
    err_mean  = reproj_err(P_refined, best_inliers).mean()
    print(f"[DLT] #inliers={len(best_inliers)}/{len(Xs_3d)}, RMSE={err_mean:.2f} px")

    # 5. descomponer P  →  K, R, t
    K, R, C, _, _, _, _ = cv2.decomposeProjectionMatrix(P_refined)
    K /= K[-1,-1]
    t = -R @ (C[:3]/C[3])

    T_wc = np.eye(4)
    T_wc[:3, :3] = R 
    T_wc[:3, 3] = t.flatten()

    T_cw = T_wc.copy()
    T_wc = np.linalg.inv(T_cw)

    state.image_data[name_image_old] = {
        'R': R,       # 3x3
        't': t.reshape(3,1),   # 3x1
        't_wc': T_wc,
        'K': K
    }
    return state

