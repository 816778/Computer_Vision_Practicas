import numpy as np
import os
import cv2


from utils import projection
from utils import pose
from utils import reconstruction

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
        index_key = other_image + "_index" 

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

    triangulated_data = {
        "points3D": X.T,    # Nx3
        img1_name + "_index": idx0,  # Nx1
        img2_name + "_index": idx1,  # Nx1
    }

    print(f"[INFO] Triangulated {len(X.T)} points between {img1_name} and {img2_name}")
    for key, value in triangulated_data.items():
        try:
            shape = value.shape
        except AttributeError:
            shape = "No tiene atributo 'shape'"
        print(f"Key: '{key}' - Shape: {shape}")
    return X, x1, x2, T_wc1, T_wc2, triangulated_data, state # Devolver como Nx3


