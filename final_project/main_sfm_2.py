# Import the necessary libraries
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import argparse
import json
import os
from collections import defaultdict
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.linalg import expm, logm
import scipy.optimize as scOptim

# Import the necessary functions from the files
import utils.camera_calibration as cam_calib
import utils.utils as utils
import utils.math as ut_math
import utils.plot_utils as plot_utils
import utils.bundle_adjustment as bun_adj
import utils.functions_cv as fcv

def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c

def crossMatrix(x):
    """Genera la matriz skew-symmetric de un vector."""
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def crossMatrixInv(M):
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return x


def project_points(K, T, X_w):
    if T.shape == (3, 4):
        T_hom = np.vstack([T, [0, 0, 0, 1]])
    else:
        T_hom = T

    if X_w.shape[0] == 3:
        X_w_hom = np.vstack([X_w, np.ones((1, X_w.shape[1]))])
    else:
        X_w_hom = X_w

    x_proj_hom = K @ np.eye(3, 4) @ np.linalg.inv(T_hom) @ X_w_hom
    x_proj = x_proj_hom / x_proj_hom[2, :]
    return x_proj


def projectionMatrix(K, T_w_c):
    R_w_c = T_w_c[0:3, 0:3]  # La rotación: parte superior izquierda (3x3)
    t_w_c = T_w_c[0:3, 3]    # La traslación: última columna (3x1)
    
    T_w_c = ensamble_T(R_w_c, t_w_c)

    # Invert to get world from the camera
    T_c_w = np.linalg.inv(T_w_c)
    
    # K*[R|t] para obtener la matriz de proyección
    P = K @ T_c_w[0:3, :]

    return P

def triangulate_from_superglue_pair(data, K, params):
    # Paso 1: Estimar pose
    R, t, T_c2c1, extra = estimate_baseline_pose_from_superglue(data, K, params)
    if R is None:
        return None, None

    pts0 = extra['pts0_inliers'] # Nx2
    pts1 = extra['pts1_inliers'] # Nx2

    T_wc1 = np.eye(4)   # se toma la primera cámara como referencia
    T_wc2 = T_c2c1 @ T_wc1  

    cameras = {'C1': T_wc1, 'C2': T_wc2}
    plot_utils.plot3DCameras(cameras)
    exit()
    P1 = projectionMatrix(K, T_wc1)
    P2 = projectionMatrix(K, T_wc2)
    """
    pts0_hom = cv2.convertPointsToHomogeneous(pts0)[:, 0, :]  # Nx3
    pts1_hom = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]  # Nx3

    pts0_norm = (np.linalg.inv(K) @ pts0_hom.T).T  # Nx3
    pts1_norm = (np.linalg.inv(K) @ pts1_hom.T).T

    x1 = pts0_norm[:, :2].T  # 2xN
    x2 = pts1_norm[:, :2].T
    """
    
    x1 = np.vstack((pts0.T, np.ones((1, pts0.T.shape[1]))))  # Homogeneizar
    x2 = np.vstack((pts1.T, np.ones((1, pts0.T.shape[1]))))  # Homogeneizar

    X = triangulate_points(P1, P2, x1, x2)  # 3xN
    
    # cameras ={'C1': T_wc1, 'C2': T_wc2}
    # plot_utils.plot3DPoints(X, cameras, world_ref=True)
    # plot_3d_points(X.T)

    return X.T, x1, x2, T_wc1, T_wc2   # Devolver como Nx3


def draw_matches(keypoints1_matched, keypoints2_matched, image_1_path, image_2_path):
    print(f"[INFO] Total matches: {len(keypoints1_matched)}")
    print(f"image_1_path: {image_1_path}")
    print(f"image_2_path: {image_2_path}")

    keypoints_cv1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in keypoints1_matched]
    keypoints_cv2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in keypoints2_matched]

    srcPts = keypoints1_matched  # Ya es un array con coordenadas (x, y)
    dstPts = keypoints2_matched
    x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))

    # Crear objetos DMatch con índices secuenciales
    matches_cv = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(keypoints_cv1))]

    img1 = cv2.imread(image_1_path)
    img2 = cv2.imread(image_2_path)

    img1 = cv2.resize(img1, (2000, 1126))
    img2 = cv2.resize(img2, (2000, 1126))
    # Dibujar los emparejamientos
    img_matches = cv2.drawMatches(img1, keypoints_cv1, img2, keypoints_cv2, matches_cv, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Mostrar el resultado
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title("Emparejamientos SuperGlue")
    plt.subplots_adjust(
        top=0.985,     # Border for top
        bottom=0.015,  # Border for bottom
        left=0.028,    # Border for left
        right=0.992,   # Border for right
    )
    plt.axis('off')
    plt.show()



def estimate_baseline_pose_from_superglue(data, K, params, draw_matches=False, fund_method=cv2.FM_RANSAC, outlier_thres=1.0, fund_prob=0.99):
    keypoints0 = data['keypoints0']  # Nx2
    keypoints1 = data['keypoints1']  # Mx2
    matches = data['matches']        # length N0
    confidence = data.get('match_confidence', np.ones_like(matches, dtype=np.float32))

    matched_idx0 = np.where(matches >= 0)[0]
    matched_idx1 = matches[matched_idx0]

    pts0 = keypoints0[matched_idx0]
    pts1 = keypoints1[matched_idx1]
    conf = confidence[matched_idx0]

    if draw_matches:
        draw_matches(pts0, pts1, params['path_images'][0][0], params['path_images'][0][1])

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

    T_c2c1 = ensamble_T(R21, t21.ravel())
    # T_c2c1 = np.linalg.inv(T_c2c1) 

    R, t = R21, t21
    return R, t, T_c2c1, {
        "F": F,
        "E": E,
        "pts0_inliers": pts0[mask],
        "pts1_inliers": pts1[mask],
        "conf_inliers": conf[mask],
        "match_indices": (matched_idx0[mask], matched_idx1[mask])
    }

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

def plot_3d_points(pts3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], s=1)
    ax.set_title("Reconstrucción 3D - Puntos triangulados")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def ply_from_superglue(params, extra, Xw, output_dir="results", filename='output.ply'):

    def save_ply(X, colors, filename):
        with open(filename, 'w') as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {X.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            print("Shape de X:", X.shape)
            for i in range(X.shape[0]):
                x, y, z = X[i]
                r, g, b = colors[i]
                f.write(f"{x} {y} {z} {r} {g} {b}\n")

    print(params['path_images'][0][0])
    print(params['path_images'][0][1])
    img0 = cv2.imread(sorted(params['path_images'])[0][0])
    img1 = cv2.imread(sorted(params['path_images'])[0][1])
    print("Imagenes cargadas:", img0.shape, img1.shape)

    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    idx0_valid, idx1_valid = extra["match_indices"]
    pts0 = extra["pts0_inliers"]
    pts1 = extra["pts1_inliers"]

    colors = np.zeros((Xw.shape[0], 3), dtype=np.uint8)
    for i in range(Xw.shape[0]):
        x, y = np.round(pts0[i]).astype(int)
        if 0 <= x < img0.shape[1] and 0 <= y < img0.shape[0]:
            colors[i] = img0[y, x]
        else:
            x1, y1 = np.round(pts1[i]).astype(int)
            if 0 <= x1 < img1.shape[1] and 0 <= y1 < img1.shape[0]:
                colors[i] = img1[y1, x1]

    save_ply(Xw, colors, os.path.join(output_dir, filename))

def run_bundle_adjustment(T, K, X_w, xData):

    def residual_bundle_adjustment(params, K, xData, nImages):

        X_w = params[nImages*6:].reshape(-1, 3).T
        residuals = np.array([])

        for i in range(nImages):
            t = params[i*6:(i*6)+3]
            th = params[(i*6)+3:(i*6)+6]
            R = expm(crossMatrix(th))

            T_wc = np.zeros((3, 4))
            T_wc[:3, :3] = R
            T_wc[:3, 3] = t

            x_proj = project_points(K, T_wc, X_w)

            residuals = np.hstack((residuals,
                ((x_proj[:2, :] - xData[i][:2, :])).flatten()
            ))

        print("Residuals: ", residuals.mean())

        return residuals
    
    if X_w.shape[0] == 4:
        X_w = (X_w[:3, :] / X_w[3, :])
    
    nImages = len(T)

    T_flat = np.array([])
    for i in range(nImages):
        t = T[i][:3, 3]
        R = T[i][:3, :3]
        th = crossMatrixInv(logm(R))

        T_flat = np.hstack((T_flat, t, th))

    X_w_flat = X_w.T.flatten()

    initial_params = np.hstack((T_flat, X_w_flat))
    #initial_params = np.hstack((T_flat, X_w_flat))
    
    #residual_bundle_adjustment(initial_params, K, xData, nImages)
    #exit(0)
    
    result = scOptim.least_squares(residual_bundle_adjustment, initial_params,
                                   args=(K, xData, nImages), method='lm')
    
    optimized_params = result.x

    T_opt = []
    for i in range(nImages):
        t = optimized_params[i*6:(i*6)+3]
        th = optimized_params[(i*6)+3:(i*6)+6]
        R = expm(crossMatrix(th))

        T_wc = np.zeros((3, 4))
        T_wc[:3, :3] = R
        T_wc[:3, 3] = t

        T_opt.append(T_wc)

    X_w_opt = optimized_params[nImages*6:].reshape(-1, 3).T

    return T_opt, X_w_opt


def compute_reprojection_error(x_gt, x_proj):
    x_gt_e = x_gt[:2] / x_gt[2:]
    x_proj_e = x_proj[:2] / x_proj[2:]
    return np.linalg.norm(x_gt_e - x_proj_e, axis=0)

def process_matches(params, K, dist, confidence_threshold=0.8):

    """
    path_new_images = params['path_new_images']
    path_new_images = glob.glob(path_new_images)
    path_new_images = sorted(path_new_images)
    path_images = params['path_images']
    resize_dim = tuple(params['resize_dim']) if params['resize_dim'] else None
    """
    match_files = params['path_superglue']
    resize_dim = tuple(params['resize_dim']) if params['resize_dim'] else None

    file_path = match_files[0]
    data = np.load(file_path)

    Xw, x1, x2, T_wc1, T_wc2 = triangulate_from_superglue_pair(data, K, params)

    print("Shape de Xw:", Xw.shape) # Nx3
    print("Shape de x1:", x1.shape) # 3xN
    print("Shape de x2:", x2.shape) # 3xN
    # ply_from_superglue(params, extra, Xw, output_dir="results", filename='output.ply')

    x1_no_opt = project_points(K, T_wc1, Xw.T)
    x2_no_opt = project_points(K, T_wc2, Xw.T)
    print("Shape de x1_no_opt:", x1_no_opt.shape) # 3xN
    print("Shape de x2_no_opt:", x2_no_opt.shape) # 3xN

    """T_opt, X_w_opt = run_bundle_adjustment([T_wc1, T_wc2], K, Xw, [x1, x2])

    T_wc1_opt = T_opt[0]
    T_wc2_opt = T_opt[1]

    x1_p_opt = project_points(K, T_wc1_opt, X_w_opt)
    x2_p_opt = project_points(K, T_wc2_opt, X_w_opt)

    image1 = cv2.imread(params['path_images'][0][0])

    plot_utils.visualize_projection_2(image1, x1, x1_no_opt, x1_p_opt, 'Image 1')
    """
    exit()

    
    image2 = cv2.imread(params['path_images'][0][1])

    plot_utils.visualize_projection(image1, x1, x1_no_opt, 'Image 1', resize_dim=resize_dim)
    plot_utils.visualize_projection(image2, x2, x2_no_opt, 'Image 2', resize_dim=resize_dim)

    


if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)
    nimages = 10
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)
    parser = argparse.ArgumentParser(description="Ejemplo de script con argumento 'test' desde la línea de comandos.")
    
    # Definir el argumento para 'test', con valor por defecto 0
    parser.add_argument(
        '--test', 
        type=int, 
        default=5, 
        help="Valor de la variable 'test'. Valor por defecto es 0."
    )


    args = parser.parse_args()
    test = args.test

    with open('data/config.json', 'r') as f:
        config = json.load(f)

    if test == 0:
        params = config['test_0']
        K, dist, rvecs, tvecs = cam_calib.load_calibration_data("data/camera_calibration.npz")
    elif test == 1:
        params = config['test_1']
        K = np.loadtxt("data/K_c.txt")
    elif test == 2:
        params = config['test_2']
        K, dist, rvecs, tvecs = cam_calib.load_calibration_data("data/camera_calibration.npz")
    else:
        params = config['test_5']
        K, dist, rvecs, tvecs = cam_calib.load_calibration_data("data/camera_calibration_2.npz")
    
    process_matches(params, K, dist)

