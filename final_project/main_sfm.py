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

# Import the necessary functions from the files
import utils.camera_calibration as cam_calib
import utils.utils as utils
import utils.math as ut_math
import utils.plot_utils as plot_utils
import utils.bundle_adjustment as bun_adj
import utils.functions_cv as fcv


class DMatch:
    def __init__(self, idx1, idx2, pt1, pt2, dist):
        self.ind_key = (idx1, idx2)
        self.point = (pt1, pt2)
        self.dist = dist  # SuperGlue confidence or distance

class Pair:
    def __init__(self, cam1, cam2, matches):
        self.cams = [cam1, cam2]
        self.matches = matches  # List of DMatch
        self.F = None  # Fundamental matrix
        self.E = None  # Essential matrix
        self.intrinsics_mat = []
        self.extrinsics_mat = []

class StructurePoint:
    def __init__(self, point3D, observations):
        self.point3D = point3D  # 3D coordinates
        self.observations = observations  # List of (cam_idx, 2D point)

class Track:
    def __init__(self):
        self.observations = []  # list of (cam_idx, 2D point)

    def add(self, cam_idx, pt2d):
        self.observations.append((cam_idx, pt2d))

    def get_camera_indices(self):
        return [obs[0] for obs in self.observations]

    def get_points(self):
        return [obs[1] for obs in self.observations]

class Graph:
    def __init__(self):
        self.cams = []  # list of camera indices
        self.intrinsics = []  # list of K matrices
        self.extrinsics = []  # list of [R|t] matrices
        self.tracks = []  # list of Track
        self.structure_points = []  # list of StructurePoint



def rodrigues_to_matrix(rvec):
    """Convierte vector de rotación a matriz 3x3."""
    R, _ = cv2.Rodrigues(rvec)
    return R

def matrix_to_rodrigues(R):
    """Convierte matriz de rotación a vector de rotación."""
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()

def flatten_params(graph):
    """Convierte extrínsecos y puntos 3D a un vector 1D para optimización."""
    extrinsics = []
    for Rt in graph.extrinsics:
        R = Rt[:, :3]
        t = Rt[:, 3]
        rvec = matrix_to_rodrigues(R)
        extrinsics.append(np.concatenate([rvec, t]))
    extrinsics = np.concatenate(extrinsics)
    points3D = np.concatenate([sp.point3D for sp in graph.structure_points])
    return np.concatenate([extrinsics, points3D])

def unflatten_params(params, graph):
    """Reconstruye extrínsecos y puntos 3D desde vector plano."""
    num_cams = len(graph.extrinsics)
    extrinsics = []
    idx = 0
    for _ in range(num_cams):
        rvec = params[idx:idx+3]
        t = params[idx+3:idx+6]
        R = rodrigues_to_matrix(rvec)
        Rt = np.hstack([R, t.reshape(3, 1)])
        extrinsics.append(Rt)
        idx += 6
    points3D = []
    for _ in range(len(graph.structure_points)):
        pt = params[idx:idx+3]
        points3D.append(pt)
        idx += 3
    return extrinsics, points3D

def residuals(params, graph):
    """Calcula los residuos (error de reproyección) para todas las observaciones."""
    extrinsics, points3D = unflatten_params(params, graph)
    residuals = []
    for pt_idx, sp in enumerate(graph.structure_points):
        pt3d = points3D[pt_idx]
        pt3d_homog = np.hstack((pt3d, 1.0))
        for cam_idx, pt2d in sp.observations:
            K = graph.intrinsics[cam_idx]
            Rt = extrinsics[cam_idx]
            P = K @ Rt
            proj = P @ pt3d_homog
            proj /= proj[2]
            reproj = proj[:2]
            residuals.extend(reproj - pt2d)
    return np.array(residuals)

def bundle_adjustment(graph):
    """Optimiza extrínsecos y puntos 3D para minimizar el error de reproyección."""
    x0 = flatten_params(graph)
    result = scipy.optimize.least_squares(residuals, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf', args=(graph,))
    optimized_extrinsics, optimized_points3D = unflatten_params(result.x, graph)
    for i, Rt in enumerate(optimized_extrinsics):
        graph.extrinsics[i] = Rt
    for i, pt3d in enumerate(optimized_points3D):
        graph.structure_points[i].point3D = pt3d
    return result.success


def reprojection_error(graph):
    """
    Calcula el error de reproyección promedio sobre todos los puntos triangulados del grafo.
    """
    total_error = 0.0
    total_tracks = 0

    for sp in graph.structure_points:
        err = 0.0
        n_obs = len(sp.observations)
        for cam_idx, pt2d in sp.observations:
            K = graph.intrinsics[cam_idx]
            Rt = graph.extrinsics[cam_idx]
            P = K @ Rt  # matriz de proyección 3x4

            pt3d_homog = np.hstack((sp.point3D, 1.0))  # [X, Y, Z, 1]
            proj = P @ pt3d_homog  # coordenada proyectada en homogénea
            proj /= proj[2]  # normalizar

            reproj_pt = proj[:2]
            err += np.linalg.norm(reproj_pt - pt2d)

        err /= n_obs
        total_error += err
        total_tracks += 1

    return total_error / total_tracks if total_tracks > 0 else float('inf')

def triangulate_nview(graph):
    points_3d = []
    for track in graph.tracks:
        pts_2d = []
        proj_mats = []
        for cam_idx, pt2d in track.observations:
            P = graph.intrinsics[cam_idx] @ graph.extrinsics[cam_idx]
            proj_mats.append(P)
            pts_2d.append(pt2d)

        if len(proj_mats) >= 2:
            A = []
            for P, pt in zip(proj_mats, pts_2d):
                x, y = pt
                row1 = x * P[2] - P[0]
                row2 = y * P[2] - P[1]
                A.append(row1)
                A.append(row2)
            A = np.array(A)
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X / X[3]
            point3D = X[:3]
            graph.structure_points.append(StructurePoint(point3D, track.observations))
            points_3d.append(point3D)
    return np.array(points_3d)

def triangulate_nview(graph):
    points_3d = []
    for track in graph.tracks:
        pts_2d = []
        proj_mats = []
        for cam_idx, pt2d in track.observations:
            P = graph.intrinsics[cam_idx] @ graph.extrinsics[cam_idx]
            proj_mats.append(P)
            pts_2d.append(pt2d)

        if len(proj_mats) >= 2:
            A = []
            for P, pt in zip(proj_mats, pts_2d):
                x, y = pt
                row1 = x * P[2] - P[0]
                row2 = y * P[2] - P[1]
                A.append(row1)
                A.append(row2)
            A = np.array(A)
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X / X[3]
            point3D = X[:3]
            graph.structure_points.append(StructurePoint(point3D, track.observations))
            points_3d.append(point3D)
    return np.array(points_3d)

def build_graph_from_pairs(pairs, K):
    graph = Graph()
    cam_indices = set()

    # Asumimos que cada par tiene K constante, así que podemos llenar las intrínsecas aquí
    for pair in pairs:
        cam_indices.update(pair.cams)

    # Ordenar cámaras por índice
    cam_indices = sorted(list(cam_indices))
    cam_idx_map = {cam: idx for idx, cam in enumerate(cam_indices)}
    graph.cams = cam_indices
    graph.intrinsics = [K for _ in cam_indices]
    graph.extrinsics = [None for _ in cam_indices]  # Se llenarán a partir de los pares

    # Llenar extrinsics desde los pares
    for pair in pairs:
        idx1 = cam_idx_map[pair.cams[0]]
        idx2 = cam_idx_map[pair.cams[1]]

        if graph.extrinsics[idx1] is None:
            graph.extrinsics[idx1] = np.eye(3, 4)
        graph.extrinsics[idx2] = pair.extrinsics_mat[1]  # [R|t] de la segunda cámara

    # Agrupar matches por correspondencia única de keypoints (tracking simplificado)
    # Este método es simplificado: cada match genera un track independiente
    tracks = []
    track_map = defaultdict(list)

    for pair in pairs:
        cam1, cam2 = pair.cams
        idx1 = cam_idx_map[cam1]
        idx2 = cam_idx_map[cam2]
        for match in pair.matches:
            pt1 = match.point[0]
            pt2 = match.point[1]

            track = Track()
            track.add(idx1, pt1)
            track.add(idx2, pt2)
            tracks.append(track)

    graph.tracks = tracks
    return graph

def estimate_pose_from_pairs(pairs, K, dist, inlier_threshold=0.6):
    for pair in pairs:
        # Extraer puntos emparejados
        pts1 = np.array([match.point[0] for match in pair.matches], dtype=np.float32)
        pts2 = np.array([match.point[1] for match in pair.matches], dtype=np.float32)

        if len(pts1) < 8:
            print(f"Not enough matches for Pair({pair.cams[0]}, {pair.cams[1]})")
            continue

        # Eliminar distorsión si existe
        if dist is not None and np.any(dist != 0):
            pts1 = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, dist).reshape(-1, 2)
            pts2 = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, dist).reshape(-1, 2)

        # Estimar matriz fundamental con RANSAC
        F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)

        if F is None or np.sum(mask) < 8:
            print(f"Failed to estimate F for Pair({pair.cams[0]}, {pair.cams[1]})")
            continue

        pair.F = F

        # Calcular matriz esencial desde F y K
        E = K.T @ F @ K
        pair.E = E
        pair.intrinsics_mat = [K, K]  # ambas cámaras usan la misma intrínseca

        # Estimar rotación y traslación (pose relativa)
        retval, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

        if retval < 8:
            print(f"Pose recovery failed for Pair({pair.cams[0]}, {pair.cams[1]})")
            continue

        # Armar matriz extrínseca [R | t]
        Rt = np.hstack((R, t))
        pair.extrinsics_mat = [np.eye(3, 4), Rt]  # cámara 1 en (I|0), cámara 2 relativa
        print(f"Pose estimated for Pair({pair.cams[0]}, {pair.cams[1]}) with {retval} inliers.")

def process_matches(params, K, dist):
    path_new_images = params['path_new_images']
    path_new_images = glob.glob(path_new_images)
    path_new_images = sorted(path_new_images)

    match_files = params['path_superglue']
    path_images = params['path_images']

    resize_dim = tuple(params['resize_dim']) if params['resize_dim'] else None

    pairs = []

    for file_path in match_files:
        data = np.load(file_path)
        
        keypoints0 = data['keypoints0']  # Nx2
        keypoints1 = data['keypoints1']  # Mx2
        matches = data['matches']        # length N0
        scores = data.get('match_scores', np.ones_like(matches, dtype=np.float32))

        matches_list = []

        for idx0, idx1 in enumerate(matches):
            if idx1 >= 0:
                pt0 = keypoints0[idx0]
                pt1 = keypoints1[idx1]
                score = scores[idx0]
                dist = 1.0 - score  # puede ajustar esta métrica si lo deseas
                matches_list.append(DMatch(idx0, idx1, pt0, pt1, dist))
        
        fname = os.path.basename(file_path)
        tokens = fname.split("_")
        cam1 = int(tokens[2])  # Suponiendo nombres como IMG_2_4_3_IMG_2_5_3_matches.npz
        cam2 = int(tokens[6])
    
        pairs.append(Pair(cam1, cam2, matches_list))
        print(f"Loaded {len(matches_list)} matches for Pair({cam1}, {cam2})")
        estimate_pose_from_pairs(pairs, K, dist)

        graph = build_graph_from_pairs(pairs, K)
        pts3D = triangulate_nview(graph)
        print(f"Triangulated {len(pts3D)} points")
        points_3d_pose = np.array([sp.point3D for sp in graph.structure_points]).T  # Shape (3, N)
        print(f"3D points shape: {points_3d_pose.shape}")
        error = reprojection_error(graph)
        print(f"Reprojection error (BEFORE bundle adjustment): {error:.4f}")

        success = bundle_adjustment(graph)
        if success:
            error_after = reprojection_error(graph)
            print(f"Reprojection error (AFTER bundle adjustment): {error_after:.4f}")
        else:
            print("Bundle Adjustment failed.")
        cameras = {}
        for idx, extr in enumerate(graph.extrinsics):
            if extr is None:
                continue  # omitir si no está definida

            # Convertir [R|t] en T_wc (transformación del mundo a cámara)
            # T_wc = inv([R|t])
            Rt = np.vstack((extr, np.array([[0, 0, 0, 1]])))  # Hacerlo 4x4
            T_cw = np.eye(4)
            T_cw[:3, :] = extr
            T_wc = np.linalg.inv(T_cw)  # transformación del mundo a cámara
            cameras[f"C{idx}"] = T_wc
        
        plot_utils.plot3DPoints(points_3d_pose, cameras, world_ref=False)
        

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

