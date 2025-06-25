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

    def __repr__(self):
        return f"DMatch({self.ind_key_}, {self.point_}, dist={self.dist_:.4f})"

class Pair:
    def __init__(self, cam1, cam2, matches):
        self.cams = [cam1, cam2]
        self.matches = matches  # List of DMatch
        self.F = None  # Fundamental matrix
        self.E = None  # Essential matrix
        self.intrinsics_mat = []
        self.extrinsics_mat = []

    def __repr__(self):
        return f"Pair(cams={self.cams_}, num_matches={len(self.matches_)})"

class Keypoint:
    def __init__(self, pt, cam_id):
        self.pt = pt
        self.cam_id = cam_id

class Track:
    def __init__(self):
        self.keypoints = []

    def add_keypoint(self, keypoint):
        self.keypoints.append(keypoint)

class StructurePoint:
    def __init__(self, coord=None, color=None):
        self.coord = coord
        self.color = color

class Graph:
    def __init__(self, pair):
        self.init_from_pair(pair)

    def init_from_pair(self, pair):
        self.ncams = 2
        self.cams = pair.cams
        self.intrinsics_mat = pair.intrinsics_mat
        self.extrinsics_mat = pair.extrinsics_mat

        self.tracks = []
        self.structure_points = []

        for match in pair.matches:
            track = Track()
            track.add_keypoint(Keypoint(match.point[0], pair.cams[0]))
            track.add_keypoint(Keypoint(match.point[1], pair.cams[1]))
            self.tracks.append(track)
            self.structure_points.append(StructurePoint())


def triangulate_and_refine(graph):
    for i, track in enumerate(graph.tracks):
        pts = []
        Ps = []

        for keypoint in track.keypoints:
            cam_idx = graph.cams.index(keypoint.cam_id)
            K = graph.intrinsics_mat[cam_idx]
            Rt = graph.extrinsics_mat[cam_idx]
            P = K @ Rt
            Ps.append(P)
            pts.append(np.array(keypoint.pt))

        if len(Ps) < 2:
            continue

        pt1 = pts[0].reshape(2, 1)
        pt2 = pts[1].reshape(2, 1)
        P1 = Ps[0]
        P2 = Ps[1]

        point_4d = cv2.triangulatePoints(P1, P2, pt1, pt2)
        point_3d = point_4d[:3] / point_4d[3]

        # Refinar punto con Gauss-Newton
        refined_3d = refine_point_3d(point_3d.flatten(), pts, Ps)

        graph.structure_points[i].coord = refined_3d


def refine_point_3d(initial_point, keypoints, projections, max_iter=30, tol=1e-6):
    X = initial_point.reshape(3, 1)

    for _ in range(max_iter):
        J = []
        r = []

        for pt2d, P in zip(keypoints, projections):
            # Project 3D point
            X_homog = np.vstack((X, [[1.0]]))  # 4x1
            proj = P @ X_homog  # 3x1
            proj /= proj[2]

            # Residual
            e = proj[:2].flatten() - pt2d
            r.extend(e)

            # Derivative (Jacobian)
            x, y, z = proj[0, 0], proj[1, 0], proj[2, 0]
            d = P[2] @ X_homog

            J_i = np.zeros((2, 3))
            for j in range(3):
                dPj = P[:2, j] * d - P[2, j] * proj[:2].flatten()
                J_i[:, j] = dPj / (d ** 2)

            J.append(J_i)

        r = np.array(r).reshape(-1, 1)
        J = np.vstack(J)

        # Solve Gauss-Newton update
        delta_X, _, _, _ = np.linalg.lstsq(J, -r, rcond=None)
        X += delta_X

        if np.linalg.norm(delta_X) < tol:
            break

    return X.flatten()

def visualizar_estructura_3d(graphs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for graph in graphs:
        coords = [sp.coord for sp in graph.structure_points if sp.coord is not None]
        if coords:
            coords = np.array(coords)
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Reconstrucción 3D - Puntos triangulados')
    plt.show()

def triangulate_linear(graph):
    for i, track in enumerate(graph.tracks):
        pts = []
        Ps = []

        for keypoint in track.keypoints:
            cam_idx = graph.cams.index(keypoint.cam_id)  # posición en listas
            K = graph.intrinsics_mat[cam_idx]
            Rt = graph.extrinsics_mat[cam_idx]
            P = K @ Rt  # matriz de proyección 3x4
            Ps.append(P)
            pts.append(keypoint.pt)

        if len(Ps) < 2:
            continue  # se necesita al menos dos vistas para triangulación

        pt1 = np.array(pts[0]).reshape(2, 1)
        pt2 = np.array(pts[1]).reshape(2, 1)
        P1 = Ps[0]
        P2 = Ps[1]

        point_4d = cv2.triangulatePoints(P1, P2, pt1, pt2)
        point_3d = point_4d[:3] / point_4d[3]

        graph.structure_points[i].coord = point_3d.flatten()


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

        inlier_matches = [m for m, inlier in zip(pair.matches, mask_pose.ravel()) if inlier]
        pair.matches = inlier_matches

        # Armar matriz extrínseca [R | t]
        Rt = np.hstack((R, t))
        pair.extrinsics_mat = [np.eye(3, 4), Rt]  # cámara 1 en (I|0), cámara 2 relativa
        print(f"Pose estimated for Pair({pair.cams[0]}, {pair.cams[1]}) with {retval} inliers.")

def process_matches(params, K, dist, confidence_threshold=0.8):

    """
    path_new_images = params['path_new_images']
    path_new_images = glob.glob(path_new_images)
    path_new_images = sorted(path_new_images)
    path_images = params['path_images']
    resize_dim = tuple(params['resize_dim']) if params['resize_dim'] else None
    """
    match_files = params['path_superglue']
    pairs = []

    for file_path in match_files:
        data = np.load(file_path)
        
        keypoints0 = data['keypoints0']  # Nx2
        keypoints1 = data['keypoints1']  # Mx2
        matches = data['matches']        # length N0
        scores = data.get('match_confidence', np.ones_like(matches, dtype=np.float32))

        matches_list = []

        for idx0, idx1 in enumerate(matches):
            if idx1 >= 0:
                score = scores[idx0]
                if score < confidence_threshold:
                    continue
                pt0 = keypoints0[idx0]
                pt1 = keypoints1[idx1]
                dist = 1.0 - score 
                matches_list.append(DMatch(idx0, idx1, pt0, pt1, dist))
        
        fname = os.path.basename(file_path)
        tokens = fname.split("_")
        cam1 = int(tokens[2])  # Nombres como IMG_2_4_3_IMG_2_5_3_matches.npz
        cam2 = int(tokens[6])
        pairs.append(Pair(cam1, cam2, matches_list))
        print(f"Loaded {len(matches_list)} matches for Pair({cam1}, {cam2})")
   
    estimate_pose_from_pairs(pairs, K, dist)

    graphs = []
    for pair in pairs:
        if pair.F is not None and pair.E is not None and pair.extrinsics_mat:
            graph = Graph(pair)
            triangulate_linear(graph)
            graphs.append(graph)
            print(f"Graph creado para cámaras {pair.cams[0]} y {pair.cams[1]} con {len(graph.tracks)} tracks.")
        else:
            print(f"Pair {pair.cams} no tiene F/E válidos. Saltando.")

    visualizar_estructura_3d(graphs)

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

