####################################################################################################
#
# Title: main.py
# Project: 
# Authors: Eryka Rimacuna
# Description: This file contains the necessary imports for the main.py file.
#
####################################################################################################

# Import the necessary libraries
import numpy as np
import glob
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import argparse
import json

# Import the necessary functions from the files
import utils.camera_calibration as cam_calib
import utils.utils as utils
import utils.plot_utils as plot_utils
import utils.functions_cv as fcv
import utils.math as ut_math
import utils.bundle_adjustment as b_adj



def process_images_sift(image_paths, K):
    poses, keypoints_list, common_coordinates = fcv.emparejamiento_sift_multiple_4(image_paths, K, verbose=True)

    x1, x2, x3 = [], [], []
    for coord_list in common_coordinates:
        x1.append(coord_list[0])  
        x2.append(coord_list[1])  
        x3.append(coord_list[2])

    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)

    print(f"x1: {x1.shape}")
    print(f"x2: {x2.shape}")
    print(f"x3: {x3.shape}")

    image1 = cv2.imread(image_paths[0])
    image2 = cv2.imread(image_paths[1])
    image3 = cv2.imread(image_paths[2])

    plot_utils.visualize_projection_whitoutGT(image1, x1.T, 'Image 1')
    plot_utils.visualize_projection_whitoutGT(image2, x2.T, 'Image 2')
    plot_utils.visualize_projection_whitoutGT(image3, x3.T, 'Image 3')

    exit()

    points_3d = []
    T_list = []
    P_list = []
    T_wc1 = np.eye(4)
    T_list.append(T_wc1)

    for i in range(len(poses) - 1):
        
        R1, t1 = poses[i]
        R2, t2 = poses[i + 1]
        T_wc1 = ut_math.ensamble_T(R1, t1.ravel())
        T_wc2 = ut_math.ensamble_T(R2, t2.ravel())
        T_list.append(T_wc2)
        # P1 = K @ np.hstack((R1, t1))
        # P2 = K @ np.hstack((R2, t2))

        P1 = ut_math.projectionMatrix(K, T_wc1) 
        P2 = ut_math.projectionMatrix(K, T_wc2)
        if i == 0:
            P_list.append(P1)
        
        P_list.append(P2)

    print(f"T_list: {len(T_list)}")
    print(f"P_list len: {len(P_list)}")
    image_points = [x1, x2, x3]
    print(f"len image_points: {len(image_points)}")

    X_w = ut_math.triangulate_points(P1, P2, x1.T, x2.T)

    image_points = [x1.T, x2.T, x3.T]
    T_opt, X_w_opt = b_adj.run_bundle_adjustment(T_list, K, X_w, image_points)

    T_wc1_opt, T_wc2_opt, T_wc3_opt = T_opt

    x1_p_opt = ut_math.project_points(K, T_wc1_opt, X_w_opt)
    x2_p_opt = ut_math.project_points(K, T_wc2_opt, X_w_opt)
    x3_p_opt = ut_math.project_points(K, T_wc3_opt, X_w_opt)

    im1_pth = image_paths[0]
    im2_pth = image_paths[1]
    im3_pth = image_paths[2]
    image1 = cv2.imread(im1_pth)
    image2 = cv2.imread(im2_pth)
    image3 = cv2.imread(im3_pth)

    plot_utils.visualize_projection(image1, x1.T, x1_p_opt, 'Image 1')
    plot_utils.visualize_projection(image2, x2.T, x2_p_opt, 'Image 2')
    plot_utils.visualize_projection(image3, x3.T, x3_p_opt, 'Image 3')

    cameras = {
        'C1': T_wc1_opt,  
        'C2': T_wc2_opt,
        'C3': T_wc3_opt
    }
    print(f"X_w_opt: {X_w_opt.shape}")
    plot_utils.plot3DPoints(X_w_opt, cameras, world_ref=False)



def process_image_sg(path_superglue, path_images, path_images_order, K, resize_dim=None, save=False):
   
    common_coordinates = fcv.emparejamiento_supergluemultiple(path_superglue, path_images, path_images_order, resize_dim=resize_dim, verbose=False)

    puntos_por_imagen = [[] for _ in range(len(common_coordinates[0]))]
    for coord_list in common_coordinates:
        for i, punto in enumerate(coord_list):
            puntos_por_imagen[i].append(punto)

    puntos_por_imagen = [np.array(puntos) for puntos in puntos_por_imagen]
    for i, puntos in enumerate(puntos_por_imagen):
        print(f"x{i+1}: {puntos.shape}")
    watch = False
    if watch:
        for i, image_path in enumerate(path_images_order):
            image = cv2.imread(image_path)
            plot_utils.visualize_projection_whitoutGT(image, puntos_por_imagen[i].T, f"Image {i+1}", resize_dim=resize_dim)
    

    num_imagenes = len(puntos_por_imagen)
    puntos_por_imagen = [p.T for p in puntos_por_imagen]
    P_list, T_list = fcv.estimar_posiciones_camaras(common_coordinates, K)
    T_wc = T_list[:num_imagenes]

    
    X_w = ut_math.triangulate_multiview(P_list, puntos_por_imagen)

    proyecciones_no_opt = []
    proyecciones_opt = []

    for i in range(num_imagenes):
        proyecciones_no_opt.append(ut_math.project_points(K, T_wc[i], X_w))
    cameras = {f'C{i+1}': T_wc[i] for i in range(num_imagenes)}
    plot_utils.plot3DPoints(X_w, cameras, world_ref=False)

    T_wc_opt, X_w_opt = b_adj.run_bundle_adjustment(T_wc, K, X_w, puntos_por_imagen)

    for i in range(num_imagenes):
        proyecciones_opt.append(ut_math.project_points(K, T_wc_opt[i], X_w_opt))

    for i in range(num_imagenes):
        image = cv2.imread(path_images_order[i])
        plot_utils.visualize_projection_2(image, puntos_por_imagen[i], proyecciones_no_opt[i], proyecciones_opt[i], f'Image {i + 1}', resize_dim=resize_dim)


    cameras = {f'C{i+1}': T_wc_opt[i] for i in range(num_imagenes)} 
    plot_utils.plot3DPoints(X_w_opt, cameras, world_ref=False)

    if save:
        np.savez("data/bundle_adjustment_results_pruebas.npz", 
                cameras={f'C{i+1}': T_wc[i] for i in range(num_imagenes)},  
                points_3D=X_w_opt,
                **{f'puntos_{i}': puntos_por_imagen[i] for i in range(num_imagenes)}) 

        print("Resultados guardados en bundle_adjustment_results_pruebas.npz")


def process_image_sg_pnp(path_superglue, path_images, path_images_order, K, dist, resize_dim=None, save=False):

    common_coordinates = fcv.emparejamiento_supergluemultiple(path_superglue, path_images, path_images_order, resize_dim=resize_dim, verbose=False, threshold=0.55)

    puntos_por_imagen = [[] for _ in range(len(common_coordinates[0]))]
    for coord_list in common_coordinates:
        for i, punto in enumerate(coord_list):
            puntos_por_imagen[i].append(punto)

    puntos_por_imagen = [np.array(puntos) for puntos in puntos_por_imagen]
    num_imagenes = len(puntos_por_imagen)
    puntos_por_imagen = [p.T for p in puntos_por_imagen]
    for i, puntos in enumerate(puntos_por_imagen):
        print(f"x{i+1}: {puntos.shape}")
    
    watch = False
    if watch:
        for i, image_path in enumerate(path_images_order):
            image = cv2.imread(image_path)
            plot_utils.visualize_projection_whitoutGT(image, puntos_por_imagen[i].T, f"Image {i+1}", resize_dim=resize_dim)
    

    P_list, T_list = fcv.estimar_posiciones_camaras(common_coordinates, K)
    X_w = ut_math.triangulate_multiview(P_list, puntos_por_imagen)


    # P_list, T_list, X_w = ut_math.estimate_position_solvepnp(puntos_por_imagen, K, P_list, T_list, X_w.T, dist)
    T_wc = T_list[:num_imagenes]

    proyecciones_no_opt = []
    proyecciones_opt = []

    for i in range(num_imagenes):
        proyecciones_no_opt.append(ut_math.project_points(K, T_wc[i], X_w))
    cameras = {f'C{i+1}': T_wc[i] for i in range(num_imagenes)}
    plot_utils.plot3DPoints(X_w, cameras, world_ref=False)

    T_wc_opt, X_w_opt = b_adj.run_bundle_adjustment(T_wc, K, X_w, puntos_por_imagen)

    for i in range(num_imagenes):
        proyecciones_opt.append(ut_math.project_points(K, T_wc_opt[i], X_w_opt))

    for i in range(num_imagenes):
        image = cv2.imread(path_images_order[i])
        plot_utils.visualize_projection_2(image, puntos_por_imagen[i], proyecciones_no_opt[i], proyecciones_opt[i], f'Image {i + 1}', resize_dim=resize_dim)


    cameras = {f'C{i+1}': T_wc_opt[i] for i in range(num_imagenes)} 
    plot_utils.plot3DPoints(X_w_opt, cameras, world_ref=False)

    if save:
        np.savez("data/bundle_adjustment_results_pruebas.npz", 
                cameras={f'C{i+1}': T_wc[i] for i in range(num_imagenes)},  
                points_3D=X_w_opt,
                **{f'puntos_{i}': puntos_por_imagen[i] for i in range(num_imagenes)}) 

        print("Resultados guardados en bundle_adjustment_results_pruebas.npz")

    

    
if __name__ == "__main__":
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)
    parser = argparse.ArgumentParser(description="Ejemplo de script con argumento 'test' desde la línea de comandos.")
    
    # Definir el argumento para 'test', con valor por defecto 0
    parser.add_argument(
        '--test', 
        type=int, 
        default=2, 
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
        params = config['test_4']
        K, dist, rvecs, tvecs = cam_calib.load_calibration_data("data/camera_calibration.npz")
    
    path_new_images = params['path_new_images']
    path_new_images = glob.glob(path_new_images)
    path_new_images = sorted(path_new_images)
    path_superglue = params['path_superglue']
    path_images = params['path_images']
    resize_dim = tuple(params['resize_dim']) if params['resize_dim'] else None
    # path_new_images = path_new_images + ['images/new/old_pilar1936.jpg']
    print(path_new_images)

    process_image_sg_pnp(path_superglue, path_images, path_new_images, K, dist, resize_dim=resize_dim)
    exit()

    