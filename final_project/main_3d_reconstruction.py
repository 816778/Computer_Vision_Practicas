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



def process_image_sg(path_superglue, path_images, path_images_order, K, resize_dim=None):
   
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

    puntos_por_imagen = [puntos_por_imagen[0].T, puntos_por_imagen[1].T, puntos_por_imagen[2].T, puntos_por_imagen[3].T, puntos_por_imagen[4].T]


    P_list, T_list = fcv.estimar_posiciones_camaras(common_coordinates, K)
   
    P1, P2, P3, P4 = P_list
    T_wc1, T_wc2, T_wc3, T_wc4 = T_list

    
    X_w = ut_math.triangulate_multiview(P_list, puntos_por_imagen)
    print(f"X_w shape: {X_w.shape}")

    x1_no_opt = ut_math.project_points(K, T_wc1, X_w)
    x2_no_opt = ut_math.project_points(K, T_wc2, X_w)
    x3_no_opt = ut_math.project_points(K, T_wc3, X_w)
    x4_no_opt = ut_math.project_points(K, T_wc4, X_w)

    x1_aux = puntos_por_imagen[0]
    x2_aux = puntos_por_imagen[1]
    x3_aux = puntos_por_imagen[2]
    x4_aux = puntos_por_imagen[3]


    cameras = {
        'C1': T_wc1,  
        'C2': T_wc2,
        'C3': T_wc3,
        'C4': T_wc4
    }
    plot_utils.plot3DPoints(X_w, cameras, world_ref=False)
 
    T_opt, X_w_opt = b_adj.run_bundle_adjustment(T_list, K, X_w, [x1_aux, x2_aux, x3_aux, x4_aux])

    T_wc1_opt, T_wc2_opt, T_wc3_opt, T_wc4_opt = T_opt
    x1_p_opt = ut_math.project_points(K, T_wc1_opt, X_w_opt)
    x2_p_opt = ut_math.project_points(K, T_wc2_opt, X_w_opt)
    x3_p_opt = ut_math.project_points(K, T_wc3_opt, X_w_opt)
    x4_p_opt = ut_math.project_points(K, T_wc4_opt, X_w_opt)

    if True:
        image1 = cv2.imread(path_images_order[0])
        image2 = cv2.imread(path_images_order[1])
        image3 = cv2.imread(path_images_order[2])
        image4 = cv2.imread(path_images_order[3])
        plot_utils.visualize_projection_2(image1, puntos_por_imagen[0], x1_no_opt, x1_p_opt, 'Image 1', resize_dim=resize_dim)
        plot_utils.visualize_projection_2(image2, puntos_por_imagen[1], x2_no_opt, x2_p_opt, 'Image 2', resize_dim=resize_dim)
        plot_utils.visualize_projection_2(image3, puntos_por_imagen[2], x3_no_opt, x3_p_opt, 'Image 3', resize_dim=resize_dim)
        plot_utils.visualize_projection_2(image4, puntos_por_imagen[3], x4_no_opt, x4_p_opt, 'Image 4', resize_dim=resize_dim)

            
        cameras = {
            'C1': T_wc1_opt,  
            'C2': T_wc2_opt,
            'C3': T_wc3_opt,
            'C4': T_wc4_opt,
        }
        print(f"X_w_opt: {X_w_opt.shape}")
        plot_utils.plot3DPoints(X_w_opt, cameras, world_ref=False)

        np.savez("data/bundle_adjustment_results_old.npz", 
                cameras={key: T_wc for key, T_wc in cameras.items()},
                points_3D=X_w_opt,
                puntos_0=puntos_por_imagen[0], 
                puntos_1=puntos_por_imagen[1], 
                puntos_2=puntos_por_imagen[2], 
                puntos_3=puntos_por_imagen[3])

        print("Resultados guardados en bundle_adjustment_results_old.npz")



    

    
if __name__ == "__main__":
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)
    parser = argparse.ArgumentParser(description="Ejemplo de script con argumento 'test' desde la línea de comandos.")
    
    # Definir el argumento para 'test', con valor por defecto 0
    parser.add_argument(
        '--test', 
        type=int, 
        default=0, 
        help="Valor de la variable 'test'. Valor por defecto es 0."
    )


    args = parser.parse_args()
    test = args.test

    if test == 0:
        path_new_images='images/new/pilar_*.jpg'
        K, dist, rvecs, tvecs = cam_calib.load_calibration_data("data/camera_calibration.npz")
        path_superglue=['results/pilar_x1_1_pilar_x1_3_matches.npz', 'results/pilar_x1_1_pilar_x1_4_matches.npz', 'results/pilar_x1_1_pilar_x1_5_matches.npz', 'results/pilar_x1_1_old_pilar1936_matches.npz']
        path_images = [("images/new/pilar_x1_1.jpg", "images/new/pilar_x1_3.jpg"), 
                       ("images/new/pilar_x1_1.jpg", "images/new/pilar_x1_4.jpg"),
                       ("images/new/pilar_x1_1.jpg", "images/new/pilar_x1_5.jpg"),
                       ("images/new/pilar_x1_1.jpg", "images/new/old_pilar1936.jpg")]
        resize_dim = (2000, 1126)
    elif test == 1:
        path_new_images='images/prueba/image_*.png'
        path_superglue=['results/image_1_image_2_matches.npz', 'results/image_1_image_3_matches.npz']
        path_images = [("images/prueba/image_1.png", "images/prueba/image_2.png"),
                       ("images/prueba/image_1.png", "images/prueba/image_3.png")]
        K = np.loadtxt("data/K_c.txt")
        resize_dim = None
    else:
        path_new_images='images/prueba/imagen_*.jpeg'
        path_superglue=['results/imagen_1_imagen_2_matches.npz', 'results/imagen_1_imagen_3_matches.npz']
        path_images = [("images/prueba/imagen_1.jpeg", "images/prueba/imagen_2.jpeg"),
                       ("images/prueba/imagen_1.jpeg", "images/prueba/imagen_3.jpeg"),]
        K, dist, rvecs, tvecs = cam_calib.load_calibration_data("data/camera_calibration.npz")
        resize_dim = None
    
    path_new_images = glob.glob(path_new_images)
    path_new_images = sorted(path_new_images)
    # add to path new images
    path_new_images = path_new_images + ['images/new/old_pilar1936.jpg']
    print(path_new_images)

    process_image_sg(path_superglue, path_images, path_new_images, K, resize_dim=resize_dim)
    exit()

    