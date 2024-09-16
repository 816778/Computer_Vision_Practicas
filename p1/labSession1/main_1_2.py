import numpy as np
import utils.plot_utils as plot_utils
import utils.utils as utils
import cv2

FOLDER_DATA = "data/"

def ej1(K_c, R_w_c1, R_w_c2, t_w_c1, t_w_c2, labels=['A', 'B', 'C', 'D', 'E'], verbose=False):
    P1 = utils.get_projection_matrix(K_c, R_w_c1, t_w_c1)
    P2 = utils.get_projection_matrix(K_c, R_w_c2, t_w_c2)

    if verbose:
        print("\nMatriz de proyección P1:")
        print(P1)
        print("\nMatriz de proyección P2:")
        print(P2)

    # Definir puntos 3D
    X_A = np.array([3.44, 0.80, 0.82])
    X_B = np.array([4.20, 0.80, 0.82])
    X_C = np.array([4.20, 0.60, 0.82])
    X_D = np.array([3.55, 0.60, 0.82])
    X_E = np.array([-0.01, 2.60, 1.21])

    points_3D = np.array([X_A, X_B, X_C, X_D, X_E])
    points_3D_hom = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))

    # Proyección de puntos en ambas cámaras
    projections_cam1 = utils.project_points(P1, points_3D_hom)
    projections_cam2 = utils.project_points(P2, points_3D_hom)

    if verbose:
        print("\nProyección cámara 1:")
        print(projections_cam1)
        print("\nProyección cámara 2:")
        print(projections_cam2)
        print()

        plot_utils.createPlot("Image1.jpg")
        plot_utils.project_and_plot_points(projections_cam1, labels)
        plot_utils.plotAndWait("Image 1")

        plot_utils.createPlot("Image2.jpg")
        plot_utils.project_and_plot_points(projections_cam2, labels)
        plot_utils.plotAndWait("Image 2")

    return P1, P2, points_3D_hom, projections_cam1, projections_cam2




def ej2(image_path, P, g_points, projections_cam, labels=['A', 'B', 'C', 'D', 'E']):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_shape = img.shape


    plot_utils.createPlot(image_path)
    # Mostrar proyecciones en las imágenes
    plot_utils.project_and_plot_points(projections_cam, labels, image_shape)

    # Proyectar y calcular l_ab
    l_ab = utils.compute_line(projections_cam[0], projections_cam[1])
    plot_utils.plot_line(l_ab, 'g')
    # Proyectar  y calcular l_cd
    l_cd = utils.compute_line(projections_cam[2], projections_cam[3])
    plot_utils.plot_line(l_cd, 'g')

    # Calcular y graficar la intersección de las líneas AB y CD
    intersection = utils.compute_intersection(l_ab, l_cd)
    plot_utils.plot_point(intersection, image_shape, 'b')

    # AB_inf = B-A Vanishing Point
    v_ab = g_points[1] - g_points[0]
    v_ab[2] = 0 # porque es punto en infinito
    AB_inf = utils.project_points(P, v_ab, return_homogeneous=True)
    plot_utils.plot_point(AB_inf, image_shape, 'b')
    
    plot_utils.adjust_limits(projections_cam, intersection, image_shape)

    plot_utils.plotAndWait("Image")
    

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    # Cargar datos de archivo
    R_w_c1 = np.loadtxt(FOLDER_DATA + 'R_w_c1.txt')
    R_w_c2 = np.loadtxt(FOLDER_DATA + 'R_w_c2.txt')
    t_w_c1 = np.loadtxt(FOLDER_DATA + 't_w_c1.txt')
    t_w_c2 = np.loadtxt(FOLDER_DATA + 't_w_c2.txt')
    K_c = np.loadtxt(FOLDER_DATA + 'K.txt')

    # Ejecutar el ejercicio 1
    P1, P2, points_3D_hom, projections_cam1, projections_cam2 = ej1(K_c, R_w_c1, R_w_c2, t_w_c1, t_w_c2, verbose=False)

    # Ejecutar ejercicio 2
    ej2("Image1.jpg", P1, points_3D_hom, projections_cam1)
    ej2("Image2.jpg", P2, points_3D_hom, projections_cam2)
