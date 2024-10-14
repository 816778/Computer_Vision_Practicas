import numpy as np
import cv2
import utils.plot_utils as plot_utils
import utils.utils as utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_PATH = 'data/'
IMAGE_PATH = 'images/'


def ej_1(x1, x2, K_c, T_w_c1, T_w_c2, X_w, verbose=False):
    """
    Triangular puntos 3D a partir de coincidencias en dos cámaras localizadas y calibradas.
    Dado un punto que aparece en dos imágenes diferentes tomadas desde distintas posiciones 
    por una cámara (o dos cámaras calibradas), podemos usar estas dos proyecciones para calcular
    la posición 3D de dicho punto en el espacio. 
    """
    # Convertir los puntos 2D en formato homogéneo
    x1_h = np.vstack((x1, np.ones((1, x1.shape[1])))) 
    x2_h = np.vstack((x2, np.ones((1, x2.shape[1]))))

    # Calcular las matrices de proyección
    P1 = utils.compute_projection_matrix(K_c, T_w_c1)
    P2 = utils.compute_projection_matrix(K_c, T_w_c2)

    # Triangular los puntos
    # Encontrar el punto 3D donde se intersectan los dos rayos
    X_h = utils.triangulate_points(P1, P2, x1_h, x2_h)

    if verbose:
        fig3D = plt.figure(3)
        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Dibujar los sistemas de referencia de las cámaras y del mundo
        plot_utils.drawRefSystem(ax, np.eye(4, 4), '-', 'W')
        plot_utils.drawRefSystem(ax, T_w_c1, '-', 'C1')
        plot_utils.drawRefSystem(ax, T_w_c2, '-', 'C2')

        # Dibujar los puntos del mundo reales
        # ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='o', label='Puntos Reales')

        # Dibujar los puntos triangulados
        ax.scatter(X_h[0, :], X_h[1, :], X_h[2, :], marker='x', c='b', label='Puntos Triangulados')
        ax.legend()
        xFakeBoundingBox = np.linspace(0, 4, 2)
        yFakeBoundingBox = np.linspace(0, 4, 2)
        zFakeBoundingBox = np.linspace(0, 4, 2)
        plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
        plt.show()    
    
    return X_h


def ej2_1(F):
    """
    APARTADO 2_1: Epipolar lines visualization
    """
    # seleccionar un punto en la primera imagen
    p1 = np.array([297, 307, 1]) 

    # Calcular y visualizar la línea epipolar correspondiente en la imagen 2
    img1 = cv2.cvtColor(cv2.imread(IMAGE_PATH + 'image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(IMAGE_PATH + 'image2.png'), cv2.COLOR_BGR2RGB)
    # l2​ = F⋅p1
    l2 = np.dot(F, p1)

    plot_utils.createPlot(IMAGE_PATH + 'image1.png')
    x_coords = [p1[0]] 
    y_coords = [p1[1]]
    labels = ['P1']  
    plot_utils.show_points_on_image(x_coords, y_coords, labels)
    plot_utils.plotAndWait()

    #Encontrar puntos  extremos de la línea epipolar
    plot_utils.createPlot(IMAGE_PATH + 'image2.png')
    y_1 = -(l2[0] * 0 + l2[2]) / l2[1] # -(a * x + c) / b Encontrar y cuando x = 0
    x_2 = -(l2[1] * 0 + l2[2]) / l2[0] # -(b * y + c) / a Encontrar x cuando y = 0
    x_coords = [0, x_2]
    y_coords = [y_1, 0]
    labels = ['P1', 'P2']  
    plot_utils.show_points_and_line(x_coords, y_coords, labels)
    plot_utils.plotAndWait()
 

def ej2_2(T_w_c1, T_w_c2, K_c, x1):
    """
    APARTADO 2.2: Fundamental matrix definition"
    """
    # l2​ = F⋅p1
    # Matriz fundamental F: p2^T⋅F⋅p1 = 0
    # p1 punto coordenadas homogéneas primera imagen
    # p2 punto coordenadas homogéneas segunda imagen
    """
    F: e puede calcular a partir de las posiciones de las cámaras.
    Se utiliza la matriz de esencialidad E
    E = [t]R
    F = K2^-T*E*K1^-1
    """
    F = utils.calculate_fundamental_matrix(T_w_c1, T_w_c2, K_c)
    img2 = cv2.cvtColor(cv2.imread(IMAGE_PATH + 'image2.png'), cv2.COLOR_BGR2RGB)
    plot_utils.createPlot(IMAGE_PATH + 'image1.png')
    
    x_coords = [368, 101, 200, 500, 363]
    y_coords = [390, 452, 247, 370, 112]
    
    assert len(x_coords) == len(y_coords), "Las longitudes de x_coords y y_coords no coinciden"
    labels = [str(i+1) for i in range(len(x_coords))]
    plot_utils.show_points_on_image(x_coords, y_coords, labels)
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    x1 = np.vstack((x_coords, y_coords))
    plot_utils.plot_epipolar_lines(F, x1, img2, num_lines=7)
    return F


def ej2_3(x1, x2):
    """
    APARTADO 2.3: Fundamental matrix linear estimation with eight point solution.
    """
    x1_h = np.vstack((x1, np.ones(x1.shape[1])))
    x2_h = np.vstack((x2, np.ones(x2.shape[1])))
    F_est = utils.estimate_fundamental_8point(x1_h, x2_h)
    print("Matriz Fundamental estimada (F): \n", F_est)

    img2 = cv2.cvtColor(cv2.imread(IMAGE_PATH + 'image2.png'), cv2.COLOR_BGR2RGB)
    plot_utils.createPlot(IMAGE_PATH + 'image1.png')
    
    x_coords = [368, 101, 200, 500, 363]
    y_coords = [390, 452, 247, 370, 112]
    
    assert len(x_coords) == len(y_coords), "Las longitudes de x_coords y y_coords no coinciden"
    labels = [str(i+1) for i in range(len(x_coords))]
    plot_utils.show_points_on_image(x_coords, y_coords, labels)
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    x1 = np.vstack((x_coords, y_coords))
    plot_utils.plot_epipolar_lines(F, x1, img2, num_lines=7)
    return F


def ej2_4_1(P2, P1, x1, x2, verbose=True):
    """
    APARTADO 2.4.1: Fundamental matrix definition"
    """
    # l2​ = F⋅p1
    # Matriz fundamental F: p2^T⋅F⋅p1 = 0
    # p1 punto coordenadas homogéneas primera imagen
    # p2 punto coordenadas homogéneas segunda imagen
    """
    F: e puede calcular a partir de las posiciones de las cámaras.
    Se utiliza la matriz de esencialidad E
    E = [t]R
    F = K2^-T*E*K1^-1
    """
    x1_h = np.vstack((x1, np.ones((1, x1.shape[1])))) 
    x2_h = np.vstack((x2, np.ones((1, x2.shape[1]))))

    # Triangular los puntos
    # Encontrar el punto 3D donde se intersectan los dos rayos
    X_h = utils.triangulate_points(P1, P2, x1_h, x2_h)

    # print("Debug: X_h", X_h, "P1:", P1, "P2:", P2)

    if verbose:
        fig3D = plt.figure(3)
        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Dibujar los sistemas de referencia de las cámaras y del mundo
        plot_utils.drawRefSystem(ax, np.eye(4, 4), '-', 'W')
        plot_utils.drawRefSystem(ax, T_w_c1, '-', 'C1')
        plot_utils.drawRefSystem(ax, T_w_c2, '-', 'C2')

        # Dibujar los puntos del mundo reales
        # ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='o', label='Puntos Reales')

        # Dibujar los puntos triangulados
        ax.scatter(X_h[0, :], X_h[1, :], X_h[2, :], marker='x', c='b', label='Puntos Triangulados')
        ax.legend()
        xFakeBoundingBox = np.linspace(0, 4, 2)
        yFakeBoundingBox = np.linspace(0, 4, 2)
        zFakeBoundingBox = np.linspace(0, 4, 2)
        plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
        plt.show()    
    
    return X_h


def ej2_4(P1, F, K1, K2, x1, x2, t_w_c1):
    """
    APARTADO 2.4: Pose estimation from two views
    """
    # Calcula la matriz esencial E a partir de la matriz fundamental F y las matrices intrínsecas K1 y K2.
    E_21 = K2.T @ F @ K1
    #E_inv = np.linalg.inv(E)
    
    print("Matriz esencial E:\n", E_21)

    # Descomponer la matriz esencial E en 4 posibles soluciones
    R1_21, R2_21, t_21, t_neg = utils.decompose_essential_matrix(E_21)
    #R1_1, R2_1, t_1, t_neg = utils.decompose_essential_matrix(E_inv)

    # Seleccionar la solución correcta triangulando los puntos 3D
    P2 = utils.select_correct_pose(P1, R1_21, R2_21, t_w_c1, K2, x1, x2)
    # P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))

    print("Matriz de proyección correcta para la segunda cámara:\n", P1)
    print("Matriz de proyección correcta para la segunda cámara:\n", P2)
    return P2


def ej2_5(X_triangulated, X_w, T_w_c1, T_w_c2):
    """
    APARTADO 2.5: Results presentation
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_triangulated[0, :], X_triangulated[1, :], X_triangulated[2, :], c='b', label='Puntos Triangulados')
    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], c='r', marker='x', label='Ground Truth')

    plot_utils.draw_camera(ax, T_w_c1, name='Cámara 1')
    plot_utils.draw_camera(ax, T_w_c2, name='Cámara 2')

    # Etiquetas y leyenda
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    rmse = utils.compute_rmse(X_triangulated, X_w[:3])
    print("RMSE:", rmse)


def ej3_1(K1, K2, T_w_c1, T_w_c2, ec_plane):
    """
    APARTADO 3.1: Homography definition
    """
    n = ec_plane[:3].reshape(3, 1)
    d = ec_plane[3] # d = abs(d) / np.linalg.norm(n)

    T_c2_c1 = np.linalg.inv(T_w_c2) @ T_w_c1

    t_c2_c1 = T_c2_c1[:3, 3].reshape(3, 1)
    R_c2_c1 = T_c2_c1[:3, :3]

    # Calcular la Homografía: H = K2 * (R_c2_c1 - (t_c2_c1 * n^T) / d) * K1^-1
    H = K2 @ (R_c2_c1 - (t_c2_c1 @ n.T) / d) @ np.linalg.inv(K1)

    return H


def ej3_2(H, x1):
    """
    APARTADO 3.2: Point transfer visualization
    """

    #H_inv = np.linalg.inv(H)

    # Convertir los puntos en coordenadas homogéneas
    x2_h = H @ x1
    x2 = x2_h[:2] / x2_h[2]

    img1 = cv2.cvtColor(cv2.imread(IMAGE_PATH + 'image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(IMAGE_PATH + 'image2.png'), cv2.COLOR_BGR2RGB)

    plt.figure(1)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.plot(x1[0, :], x1[1, :], 'bx', markersize=10, label='Proyección real')
    plt.title('Image 1 - Puntos en el suelo')
    plt.draw()

    plt.figure(2)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.plot(x2[0, :], x2[1, :], 'rx', markersize=10, label='Proyección estimada')
    plt.title('Image 2 - Proyección de puntos')
    plt.legend()
    plt.show()

    return x2


def ej3_3(x1, x2):
    """
    APARTADO 3.3: Point transfer visualization
    """
    H = utils.compute_homography(x1,x2)
    img1 = cv2.cvtColor(cv2.imread(IMAGE_PATH + 'image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(IMAGE_PATH + 'image2.png'), cv2.COLOR_BGR2RGB)

    plt.figure(1)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.plot(x1[0, :], x1[1, :], 'bx', markersize=10, label='Proyección real')
    plt.title('Image 1 - Puntos en el suelo')
    plt.draw()

    plt.figure(2)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.plot(x2[0, :], x2[1, :], 'rx', markersize=10, label='Proyección estimada')
    plt.title('Image 2 - Proyección de puntos')
    plt.legend()
    plt.show()

    return H

if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    T_w_c1 = np.loadtxt(DATA_PATH + 'T_w_c1.txt')
    T_w_c2 = np.loadtxt(DATA_PATH + 'T_w_c2.txt')

    K_c = np.loadtxt(DATA_PATH + 'K_c.txt')
    X_w = np.loadtxt(DATA_PATH + 'X_w.txt')

    x1 = np.loadtxt(DATA_PATH + 'x1Data.txt')
    x2 = np.loadtxt(DATA_PATH + 'x2Data.txt')

    F = np.loadtxt(DATA_PATH + 'F_21_test.txt')

    P1 = utils.compute_projection_matrix(K_c, T_w_c1)
    P2 = ej2_4(P1, F, K_c, K_c, x1, x2, T_w_c1)
    X_triangulated = ej2_4_1(P2, P1, x1, x2, verbose=True)
    ej2_5(X_triangulated, X_w, T_w_c1, T_w_c2)


