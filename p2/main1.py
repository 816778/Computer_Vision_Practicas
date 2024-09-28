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
    x1_h = np.vstack((x1, np.ones((1, x1.shape[1]))))  # Añadimos una fila de unos (puntos homogéneos)
    x2_h = np.vstack((x2, np.ones((1, x2.shape[1]))))

    # Calcular las matrices de proyección
    P1 = utils.compute_projection_matrix(K_c, T_w_c1)
    P2 = utils.compute_projection_matrix(K_c, T_w_c2)

    # Triangular los puntos
    # Encontrar el punto 3D donde se intersectan los dos rayos
    X_h = cv2.triangulatePoints(P1, P2, x1_h[:2], x2_h[:2])

    # Convertir los puntos homogéneos a coordenadas 3D
    X = X_h[:3] / X_h[3]  # Dividimos por la cuarta coordenada

    if verbose:
        print("Puntos 3D triangulados: \n", X)
    
    img1 = cv2.cvtColor(cv2.imread(IMAGE_PATH + 'image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(IMAGE_PATH + 'image2.png'), cv2.COLOR_BGR2RGB)

    plot_utils.createPlot()
    plot_utils.points_3d(X, X_w)
    plot_utils.plotAndClose()


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

    print(l2)

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
    print("Matriz Fundamental (F): \n", F)
    img1 = cv2.cvtColor(cv2.imread(IMAGE_PATH + 'image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(IMAGE_PATH + 'image2.png'), cv2.COLOR_BGR2RGB)

    # Dibujar las líneas epipolares en la imagen 2
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



def ej2_4():
    exit()



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
    ej2_4(x1, x2)
