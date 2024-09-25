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


if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    T_w_c1 = np.loadtxt(DATA_PATH + 'T_w_c1.txt')
    T_w_c2 = np.loadtxt(DATA_PATH + 'T_w_c2.txt')

    K_c = np.loadtxt(DATA_PATH + 'K_c.txt')
    X_w = np.loadtxt(DATA_PATH + 'X_w.txt')

    x1 = np.loadtxt(DATA_PATH + 'x1Data.txt')
    x2 = np.loadtxt(DATA_PATH + 'x2Data.txt')

    ej_1(x1, x2, K_c, T_w_c1, T_w_c2, X_w)