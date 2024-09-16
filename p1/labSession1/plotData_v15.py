#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 1
#
# Title: 2D-3D geometry in homogeneous coordinates and camera projection
#
# Date: 5 September 2024
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.5
#
#####################################################################################

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import cv2



# Ensamble T matrix
def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c


def plotLabeledImagePoints(x, labels, strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], labels[k], color=strColor)


def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], str(k), color=strColor)


def plotLabelled3DPoints(ax, X, labels, strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], labels[k], color=strColor)

def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], str(k), color=strColor)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)

# Función para ensamblar la matriz de proyección P
def get_projection_matrix(K, R, t):
    """Calcula la matriz de proyección P = K * [R | t]"""
    T = ensamble_T(R, t)
    P = K @ T[:3, :]  # Matriz de proyección
    return P


def project_points(P, points_3D_hom):
    projected_points_hom = P @ points_3D_hom.T  # Proyección
    print("\nprojected points homogeneous\n", projected_points_hom)
    # Convertir de homogéneo a cartesiano
    projected_points_2D = projected_points_hom[:2] / projected_points_hom[2]
    return projected_points_2D.T


def project_and_plot_points(image_path, projected_points, labels, title):
    # Cargar la imagen y convertirla a RGB
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    # Crear una figura para la imagen
    plt.figure()
    plt.imshow(img)
    
    # Graficar los puntos proyectados
    plt.plot(projected_points[:, 0], projected_points[:, 1], '+r', markersize=15)
    
    # Etiquetas de los puntos proyectados (opcional)
    plotLabeledImagePoints(projected_points, labels, 'r', (20, -20))  # Opción con etiquetas
    plotNumberedImagePoints(projected_points, 'r', (20, 25))  # Opción con números
    
    # Título y visualización
    plt.title(title)
    plt.draw()  # Actualizar la figura
    print('Click en la imagen para continuar...')
    plt.waitforbuttonpress()  # Espera un click para continuar
    plt.close()


def ej1(K_c, R_w_c1, R_w_c2, t_w_c2, t_w_c1, labels=['A', 'B', 'C', 'D', 'E']):
    P1 = get_projection_matrix(K_c, R_w_c1, t_w_c1)
    P2 = get_projection_matrix(K_c, R_w_c2, t_w_c2)

    print("Matriz de proyección P1:")
    print(P1)
    print("\nMatriz de proyección P2:")
    print(P2)


    X_A = np.array([3.44, 0.80, 0.82])
    X_B = np.array([4.20, 0.80, 0.82])
    X_C = np.array([4.20, 0.60, 0.82])
    X_D = np.array([3.55, 0.60, 0.82])
    X_E = np.array([-0.01, 2.60, 1.21])

    points_3D = np.array([X_A, X_B, X_C, X_D, X_E])
    points_3D_hom = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))

    projections_cam1 = project_points(P1, points_3D_hom)
    projections_cam2 = project_points(P2, points_3D_hom)

    project_and_plot_points("Image1.jpg", projections_cam1, labels, 'Image 1')
    project_and_plot_points("Image2.jpg", projections_cam2, labels, 'Image 2')

    return P1, P2, projections_cam1, projections_cam2


def plot_line_in_image(img, line, color='r'):
    """Dibuja una línea homogénea en la imagen, basada en los bordes de la imagen"""
    h, w, _ = img.shape
    
    # Para calcular dos puntos extremos en la imagen (en los bordes)
    # Puntos en los bordes izquierdo y derecho
    x_left = 0
    y_left = -line[2] / line[1]  # Resolver para y cuando x = 0
    x_right = w
    y_right = -(line[2] + line[0] * x_right) / line[1]  # Resolver para y cuando x = ancho imagen

    # Asegurarse de que los puntos están dentro de los límites de la imagen
    if y_left < 0 or y_left > h:
        y_left = 0 if y_left < 0 else h
    if y_right < 0 or y_right > h:
        y_right = 0 if y_right < 0 else h

    # Dibuja la línea entre los puntos en los bordes
    plt.plot([x_left, x_right], [y_left, y_right], color)

def plot_intersection_point(p, color='g'):
    """Dibuja el punto de intersección en la imagen"""
    plt.plot(p[0], p[1], color + 'o', markersize=10)


def ej2(projections_cam, A, B, P, image_path, labels=['A', 'B', 'C', 'D', 'E']):

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Calcular y graficar la línea que pasa por A y B en la imagen 1
    # l_ab = a×b
    l_ab = np.cross(projections_cam[0], projections_cam[1]) 
    # l_cd = c×d
    l_cd = np.cross(projections_cam[2], projections_cam[3]) 
    # p_12 = l_ab × l_cd
    p_intersection = np.cross(l_ab, l_cd)
    p_intersection = p_intersection / p_intersection[2]
    # AB_inf = B-A
    direction = B - A
    direction /= np.linalg.norm(direction)
    AB_inf = np.append(direction, 0)
    # Proyectar P×AB_inf
    projected_point = P @ AB_inf
    ab_inf = projected_point[:2] / projected_point[2]

    plt.figure()
    plt.imshow(img)

    # Graficar las líneas l_ab y l_cd
    plot_line_in_image(img, l_ab, color='r')
    plot_line_in_image(img, l_cd, color='b')

    # Graficar el punto de intersección p_12
    plot_intersection_point(p_intersection, color='g')

    # Graficar el punto de fuga ab_inf
    plot_intersection_point(ab_inf, color='m')

    # Mostrar etiquetas de los puntos proyectados
    plotLabeledImagePoints(projections_cam.T, labels, 'r', (10, -10))

    # Mostrar la imagen
    plt.title('Líneas y punto de intersección en la imagen')
    plt.show()

if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    R_w_c1 = np.loadtxt('data/R_w_c1.txt')
    R_w_c2 = np.loadtxt('data/R_w_c2.txt')

    t_w_c1 = np.loadtxt('data/t_w_c1.txt')
    t_w_c2 = np.loadtxt('data/t_w_c2.txt')

    K_c = np.loadtxt('data/K.txt')

    P1, P2, projections_cam1, projections_cam2 = ej1(K_c, R_w_c1, R_w_c2, t_w_c2, t_w_c1)
    
    A = np.array([3.44, 0.80, 0.82, 1.0])  # Punto A en 3D
    B = np.array([4.20, 0.80, 0.82, 1.0])

    # ej2(projections_cam1, A, B, P1, 'Image1.jpg')

    exit()
    
    """
    print(np.array([[3.44, 0.80, 0.82]]).T) #transpose need to have dimension 2
    print(np.array([3.44, 0.80, 0.82]).T) #transpose does not work with 1 dim arrays

    # Example of transpose (need to have dimension 2)  and concatenation in numpy
    X_w = np.vstack((np.hstack((np.reshape(X_A,(3,1)), np.reshape(X_C,(3,1)))), np.ones((1, 2))))

    ##Plot the 3D cameras and the 3D points
    fig3D = plt.figure(3)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')

    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')
    plotNumbered3DPoints(ax, X_w, 'r', (0.1, 0.1, 0.1)) # For plotting with numbers (choose one of the both options)
    plotLabelled3DPoints(ax, X_w, ['A','C'], 'r', (-0.3, -0.3, 0.1)) # For plotting with labels (choose one of the both options)

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')

    #Drawing a 3D segment
    draw3DLine(ax, X_A, X_C, '--', 'k', 1)

    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()

    ## 2D plotting example
    img1 = cv2.cvtColor(cv2.imread("Image1.jpg"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("Image2.jpg"), cv2.COLOR_BGR2RGB)

    x1 = np.array([[527.7253,334.1983],[292.9017,392.1474]])

    plt.figure(1)
    plt.imshow(img1)
    plt.plot(x1[0, :], x1[1, :],'+r', markersize=15)
    plotLabeledImagePoints(x1, ['a','c'], 'r', (20,-20)) # For plotting with labels (choose one of the both options)
    plotNumberedImagePoints(x1, 'r', (20,25)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 1')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    """
