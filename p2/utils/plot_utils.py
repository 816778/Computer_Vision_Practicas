import matplotlib.pyplot as plt
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def createPlot(image_path=None):
    # Cargar la imagen y convertirla a RGB
    if image_path is not None:
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Crear una figura para la imagen
    plt.figure()
    if image_path is not None:
        plt.imshow(img)
        plt.axis('on')

def plotAndWait(title='Grafico'):

    # Título y visualización
    plt.title(title)
    plt.draw()  # Actualizar la figura
    print('Click en la imagen para continuar...')
    plt.waitforbuttonpress()  # Espera un click para continuar
    plt.close()

def plotAndClose(title='Grafico'):

    # Título y visualización
    plt.title(title)
    plt.draw() 
    plt.show()
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.close()

"""
FUNCIONES PARA LOS EJERCICIOS 1 Y 2
"""
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

def drawRefSystem(ax, T_w_c, strStyle, nameStr, scale=1.0):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + scale * T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + scale * T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + scale * T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze(T_w_c[0, 3]+0.1), np.squeeze(T_w_c[1, 3]+0.1), np.squeeze(T_w_c[2, 3]+0.1), nameStr)



def points_3d(X, X_w=None):
    # Visualizar los puntos 3D triangulados en un gráfico 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar los puntos 3D
    ax.scatter(X[0, :], X[1, :], X[2, :], c='b', marker='o', label='Puntos triangulados')
    
    if X_w is not None:
        ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], c='r', marker='^', label='Puntos originales (X_w)')

    # Etiquetas de los ejes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Puntos 3D Triangulados')
    
    # Mostrar leyenda
    ax.legend()
    
def draw_epipolar_line(img, line, color='r'):
    """
    Función para dibujar la línea epipolar en una imagen usando Matplotlib.
    img: Imagen sobre la que se dibuja.
    line: Parámetros de la línea epipolar [a, b, c], con la ecuación ax + by + c = 0.
    """
    h, w, _ = img.shape
    # Calculamos los puntos de intersección de la línea con los bordes de la imagen
    x0, y0 = 0, int(-line[2] / line[1])  # Intersección con el borde izquierdo (x = 0)
    x1, y1 = w, int(-(line[2] + line[0] * w) / line[1])  # Intersección con el borde derecho (x = w)

    # Limitar los valores dentro de los límites de la imagen
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    
    # Dibujar la línea en matplotlib
    plt.plot([x0, x1], [y0, y1], color=color)


def show_points_on_image(x_coords, y_coords, labels=None):
    """
    Muestra los puntos en la imagen y los etiqueta.
    
    Args:
        image_path (str): Ruta de la imagen donde se mostrarán los puntos.
        points (list or array): Lista o array de puntos a mostrar (cada punto debe ser un array [x, y]).
        labels (list, optional): Lista de etiquetas para los puntos. Si no se proporciona, no se etiquetan.
    """     
    # Dibujar los puntos en la imagen
    plt.scatter(x_coords, y_coords, color='yellow', s=100, marker='x', label='Puntos')
    
    # Etiquetar los puntos si se proporcionan etiquetas
    if labels is not None:
        for i, label in enumerate(labels):
            plt.text(x_coords[i] + 10, y_coords[i] - 10, label, color='red', fontsize=12)
    
    # Mostrar el resultado
    plt.title('Puntos proyectados en la imagen')
    plt.axis('off')  # Ocultar los ejes
    plt.show()

def show_points_and_line(x_coords, y_coords, labels=None):
    """
    Dibuja puntos y una línea entre ellos sobre la imagen.
    
    Args:
        x_coords (list or array): Lista con las coordenadas X de los puntos.
        y_coords (list or array): Lista con las coordenadas Y de los puntos.
        labels (list, optional): Lista de etiquetas para los puntos.
    """
    # Dibujar los puntos en la imagen
    plt.scatter(x_coords, y_coords, color='yellow', s=100, marker='x')

    # Etiquetar los puntos si se proporcionan etiquetas
    if labels is not None:
        for i, label in enumerate(labels):
            plt.text(x_coords[i] + 10, y_coords[i] - 10, label, color='red', fontsize=12)

    # Dibujar la línea que conecta los puntos
    plt.plot(x_coords, y_coords, color='blue', linewidth=2)



def plot_epipolar_lines(F, x1, img2, num_lines=5):
    """
    Dibuja las líneas epipolares en la segunda imagen a partir de los puntos en la primera imagen.
    """
    plt.figure()
    plt.imshow(img2, cmap='gray')

    for i in range(len(x1[0])):
        p1 = np.append(x1[:, i], 1)  # Punto en coordenadas homogéneas
        l2 = F @ p1  # Línea epipolar en la segunda imagen

        # Dibujar la línea epipolar
        x_vals = np.array([0, img2.shape[1]])  # Límites de la imagen
        y_vals = -(l2[0] * x_vals + l2[2]) / l2[1]
        plt.plot(x_vals, y_vals, color='blue')

    plt.title('Líneas Epipolares en la Imagen 2')
    plt.show()


def draw_camera(ax, T, name='Camera'):
    """
    Dibuja la cámara en la visualización 3D.
    """
    # La posición de la cámara en coordenadas del mundo es la columna de traslación de T
    camera_position = T[:3, 3]

    # Dibujar el punto de la cámara
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='g', label=name, s=100)

    # Dibujar los ejes de la cámara
    scale = 1 # Escala para los ejes
    x_axis = camera_position + scale * T[:3, 0]
    y_axis = camera_position + scale * T[:3, 1]
    z_axis = camera_position + scale * T[:3, 2]

    # Dibujar líneas para los ejes
    ax.plot([camera_position[0], x_axis[0]], [camera_position[1], x_axis[1]], [camera_position[2], x_axis[2]], color='r', label=name + ' X-axis')
    ax.plot([camera_position[0], y_axis[0]], [camera_position[1], y_axis[1]], [camera_position[2], y_axis[2]], color='g', label=name + ' Y-axis')
    ax.plot([camera_position[0], z_axis[0]], [camera_position[1], z_axis[1]], [camera_position[2], z_axis[2]], color='b', label=name + ' Z-axis')


def capture_points(img):

    points = []

    # Mouse event callback function
    def mouse_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Clicked at: ", x, y)

            points.append((x, y))

            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", img)

    # Set the mouse callback function
    cv2.setMouseCallback("Image", mouse_event)

    # Wait for the user to click on the image
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array(points)