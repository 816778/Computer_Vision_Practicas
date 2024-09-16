import matplotlib.pyplot as plt
import cv2
import numpy as np

def plotLabeledImagePoints(x, labels, strColor, offset):
    """
    Dibuja los índices de los puntos en una imagen 2D.
    """
    for k in range(x.shape[1]):
        plt.text(x[0, k] + offset[0], x[1, k] + offset[1], labels[k], color=strColor)

def plotNumberedImagePoints(x, strColor, offset):
    """
    Dibuja los números de los puntos en una imagen 2D.
    """
    for k in range(x.shape[1]):
        plt.text(x[0, k] + offset[0], x[1, k] + offset[1], str(k), color=strColor)

def plotLabelled3DPoints(ax, X, labels, strColor, offset):
    """
    Dibuja los índices de los puntos en un gráfico 3D.
    """
    for k in range(X.shape[1]):
        ax.text(X[0, k] + offset[0], X[1, k] + offset[1], X[2, k] + offset[2], labels[k], color=strColor)

def plotNumbered3DPoints(ax, X, strColor, offset):
    """
    Dibuja los números de los puntos en un gráfico 3D.
    """
    for k in range(X.shape[1]):
        ax.text(X[0, k] + offset[0], X[1, k] + offset[1], X[2, k] + offset[2], str(k), color=strColor)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Dibuja un segmento en un gráfico 3D.
    """
    ax.plot([xIni[0], xEnd[0]], [xIni[1], xEnd[1]], [xIni[2], xEnd[2]], strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
    Dibuja un sistema de referencia en un gráfico 3D.
    """
    draw3DLine(ax, T_w_c[0:3, 3], T_w_c[0:3, 3] + T_w_c[0:3, 0], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3], T_w_c[0:3, 3] + T_w_c[0:3, 1], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3], T_w_c[0:3, 3] + T_w_c[0:3, 2], strStyle, 'b', 1)
    ax.text(T_w_c[0, 3] + 0.1, T_w_c[1, 3] + 0.1, T_w_c[2, 3] + 0.1, nameStr)


def createPlot(image_path):
    # Cargar la imagen y convertirla a RGB
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Crear una figura para la imagen
    plt.figure()
    plt.imshow(img)

def plotAndWait(title):

    # Título y visualización
    plt.title(title)
    plt.draw()  # Actualizar la figura
    print('Click en la imagen para continuar...')
    plt.waitforbuttonpress()  # Espera un click para continuar
    plt.close()

def project_and_plot_points(projected_points, labels, image_shape):

    h, w = image_shape[:2]
    # Graficar los puntos proyectados
    plt.plot(projected_points[:, 0], projected_points[:, 1], '+r', markersize=15)
    
    # Etiquetas de los puntos proyectados
    plotLabeledImagePoints(projected_points, labels, 'r', (20, -20))
    plotNumberedImagePoints(projected_points, 'r', (20, 25))


def adjust_limits(projected_points, px, image_shape, verbose=False):
    """
    Ajusta los límites del gráfico si los puntos proyectados o el punto px están fuera de la imagen.

    Parameters
    ----------
    projected_points : numpy array
        Los puntos proyectados en el plano 2D.
    px : tuple or list
        El punto adicional que podría estar fuera de los límites.
    image_shape : tuple
        La forma de la imagen (alto, ancho).
    """

    h, w = image_shape[:2]
    expand_x = False
    expand_y = False
    min_x = None
    max_x = None
    min_y = None
    max_y = None

    if px[0] < 0 or px[0] > w:
        expand_x = True
    if px[1] < 0 or px[1] > h:
        expand_y = True

    # Comprobar si los puntos proyectados están fuera de los límites
    if np.any(projected_points[:, 0] < 0) or np.any(projected_points[:, 0] > w):
        expand_x = True
    if np.any(projected_points[:, 1] < 0) or np.any(projected_points[:, 1] > h):
        expand_y = True
    

    # Ajustar los límites de x
    if expand_x:
        min_x = min(0, np.min(projected_points[:, 0]), px[0] - 50)
        max_x = max(w, np.max(projected_points[:, 0]), px[0] + 50)
        plt.xlim(min_x, max_x)
    else:
        plt.xlim(0, w)

    # Ajustar los límites de y
    if expand_y:
        min_y = min(0, np.min(projected_points[:, 1]), px[1] - 50)
        max_y = max(h, np.max(projected_points[:, 1]), px[1] + 50)
        plt.ylim(min_y, max_y)
    else:
        plt.ylim(0, h)
    
    if verbose:
        print(f"min_x: {min_x}\n max_x: {max_x}\n")
        print(f"min_y: {min_y}\n max_y: {max_y}\n")
    
    # Invertir el eje y para que el origen esté en la esquina superior izquierda
    plt.gca().invert_yaxis()
    

def plot_line_2(px1, px2, color='r'):
    plt.plot([px1[0], px2[0]], [px1[1], px2[1]], color)

def plot_line(line, color='r'):
    """
    Dibuja una línea en el plano 2D.
    
    Parameters
    ----------
    line : numpy array
        Un vector de 3x1 que representa una línea en coordenadas homogéneas.
    color : str
        El color de la línea.
    """
    # Desempaquetar los coeficientes de la línea
    a, b, c = line

    # Calcular los puntos extremos de la línea
    x_vals = np.array([0, 640])
    y_vals = (-c - a * x_vals) / b

    # Dibujar la línea
    plt.plot(x_vals, y_vals, color)


# Plot infinite line that passes through both points
def plot_and_compute_inf_line(px1, px2, color='r'):
    """
    Dibuja una línea infinita entre dos puntos y calcula la ecuación de la recta.
    
    Parámetros:
    p1, p2: Tuplas (x, y)
        Coordenadas de los puntos por los cuales pasa la línea.
    color: str
        Color de la línea.
        
    Retorna:
    tuple (a, b, c)
        Coeficientes de la ecuación ax + by + c = 0.
    """
    x1, y1 = px1
    x2, y2 = px2

    # Calcular los coeficientes de la línea ax + by + c = 0
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    # Dibujo de la línea infinita
    if x1 != x2:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    else:
        slope = None

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # If the line is vertical, plot a vertical line
    if slope is None:
        plt.axvline(x=x1, color=color)
    else:
        # Calculate the y-values of the line at the extremes of the x-limits
        x_vals = np.array(xlim)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, color=color)

    # Mark the points for clarity
    plt.scatter([x1, x2], [y1, y2], color=color, zorder=5)
    plt.xlim(xlim)
    plt.ylim(ylim)


def plot_points(points, color='r', marker='o', label=None):
    plt.plot(points[:, 0], points[:, 1], marker, color=color, label=label)



def plot_point(px, image_shape, color='r'):
    # Dibujar el punto
    plt.scatter([px[0]], [px[1]], color=color)
    


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