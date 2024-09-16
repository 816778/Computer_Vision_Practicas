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

def project_and_plot_points(projected_points, labels):

    # Graficar los puntos proyectados
    plt.plot(projected_points[:, 0], projected_points[:, 1], '+r', markersize=15)
    
    # Etiquetas de los puntos proyectados
    plotLabeledImagePoints(projected_points, labels, 'r', (20, -20))
    plotNumberedImagePoints(projected_points, 'r', (20, 25))
    

def plot_line(px1, px2, color='r'):
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
def plot_inf_line(px1, px2, color='r'):
    """
    Plots an infinite line passing through two given pixel coordinates.
    
    Parameters:
    coord1: tuple of ints (x1, y1)
        The first pixel coordinate.
    coord2: tuple of ints (x2, y2)
        The second pixel coordinate.
    """
    x1, y1 = px1
    x2, y2 = px2

    # Calculate the slope (m) of the line
    if x1 != x2:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    else:
        # Vertical line case
        slope = None

    # Get the current axis limits
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

        # Extend the line only within the current y-limits
        plt.plot(x_vals, y_vals, color=color)

    # Mark the points for clarity
    plt.scatter([x1, x2], [y1, y2], color=color, zorder=5)
    plt.xlim(xlim)
    plt.ylim(ylim)


def plot_points(points, color='r', marker='o', label=None):
    plt.plot(points[:, 0], points[:, 1], marker, color=color, label=label)

def plot_point(px, color='r', marker='o', label=None):
    plt.plot(px[0], px[1], marker, color=color, label=label)

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