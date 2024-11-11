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


def plotResidual(x,xProjected,strStyle):
    """
        Plot the residual between an image point and an estimation based on a projection model.
         -input:
             x: Image points.
             xProjected: Projected points.
             strStyle: Line style.
         -output: None
         """

    for k in range(x.shape[1]):
        plt.plot([x[0, k], xProjected[0, k]], [x[1, k], xProjected[1, k]], strStyle)

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
        plt.text(x[0, k]+offset, x[1, k]+offset, str(k), color=strColor)

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
        ax.text(X[0, k]+offset, X[1, k]+offset, X[2,k]+offset, str(k), color=strColor)

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


def project_points_plot(image, xData, xProj, title):
    plt.figure()
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plotResidual(xData, xProj, 'k-')
    plt.plot(xProj[0, :], xProj[1, :], 'bo')
    plt.plot(xData[0, :], xData[1, :], 'rx')
    plotNumberedImagePoints(xData[0:2, :], 'r', 4)
    plt.title(title)
    print('Close the figures to continue.')
    plt.show()


def visualize_projection(image, xData, xProj, title):
    plt.figure()
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plotResidual(xData, xProj, 'k-')
    plt.plot(xProj[0, :], xProj[1, :], 'bo')
    plt.plot(xData[0, :], xData[1, :], 'rx')
    plotNumberedImagePoints(xData[0:2, :], 'r', 4)
    plt.title(title)
    print('Close the figures to continue.')
    plt.show()
    

##############################################################################
def visualize_projection_2(image, xData, xProj_no_opt, xProj_opt, title):
    """
    Visualiza la proyección de los puntos iniciales (no optimizados) y optimizados,
    junto con sus errores en el mismo gráfico.
    
    Args:
    - image: Imagen de fondo.
    - xData: Puntos 2D observados en la imagen (ground truth).
    - xProj_no_opt: Puntos proyectados antes de la optimización.
    - xProj_opt: Puntos proyectados después de la optimización.
    - title: Título del gráfico.
    """
    plt.figure()
    plt.subplots_adjust(
        top=0.97,     
        bottom=0.055,  
        left=0.125,    
        right=0.9,  
    )
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)

    plotResidual(xData, xProj_no_opt, 'g-')
    plotResidual(xData, xProj_opt, 'k-')

    plt.plot(xProj_no_opt[0, :], xProj_no_opt[1, :], 'go', label="No Optimizado")

    plt.plot(xProj_opt[0, :], xProj_opt[1, :], 'bo', label="Optimizado")

    plt.plot(xData[0, :], xData[1, :], 'rx', label="Puntos Observados")

    plotNumberedImagePoints(xData[0:2, :], 'r', 4)

    plt.title(title)
    plt.legend()
    print('Close the figures to continue.')
    plt.show()


def plot_epipolar_lines(F, x1, img2, title='Epipolar lines'):
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

    plt.title(title)
    plt.show()