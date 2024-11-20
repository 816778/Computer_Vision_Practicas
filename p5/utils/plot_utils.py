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


def computeRMSE(xData, xProj):
    """
    Compute the Root Mean Square Error between the data and the projection.
    -input:
        xData: Data points.
        xProj: Projected points.
    -output:
        rmse: Root Mean Square Error.
    """
    return np.sqrt(np.mean(np.sum((xData-xProj)**2, axis=0)))

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


def visualize_triangulation_results(fisheye1_image, fisheye2_image, points_2d_left, points_2d_right):
    """
    Visualiza los puntos 3D proyectados en ambas imágenes.
    
    Parámetros:
        fisheye1_image: Imagen de la cámara izquierda
        fisheye2_image: Imagen de la cámara derecha
        points_3d: Puntos 3D triangulados (3, N)
        K_1: Matriz intrínseca de la cámara izquierda (3, 3)
        K_2: Matriz intrínseca de la cámara derecha (3, 3)
        T_wc1: Matriz de transformación de la cámara izquierda (4, 4)
        T_wc2: Matriz de transformación de la cámara derecha (4, 4)
    """
    # Visualización en la imagen de la cámara izquierda
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(cv2.cvtColor(fisheye1_image, cv2.COLOR_BGR2RGB))
    ax1.scatter(points_2d_left[0, :], points_2d_left[1, :], color='red', s=10, label='Proyección 3D')
    ax1.set_title("Proyección de Puntos 3D en la Imagen de la Cámara Izquierda")
    ax1.legend()

    # Visualización en la imagen de la cámara derecha
    ax2.imshow(cv2.cvtColor(fisheye2_image, cv2.COLOR_BGR2RGB))
    ax2.scatter(points_2d_right[0, :], points_2d_right[1, :], color='blue', s=10, label='Proyección 3D')
    ax2.set_title("Proyección de Puntos 3D en la Imagen de la Cámara Derecha")
    ax2.legend()

    plt.show()



def visualize_projection(image, xData, xProj, title, block=True):
    plt.figure()
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plotResidual(xData, xProj, 'k-')
    plt.plot(xProj[0, :], xProj[1, :], 'bo')
    plt.plot(xData[0, :], xData[1, :], 'rx')
    plotNumberedImagePoints(xData[0:2, :], 'r', 4)
    plt.title(title)
    print('Close the figures to continue.')
    plt.show(block=block)
    

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


def plot3DPoints(points_3d_pose, cameras, world_ref=True, bounding_box_size=1, block=True):
    """
    Visualiza puntos 3D junto con los sistemas de referencia de varias cámaras en un espacio 3D.

    Parámetros:
        points_3d_pose: Puntos 3D a visualizar (3, N)
        cameras: Diccionario de cámaras con nombres como claves y matrices de transformación 4x4 como valores
                 Ejemplo: {'C1': T_wc1, 'C2': T_wc2}
        world_ref: Booleano para indicar si se debe dibujar el sistema de referencia del mundo.
    """
    fig3D = plt.figure()
    ax = fig3D.add_subplot(111, projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Dibujar el sistema de referencia del mundo si se especifica
    if world_ref:
        drawRefSystem(ax, np.eye(4), '-', 'W')

    # Dibujar cada sistema de referencia de cámara
    for cam_name, T_wc in cameras.items():
        drawRefSystem(ax, T_wc, '-', cam_name)

    # Dibujar los puntos 3D
    ax.scatter(points_3d_pose[0, :], points_3d_pose[1, :], points_3d_pose[2, :], marker='.', color='b')

    # Crear una caja de límites para una visualización más equilibrada
    x_fake = np.linspace(-bounding_box_size, bounding_box_size, 2)
    y_fake = np.linspace(-bounding_box_size, bounding_box_size, 2)
    z_fake = np.linspace(-bounding_box_size, bounding_box_size, 2)
    ax.plot(x_fake, y_fake, z_fake, 'w.')

    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show(block=block)