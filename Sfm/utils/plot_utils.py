import matplotlib.pyplot as plt
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

####################################################################################################
# Funciones de visualización
####################################################################################################

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


####################################################################################################
# VISUALIZACIÓN DE PUNTOS 2D
####################################################################################################


def draw_matches(keypoints1_matched, keypoints2_matched, image_1_path, image_2_path, resize_dim=(2000, 1126)):
    print(f"[INFO] Total matches: {len(keypoints1_matched)}")
    print(f"image_1_path: {image_1_path}")
    print(f"image_2_path: {image_2_path}")

    keypoints_cv1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in keypoints1_matched]
    keypoints_cv2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in keypoints2_matched]

    srcPts = keypoints1_matched  # Ya es un array con coordenadas (x, y)
    dstPts = keypoints2_matched
    x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))

    # Crear objetos DMatch con índices secuenciales
    matches_cv = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(keypoints_cv1))]

    img1 = cv2.imread(image_1_path)
    img2 = cv2.imread(image_2_path)

    img1 = cv2.resize(img1, resize_dim)
    img2 = cv2.resize(img2, resize_dim)
    # Dibujar los emparejamientos
    img_matches = cv2.drawMatches(img1, keypoints_cv1, img2, keypoints_cv2, matches_cv, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Mostrar el resultado
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title("Emparejamientos SuperGlue")
    plt.subplots_adjust(
        top=0.985,     # Border for top
        bottom=0.015,  # Border for bottom
        left=0.028,    # Border for left
        right=0.992,   # Border for right
    )
    plt.axis('off')
    plt.show()


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



def visualize_projection_whitoutGT(image, xProj, title, resize_dim=None):
    plt.figure()
    if resize_dim:
        image = cv2.resize(image, resize_dim)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.plot(xProj[0, :], xProj[1, :], 'bo')
    plt.title(title)
    print('Close the figures to continue.')
    plt.show()


def visualize_projection(image, xData, xProj, title, resize_dim=None):
    plt.figure()
    if resize_dim:
        image = cv2.resize(image, resize_dim)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plotResidual(xData, xProj, 'k-')
    plt.plot(xProj[0, :], xProj[1, :], 'bo')
    plt.plot(xData[0, :], xData[1, :], 'rx')
    plotNumberedImagePoints(xData[0:2, :], 'r', 4)
    plt.title(title)
    print('Close the figures to continue.')
    plt.show()
    

def visualize_projection_2(image, xData, xProj_no_opt, xProj_opt, title, resize_dim=None):
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
    if resize_dim:
        image = cv2.resize(image, resize_dim)
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



def show_points_on_image(x_coords, y_coords, labels=None, block=True):
    """
    Muestra los puntos en la imagen y los etiqueta.
    
    Args:
        image_path (str): Ruta de la imagen donde se mostrarán los puntos.
        points (list or array): Lista o array de puntos a mostrar (cada punto debe ser un array [x, y]).
        labels (list, optional): Lista de etiquetas para los puntos. Si no se proporciona, no se etiquetan.
    """     
    # Dibujar los puntos en la imagen
    plt.scatter(x_coords, y_coords, color='b', s=100, marker='x', label='Puntos')
    
    # Etiquetar los puntos si se proporcionan etiquetas
    if labels is not None:
        for i, label in enumerate(labels):
            plt.text(x_coords[i] + 10, y_coords[i] - 10, label, color='red', fontsize=12)
    
    # Mostrar el resultado
    plt.title('Puntos proyectados en la imagen')
    plt.axis('off')  # Ocultar los ejes
    # Plot and continue
    plt.show(block=block)    

    
####################################################################################################
# VISUALIZACIÓN DE PUNTOS 3D
####################################################################################################

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


def plot3DPoints(points_3d_pose, cameras, world_ref=True):
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
        # print(f'Transformación de cámara {cam_name}:\n{T_wc}')
        drawRefSystem(ax, T_wc, '-', cam_name)

    # Dibujar los puntos 3D
    ax.scatter(points_3d_pose[0, :], points_3d_pose[1, :], points_3d_pose[2, :], marker='.', color='b')

    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()

def plot3DCameras(cameras):
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


    for cam_name, T_wc in cameras.items():
        print(f'Transformación de cámara {cam_name}:\n{T_wc}')
        drawRefSystem(ax, T_wc, '-', cam_name)

    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()


def plot3DPoints_points(points_3d_pose, world_ref=True):
    """
    Visualiza puntos 3D sin los sistemas de referencia de cámaras, solo los puntos 3D.

    Parámetros:
        points_3d_pose: Puntos 3D a visualizar (3, N)
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

    # Dibujar los puntos 3D
    ax.scatter(points_3d_pose[0, :], points_3d_pose[1, :], points_3d_pose[2, :], marker='.', color='b')

    # Crear una caja de límites para una visualización más equilibrada
    bounding_box_size = 4
    x_fake = np.linspace(-bounding_box_size, bounding_box_size, 2)
    y_fake = np.linspace(-bounding_box_size, bounding_box_size, 2)
    z_fake = np.linspace(-bounding_box_size, bounding_box_size, 2)
    ax.plot(x_fake, y_fake, z_fake, 'w.')

    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()


    
def plot3DPoints_tuple(points_3d, cameras, world_ref=True):
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
    for points in points_3d:
        print(points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

    # Crear una caja de límites para una visualización más equilibrada
    bounding_box_size = 4
    x_fake = np.linspace(-bounding_box_size, bounding_box_size, 2)
    y_fake = np.linspace(-bounding_box_size, bounding_box_size, 2)
    z_fake = np.linspace(-bounding_box_size, bounding_box_size, 2)
    ax.plot(x_fake, y_fake, z_fake, 'w.')

    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()



 