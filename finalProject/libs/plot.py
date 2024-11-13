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


def plot_3D_scene(T, X_w):
    fig3D = plt.figure()
    ax = fig3D.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    for (i, t) in enumerate(T):
        drawRefSystem(ax, t, '-', 'C' + str(i + 1))

    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')

    # Plot unnumbered 3d points
    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')

    # Set the same scale for all axes
    max_range = np.array([X_w[0, :].max() - X_w[0, :].min(), 
                            X_w[1, :].max() - X_w[1, :].min(), 
                            X_w[2, :].max() - X_w[2, :].min()]).max() / 2.0

    mid_x = (X_w[0, :].max() + X_w[0, :].min()) * 0.5
    mid_y = (X_w[1, :].max() + X_w[1, :].min()) * 0.5
    mid_z = (X_w[2, :].max() + X_w[2, :].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    print('Close the figures to continue.')
    plt.show()



def visualize_matches_cv(image1, image2, kp1, kp2, dMatchesList, title):
    imgMatched = cv2.drawMatches(image1, kp1, image2, kp2, dMatchesList, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure()
    plt.imshow(imgMatched)
    plt.title(title)
    print('Close the figures to continue.')
    plt.show()

def visualize_matches(image1, image2, srcPts, dstPts, title):
    """
    Plot matches between two images, showing the lines connecting the matches.

    Args:
    - image1: First image.
    - image2: Second image.
    - srcPts: Source points (2d np array in the first image).
    - dstPts: Destination points (2d np aray the second image).
    - title: Title of the plot.
    """

    x1 = srcPts.T
    x2 = dstPts.T

    # Convert points to cv match list
    pts1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in x1]
    pts2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in x2]

    cv_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(srcPts.shape[1])]

    # Draw matches
    visualize_matches_cv(image1, image2, pts1, pts2, cv_matches, title)

def visualize_projection_error(image, xData, xProj, title, block=True):
    plt.figure()
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plotResidual(xData, xProj, 'k-')
    plt.plot(xProj[0, :], xProj[1, :], 'bo')
    plt.plot(xData[0, :], xData[1, :], 'rx')
    plotNumberedImagePoints(xData[0:2, :], 'r', 4)
    plt.title(title)
    print('Close the figures to continue.')
    plt.show(block=block)

def visualize_projection(image, xProj, title, block=True):
    plt.figure()
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.plot(xProj[0, :], xProj[1, :], 'bo', markersize=3)
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



def plot_epipolar_lines(image1, image2, p1, p2, F):
    """
    Plot epipolar lines for a set of points given in homogeneous coordinates and a fundamental matrix over an OpenCV image.
    
    Parameters:
    image (np.ndarray): The image over which to plot the lines.
    points (np.ndarray): An array of shape (N, 3) where each row is a 2D point in homogeneous coordinates.
    F (np.ndarray): A 3x3 fundamental matrix.
    """

    # Plot image 1 with its points
    plt.figure(figsize=(10, 10))
    if len(image1.shape) == 2:  # Grayscale
        plt.imshow(image1, cmap='gray')
    else:  # Color
        plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

    img_height, img_width = image1.shape[:2]

    for i in range(p1.shape[1]):
        # Plot the original point
        x, y, _ = p1[:, i]
        plt.plot(x, y, 'o', label=f'Point {i+1}', markersize=5)
    
    plt.xlim(0, img_width)
    plt.ylim(img_height, 0)  # Invert y-axis for image coordinates
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points on Image 1')
    plt.legend()
    plt.show(block=False)

    # Display the image
    plt.figure(figsize=(10, 10))
    if len(image2.shape) == 2:  # Grayscale
        plt.imshow(image2, cmap='gray')
    else:  # Color
        plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    

    # Convert points to epipolar lines
    lines = F @ p1  # Shape: (3, N)
    img_height, img_width = image2.shape[:2]
    
    for i in range(p1.shape[1]):
        # Epipolar line parameters (a, b, c) from the line equation ax + by + c = 0
        a, b, c = lines[:, i]
        
        # Define points for plotting the line within the image bounds
        x_vals = np.array([0, img_width - 1])  # x-coordinates at image left and right boundaries
        y_vals = -(a * x_vals + c) / b if b != 0 else np.array([0, img_height - 1])

        # Plot the line
        plt.plot(x_vals, y_vals, label=f'Line {i+1}')
        
        # Plot the original point
        x, y, _ = p2[:, i]
        plt.plot(x, y, 'o', label=f'Point {i+1}', markersize=5)

    plt.xlim(0, img_width)
    plt.ylim(img_height, 0)  # Invert y-axis for image coordinates
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Epipolar Lines on Image')
    plt.legend()
    plt.show()
