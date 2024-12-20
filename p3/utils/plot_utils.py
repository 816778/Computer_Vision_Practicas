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


def draw_epipolar_lines(img1, img2, F, pts1, pts2):
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1_ep = img1.copy()
    img2_ep = img2.copy()
    for r, pt1, pt2 in zip(lines1, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Convertir puntos a enteros y asegurar formato de tupla
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1]])
        
        # Dibujar línea epipolar en img1
        img1_ep = cv2.line(img1_ep, (x0, y0), (x1, y1), color, 1)
        # Dibujar puntos en ambas imágenes
        img1_ep = cv2.circle(img1_ep, pt1, 5, color, -1)
        img2_ep = cv2.circle(img2_ep, pt2, 5, color, -1)
    
    # Mostrar las imágenes con líneas epipolares
    plt.subplot(121), plt.imshow(img1_ep)
    plt.subplot(122), plt.imshow(img2_ep)
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

def plot_epipolar_lines(F, H, x1, img2, title='Epipolar lines'):
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

    # Append homogeneous coordinate
    p1 = np.append(x1, np.ones((1, x1.shape[1])), axis=0)
    print(p1)
    x2 = H @ p1
    x2 = x2 / x2[2, :]
    plt.scatter(x2[0], x2[1], color='red', s=50, marker='x', label='Puntos proyectados')

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



def transfer_points_with_homography(H, matched_points, path_image_1, path_image_2, title="Transferred Points"):
    """
    Transfer points from image 1 to image 2 using the homography H.
    
    :param H: Homography matrix
    :param matched_points: Matched points between the two images (N x 4 matrix where first 2 cols are x,y in image1, last 2 cols are x,y in image2)
    :param path_image_1: Path to image 1
    :param path_image_2: Path to image 2
    :param title: Title of the plot
    """
    # Read images
    image1 = cv2.imread(path_image_1)
    image2 = cv2.imread(path_image_2)
    
    # Get the source points (points from image 1)
    src_points = matched_points[:, :2]

    # Convert the points to homogeneous coordinates for applying the homography
    src_points_hom = np.hstack([src_points, np.ones((src_points.shape[0], 1))])  # Convert to N x 3

    # Apply the homography to transfer the points to image 2
    projected_points_hom = H @ src_points_hom.T
    projected_points_hom /= projected_points_hom[2, :]  # Normalize to homogeneous coordinates

    # Extract the transferred points from homogeneous coordinates
    projected_points = projected_points_hom[:2, :].T

    # Visualize the points before and after the transformation
    plt.figure(figsize=(10, 5))

    # Show original image 1 with the keypoints
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.scatter(src_points[:, 0], src_points[:, 1], marker='o', color='r', label='Image 1 Points')
    plt.title(f'Image 1: Original Points')

    # Show image 2 with the projected points
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.scatter(projected_points[:, 0], projected_points[:, 1], marker='x', color='b', label='Projected Points')
    plt.title(f'Image 2: Projected Points using Homography')

    plt.suptitle(title)
    plt.show()




