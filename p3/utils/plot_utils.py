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



def plot_epipolar_lines(F, x1, img2):
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


