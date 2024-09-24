import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scAlg
import utils.plot_utils as plot_utils
import utils.utils as utils

FOLDER_DATA = "data/"

def ej3(xGT, x, verbose=False):
    plot_utils.plot_true_line()
    """
    Ejercicio 3.1
    """
    # Plot the original points
    plt.plot(xGT[0, :], xGT[1, :], 'b.')
    plt.plot(x[0, :], x[1, :], 'rx')
    plt.draw()

    # Select only the 2 extreme points from the noisy points x
    x_extreme = x[:, [0, -1]]  # Select the first and the last points

    # Perform SVD on the 2 extreme points
    u, s, vh = np.linalg.svd(x_extreme.T)  # SVD on the two extreme points

    # El último vector en V^T (vh) es el vector que minimiza el error
    l_ls = vh[-1, :]  # Last row of vh is the solution

    # Dibujar la línea ajustada
    plot_utils.drawLine(l_ls, 'r--', 1, 'Line extreme noisy points')

    if verbose:
        print(f'SVD with extreme points:')
        print(f'x_extreme size: {x_extreme.shape}, U size: {u.shape}, S size: {s.shape}, V^T size: {vh.shape}')
        print(f'U:\n{u}, S:\n{s}, V^T:\n{vh}')
        
    """
    El tamaño original de x_xtreme es 2x3, de modo que los tamaños en la descomposición son:
    * U:    2x2
    * S:    2
    * V^T:  3x3
    """
    """
    Ejercicio 3.2
    """ 
    u_gt, s_gt, vh_gt = np.linalg.svd(xGT.T)
    l_ls_gt = vh_gt[-1, :]
    plot_utils.drawLine(l_ls_gt, 'b--', 1, label='Line perfect points')

    if verbose:
        print(f'SVD with perfect points:')
        print(f'xGT size: {xGT.shape},U size: {u_gt.shape}, S size: {s_gt.shape}, V^T size: {vh_gt.shape}')
        print(f'U_gt:\n{u_gt}, S_gt:\n{s_gt}, V^T_gt:\n{vh_gt}')
    """
    El tamaño original de x_xtreme es 5x3, de modo que los tamaños en la descomposición son:
    * U:    5x5
    * S:    3
    * V^T:  3x3
    """

    """
    Ejercicio 3.4
    """
    plt.draw()
    plt.legend()
    plot_utils.plotAndWait('Ej 3_1')
    print('End')



def ej4():
    """
    4.1 Encontrar el plano que pasa por cuatro puntos en 3D. Ecuación del pano 3d es:
    ax+by+cz+d=0
    """
    """
    1. tenemos cuatro puntos, queremos encontrar el plano que contiene a estos puntos
    Cada uno de estos puntos se representa en coordenadas homogéneas
    Formamos una matriz M donde cada fila es uno de los puntos en coordenadas homogéneas:
    """
    X_A = np.array([3.44, 0.80, 0.82])
    X_B = np.array([4.20, 0.80, 0.82])
    X_C = np.array([4.20, 0.60, 0.82])
    X_D = np.array([3.55, 0.60, 0.82])
    X_E = np.array([-0.01, 2.60, 1.21])
    points_3D = np.array([X_A, X_B, X_C, X_D])
    points_3D_hom = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))

    """
    2. Realizamos la descomposición SVD de la matriz M. El vector que corresponde al valor singular más pequeño en V^T (última fila de V^T) es el
    vector normal del plano y los coeficientes [a,b,c,d] de la ecuación del plano
    """
    U, S, Vt = np.linalg.svd(points_3D_hom)
    plane_eq = Vt[-1, :]
    a, b, c, d= plane_eq
    print(f"Ecuación del plano: {a}x + {b}y + {c}z + {d}= 0")
    print(f"{a}, {b}, {c}, {d}")

    """
    4.2 calcular la distancia perpendicular desde cada uno de los puntos al plano
    """
    """
    1. La distancia d desde un punto P(x1,y1,z1) a un plano con ecuación general: ax+by+cz+d=0 se puede
    calcular usando: d = (|ax1+by1+cz1+d|)/(sqrt(a^2+b^2+c^2))
    """
    points_3D_with_E = np.vstack((points_3D, X_E))
    distances_with_E = [np.abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2) 
                    for x, y, z in points_3D_with_E]
    for i, dist in enumerate(distances_with_E):
        print(f"Punto {chr(65 + i)}: {dist:.2f} m")

    plot_utils.plot_plane_3d(points_3D_with_E, a, b, c, d, distances_with_E)





if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    # Load points from files
    xGT = np.loadtxt(FOLDER_DATA + 'x2DGTLineFittingSVD.txt')  # Points on the true line
    x = np.loadtxt(FOLDER_DATA + 'x2DLineFittingSVD.txt')  # Points with noise

    ej4()
    exit()
    ej3(xGT, x, verbose=True)
    
