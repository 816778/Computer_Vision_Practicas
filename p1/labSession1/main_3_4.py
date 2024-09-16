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

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    # Load points from files
    xGT = np.loadtxt(FOLDER_DATA + 'x2DGTLineFittingSVD.txt')  # Points on the true line
    x = np.loadtxt(FOLDER_DATA + 'x2DLineFittingSVD.txt')  # Points with noise
    
    ej3(xGT, x, verbose=True)
    
