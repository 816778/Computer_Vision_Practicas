from scipy.linalg import expm, logm
import numpy as np
import scipy.optimize as scOptim
from scipy.optimize import least_squares


from utils import geometry, projection

def residual_bundle_adjustment(params, K, xData, nImages, verbose):

    X_w = params[nImages*6:].reshape(-1, 3).T
    residuals = np.array([])

    for i in range(nImages):
        t = params[i*6:(i*6)+3]
        th = params[(i*6)+3:(i*6)+6]
        R = expm(geometry.crossMatrix(th))

        T_wc = np.zeros((3, 4))
        T_wc[:3, :3] = R
        T_wc[:3, 3] = t

        x_proj = projection.project_points(K, T_wc, X_w)

        residuals = np.hstack((residuals,
            ((x_proj[:2, :] - xData[i][:2, :])).flatten()
        ))

    if verbose:
        print("Residuals: ", residuals.mean())

    return residuals


def run_bundle_adjustment(T, K, X_w, xData, verbose=False):
    
    nImages = len(T)
    if X_w.shape[0] == 4:
        X_w = (X_w[:3, :] / X_w[3, :])

    T_flat = np.array([])
    for i in range(nImages):
        t = T[i][:3, 3]
        R = T[i][:3, :3]
        th = geometry.crossMatrixInv(logm(R))

        T_flat = np.hstack((T_flat, t, th))

    X_w_flat = X_w.T.flatten()
    initial_params = np.hstack((T_flat, X_w_flat))

    result = scOptim.least_squares(residual_bundle_adjustment, initial_params,
                                   args=(K, xData, nImages, verbose), method='lm')
    
    optimized_params = result.x
    T_opt = []

    for i in range(nImages):
        t = optimized_params[i*6:(i*6)+3]
        th = optimized_params[(i*6)+3:(i*6)+6]
        R = expm(geometry.crossMatrix(th))

        T_wc = np.zeros((3, 4))
        T_wc[:3, :3] = R
        T_wc[:3, 3] = t

        T_opt.append(T_wc)

    X_w_opt = optimized_params[nImages*6:].reshape(-1, 3).T

    return T_opt, X_w_opt

