#####################################################################################
#
# MRGCV Unizar - Computer vision 
#
# Title: Examples with SVD
#
# Date: 15 September 2022
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.linalg as scAlg


def drawParametricLine(l, mu, strFormat, lWidth):
    l = l.reshape(3,1)
    l_norm = l / np.sqrt(np.sum(l[0:2] ** 2, axis=0))
    x_l0 = np.vstack((-l_norm[0:2] * l_norm[2], 1))
    x = x_l0 + np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) @ (l_norm * mu)
    plt.plot(x[0,:],x[1,:],strFormat,linewidth=lWidth)



def drawLine(l,strFormat,lWidth):
    """
    Draw a line
    -input:
      l: image line in homogenous coordinates
      strFormat: line format
      lWidth: line width
    -output: None
    """
    # p_l_y is the intersection of the line with the axis Y (x=0)
    p_l_y = np.array([0, -l[2] / l[1]])
    # p_l_x is the intersection point of the line with the axis X (y=0)
    p_l_x = np.array([-l[2] / l[0], 0])
    # Draw the line segment p_l_x to  p_l_y
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], strFormat, linewidth=lWidth)



if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    Pi1 = np.array([2, 1, 0, -8])
    Pi2 = np.array([-1, 1, 0, 1])
    Pi3 = np.array([2, 3, 0, -12])
    Pi4 = np.array([1, -2, 0, 1])

    l1 = np.array([2, 1, -8])
    l2 = np.array([-1, 1, 1])
    l3 = np.array([2, 3, -12])
    l4 = np.array([1, -2, 1])

    plt.figure(1)
    plt.plot([-0.5, 5], [0, 0], '--k', linewidth=1) #Plot the axis
    plt.plot([0, 0], [-0.5, 5], '--k', linewidth=1) #Plot the axis
    # Draw the line segment p_l_x to  p_l_y

    mu = np.arange(-5, 5, 1)
    drawParametricLine(l1, mu,'r-',1)
    drawParametricLine(l2, mu, 'g-', 1)
    drawParametricLine(l3, mu, 'b-', 1)
    drawParametricLine(l4, mu, 'k-', 1)
    plt.draw()
    plt.axis('equal')
    plt.show()

    # Computing intersection with cross product
    x_I = np.cross(l1,l2)
    x_I = x_I/x_I[2]

    # With two lines
    A = np.double(np.vstack((l1, l2)))
    u, s, vh = np.linalg.svd(A)   # vh is v.T
    sM = scAlg.diagsvd(s, u.shape[0], vh.shape[0])
    v = vh.T
    x_I = v[:, -1] / v[-1, -1]

    # With three lines
    A = np.double(np.vstack((l1,l2,l3)))
    u, s, vh = np.linalg.svd(A)   # vh is v.T
    sM = scAlg.diagsvd(s, u.shape[0], vh.shape[0])
    v = vh.T
    x_I = v[:, -1] / v[-1, -1]

    # With three lines (not perfect points)
    A = np.double(np.vstack((l1,l2,l3)))
    A[0:3, 2:3] = A[0:3, 2:3] + np.random.normal(0, 0.01, (3, 1))
    u, s, vh = np.linalg.svd(A)   # vh is v.T
    sM = scAlg.diagsvd(s, u.shape[0], vh.shape[0])
    v = vh.T
    x_I = v[:, -1] / v[-1, -1]

    # With four lines (not perfect points)
    A = np.double(np.vstack((l1, l2, l3, l4)))
    A[0:4, 2:3] = A[0:4, 2:3] + np.random.normal(0, 0.01, (4, 1))
    u, s, vh = np.linalg.svd(A)   # vh is v.T
    sM = scAlg.diagsvd(s, u.shape[0], vh.shape[0])
    v = vh.T
    x_I = v[:, -1] / v[-1, -1]


    print('End')
