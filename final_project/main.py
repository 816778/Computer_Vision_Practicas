####################################################################################################
#
# Title: main.py
# Project: 
# Authors: Eryka Rimacuna and Hugo Mateo
# Description: This file contains the necessary imports for the main.py file.
#
####################################################################################################

import numpy as np
import cv2
import scipy.linalg as scAlg
import csv
import scipy as sc
import scipy.optimize as scOptim
import scipy.io as sio

import utils.utils as utils
import utils.plot_utils as plot_utils

if __name__ == "__main__":
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)
    