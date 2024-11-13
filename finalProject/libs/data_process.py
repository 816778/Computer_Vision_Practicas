import cv2
import numpy as np
import torch


def indexMatrixToMatchesList(matchesList):
    """
    Convert a numpy matrix of index in a list of DMatch OpenCv matches.
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0].astype('int'), _trainIdx=row[1].astype('int'), _distance=row[2]))
    return dMatchesList


def matchesListToIndexMatrix(dMatchesList):
    """
    Convert a list of DMatch OpenCv matches into a numpy matrix of index.

     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([np.int32(dMatchesList[k].queryIdx), np.int32(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Image not found")
    
    return image
