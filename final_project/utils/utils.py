import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as scAlg
import scipy as sc
import scipy.optimize as scOptim
import scipy.io as sio
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm, logm
import time


###############################################################################################
# Funciones para la calibración de la cámara
###############################################################################################
def indexMatrixToMatchesList(matchesList):
    """
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return dMatchesList


def matchWith2NDRR(desc1, desc2, distRatio, maxDist=100):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.
    -input:
        desc1: descriptors from image 1 nDesc x 128
        desc2: descriptors from image 2 nDesc x 128
        distRatio:
    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        # Ordena las distancias y selecciona los índices de los dos más cercanos
        indexSort = np.argsort(dist)
        d1 = dist[indexSort[0]]  # Distancia al vecino más cercano
        d2 = dist[indexSort[1]] # Distancia al segundo vecino más cercano

        if d1 < d2 * distRatio and d1 < maxDist:
            matches.append([kDesc1, indexSort[0], d1])
    
    return matches



def visualize_matches(path_image_1, path_image_2, distRatio, maxDist, draw=True):
    image1 = cv2.imread(path_image_1)
    image2 = cv2.imread(path_image_2)

    # Feature extraction
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    matchesList = matchWith2NDRR(descriptors1, descriptors2, distRatio=distRatio, maxDist=maxDist)
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)

    # Dibujar los primeros 100 emparejamientos
    img_matched = cv2.drawMatches(
        image1, keypoints1, image2, keypoints2, dMatchesList[:100], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    if draw:
        plt.imshow(img_matched, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Emparejamientos con distRatio = {distRatio}")
        plt.subplots_adjust(
            top=0.985,     # Border for top
            bottom=0.015,  # Border for bottom
            left=0.028,    # Border for left
            right=0.992,   # Border for right
        )
        plt.draw()
        plt.waitforbuttonpress()
    
    return dMatchesList, keypoints1, keypoints2  # Retornar las coincidencias para análisis posterior



def load_matches_superglue(path):
    npz = np.load(path)
    keypoints1 = npz['keypoints0'] 
    keypoints2 = npz['keypoints1']  
    matches = npz['matches']

    valid_matches_idx = matches > -1
    keypoints1_matched = keypoints1[valid_matches_idx]
    keypoints2_matched = keypoints2[matches[valid_matches_idx]]

    keypoints_cv1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in keypoints1_matched]
    keypoints_cv2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in keypoints2_matched]

    # Crear objetos DMatch con índices secuenciales
    dMatchesList = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(keypoints_cv1))]


    # Convierte los emparejamientos a coordenadas (x, y)
    srcPts = keypoints1_matched  # Ya es un array con coordenadas (x, y)
    dstPts = keypoints2_matched
    x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))

    matched_points = np.hstack((x1, x2))
    matched_points = np.hstack((srcPts, dstPts))
    return matched_points, srcPts, dstPts




def do_matches(option=0, path_image_1='images/image1.png', path_image_2='images/image2.png', draw=False):
    if option == 0:
        path = './results/image1_image2_matches.npz'
        npz = np.load(path)

        keypoints1 = npz['keypoints0'] 
        keypoints2 = npz['keypoints1']  
        matches = npz['matches']

        valid_matches_idx = matches > -1
        keypoints1_matched = keypoints1[valid_matches_idx]
        keypoints2_matched = keypoints2[matches[valid_matches_idx]]

        keypoints_cv1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in keypoints1_matched]
        keypoints_cv2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in keypoints2_matched]

        # Crear objetos DMatch con índices secuenciales
        dMatchesList = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(keypoints_cv1))]


        # Convierte los emparejamientos a coordenadas (x, y)
        srcPts = keypoints1_matched  # Ya es un array con coordenadas (x, y)
        dstPts = keypoints2_matched
        x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
        x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))

        matched_points = np.hstack((x1, x2))
        matched_points = np.hstack((srcPts, dstPts))

    elif option == 1:
        distRatio = 0.75
        maxDist = 500
        dMatchesList, keypoints1, keypoints2 = visualize_matches(path_image_1, path_image_2, distRatio, maxDist, draw=draw)

        print("Total de keypoints en la primera imagen:", len(keypoints1))
        print("Total de keypoints en la segunda imagen:", len(keypoints2))

        # Convierte los emparejamientos a coordenadas (x, y)
        srcPts = np.float32([keypoints1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
        dstPts = np.float32([keypoints2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
        x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
        x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))

        matched_points = np.hstack((x1, x2))
        matched_points = np.hstack((srcPts, dstPts))

    else:
        matched_points = None

    return matched_points, srcPts, dstPts


####################################################################################
## Auxiliar functions
#####################################################################################
import numpy as np 
import cv2 
import pdb

def SerializeKeypoints(kp): 
    """Serialize list of keypoint objects so it can be saved using pickle
    
    Args: 
    kp: List of keypoint objects 
    
    Returns: 
    out: Serialized list of keypoint objects"""

    out = []
    for kp_ in kp: 
        temp = (kp_.pt, kp_.size, kp_.angle, kp_.response, kp_.octave, kp_.class_id)
        out.append(temp)

    return out

def DeserializeKeypoints(kp): 
    """Deserialize list of keypoint objects so it can be converted back to
    native opencv's format.
    
    Args: 
    kp: List of serialized keypoint objects 
    
    Returns: 
    out: Deserialized list of keypoint objects"""

    out = []
    for point in kp:
        temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2],
         _response=point[3], _octave=point[4], _class_id=point[5]) 
        out.append(temp)

    return out

def SerializeMatches(matches): 
    """Serializes dictionary of matches so it can be saved using pickle
    
    Args: 
    matches: List of matches object
    
    Returns: 
    out: Serialized list of matches object"""

    out = []
    for match in matches: 
        matchTemp = (match.queryIdx, match.trainIdx, match.imgIdx, match.distance) 
        out.append(matchTemp)
    return out

def DeserializeMatches(matches): 
    """Deserialize dictionary of matches so it can be converted back to 
    native opencv's format. 
    
    Args: 
    matches: Serialized list of matches object
    
    Returns: 
    out: List of matches object"""

    out = []
    for match in matches:
        out.append(cv2.DMatch(match[0],match[1],match[2],match[3])) 
    return out

def GetAlignedMatches(kp1,desc1,kp2,desc2,matches):
    """Aligns the keypoints so that a row of first keypoints corresponds to the same row 
    of another keypoints
    
    Args: 
    kp1: List of keypoints from first (left) image
    desc1: List of desciptros from first (left) image
    kp2: List of keypoints from second (right) image
    desc2: List of desciptros from second (right) image
    matches: List of matches object
    
    Returns: 
    img1pts, img2pts: (n,2) array where img1pts[i] corresponds to img2pts[i] 
    """

    #Sorting in case matches array isn't already sorted
    matches = sorted(matches, key = lambda x:x.distance)

    #retrieving corresponding indices of keypoints (in both images) from matches.  
    img1idx = np.array([m.queryIdx for m in matches])
    img2idx = np.array([m.trainIdx for m in matches])

    #filtering out the keypoints that were NOT matched. 
    kp1_ = (np.array(kp1))[img1idx]
    kp2_ = (np.array(kp2))[img2idx]

    #retreiving the image coordinates of matched keypoints
    img1pts = np.array([kp.pt for kp in kp1_])
    img2pts = np.array([kp.pt for kp in kp2_])

    return img1pts,img2pts

def pts2ply(pts,colors,filename='out.ply'): 
    """Saves an ndarray of 3D coordinates (in meshlab format)"""

    with open(filename,'w') as f: 
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(pts.shape[0]))
        
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        
        f.write('end_header\n')
        
        #pdb.set_trace()
        colors = colors.astype(int)
        for pt, cl in zip(pts,colors): 
            f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],
                                                cl[0],cl[1],cl[2]))

def DrawCorrespondences(img, ptsTrue, ptsReproj, ax, drawOnly=50): 
    """
    Draws correspondence between ground truth and reprojected feature point

    Args: 
    ptsTrue, ptsReproj: (n,2) numpy array
    ax: matplotlib axis object
    drawOnly: max number of random points to draw

    Returns: 
    ax: matplotlib axis object
    """
    ax.imshow(img)
    
    randidx = np.random.choice(ptsTrue.shape[0],size=(drawOnly,),replace=False)
    ptsTrue_, ptsReproj_ = ptsTrue[randidx], ptsReproj[randidx]
    
    colors = colors=np.random.rand(drawOnly,3)
    
    ax.scatter(ptsTrue_[:,0],ptsTrue_[:,1],marker='x',c='r',linewidths=.1, label='Ground Truths')
    ax.scatter(ptsReproj_[:,0],ptsReproj_[:,1],marker='x',c='b',linewidths=.1, label='Reprojected')
    ax.legend()

    return ax