import torch
import libs.data_process as dp
import cv2
import numpy as np

from SuperGlue.models.matching import Matching
from SuperGlue.models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)




# Function to convert numpy keypoints to OpenCV KeyPoint objects
def convert_to_cv_keypoints(kpts):
    return [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in kpts]


def init_superglue():

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': 5,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': "indoor",
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }

    # Load the SuperGlue model.
    matching = Matching(config).eval().to(device)

    return matching, device


def superglue_data_to_match_masked(sp2, sp3, matches12, matches23):
    """
    Returns matched keypoints between 2 and 3 images,
    considering only points on 2 that had a match with 1
    """

    kp2, kp3 = sp2['keypoints'], sp3['keypoints']

    mask12 = matches12 > -1
    mask23 = matches23 > -1

    # Take only points that had a match with 1
    mask2 = np.zeros(len(kp2), dtype=bool)
    mask2[matches12[mask12]] = True
    mask2 = mask2 & mask23

    # Take matches between 2 and 3
    mask3 = np.zeros(len(kp3), dtype=bool)
    mask3[matches23[mask2]] = True

    kp2 = kp2[mask2, :].T
    kp3 = kp3[matches23[mask2], :].T

    # Add homogenous coordinates
    kp2 = np.vstack((kp2, np.ones((1, kp2.shape[1]))))
    kp3 = np.vstack((kp3, np.ones((1, kp3.shape[1]))))

    return kp2, kp3, mask2[matches12[mask12]]



def tensors_to_matches(sp1, sp2):

    kp0, kp1 = sp1['keypoints'][0].T.cpu().numpy(), sp2['keypoints'][0].T.cpu().numpy()

    # Add homogenous coordinates
    kp0 = np.vstack((kp0, np.ones((1, kp0.shape[1]))))
    kp1 = np.vstack((kp1, np.ones((1, kp1.shape[1]))))

    return kp0, kp1

def match_superglue(img1, img2, model, device, spData1=None, spData2=None):

    tensor1 = dp.frame2tensor(img1, device)
    tensor2 = dp.frame2tensor(img2, device)

    data = {'image0': tensor1, 'image1': tensor2}

    if spData1 is None:
        pred0 = model.superpoint({'image': data['image0']})
        spData1 = {k+'0': v for k, v in pred0.items()}
    else:
        spData1 = {k+'0': v for k, v in spData1.items()}

    if spData2 is None:
        pred1 = model.superpoint({'image': data['image1']})
        spData2 = {k+'1': v for k, v in pred1.items()}
    else:
        spData2 = {k+'1': v for k, v in spData2.items()}

    data = {**data, **spData1, **spData2}
    pred = model(data)

    matches = pred['matches0'].cpu().short().numpy().flatten()
    mask0 = matches > -1


    # Extract SuperPoint (keypoints, scores, descriptors) erasing the number id from key
    spData1 = {k[:-1]: v for k, v in spData1.items()}
    spData2 = {k[:-1]: v for k, v in spData2.items()}

    spData1['keypoints'] = [spData1['keypoints'][0][mask0]]
    spData2['keypoints'] = [spData2['keypoints'][0][matches[mask0]]]

    spData1['descriptors'] = [spData1['descriptors'][0][:, mask0]]
    spData2['descriptors'] = [spData2['descriptors'][0][:, matches[mask0]]]

    spData1['scores'] = [spData1['scores'][0][mask0]]
    spData2['scores'] = [spData2['scores'][0][matches[mask0]]]

    return spData1, spData2, mask0
 