import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
After execute:
    python ../SuperGluePretrainedNetwork/match_pairs.py --resize 752 --superglue indoor --max_keypoints 2048 --nms_radius 3 --resize_float --input_dir images --input_pairs data/euroc_sample_pairs.txt --output_dir results
"""
if __name__ == '__main__':
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
    matches_cv = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(keypoints_cv1))]

    img1 = cv2.imread('images/image1.png')
    img2 = cv2.imread('images/image2.png')

    # Dibujar los emparejamientos
    img_matches = cv2.drawMatches(img1, keypoints_cv1, img2, keypoints_cv2, matches_cv, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Mostrar el resultado
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title("Emparejamientos SuperGlue")
    plt.subplots_adjust(
        top=0.985,     # Border for top
        bottom=0.015,  # Border for bottom
        left=0.028,    # Border for left
        right=0.992,   # Border for right
    )
    plt.axis('off')
    plt.show()