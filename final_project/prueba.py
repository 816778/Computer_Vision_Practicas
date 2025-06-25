import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
After execute:
    python ../SuperGluePretrainedNetwork/match_pairs.py --resize 752 --superglue indoor --max_keypoints 2048 --nms_radius 3 --resize_float --input_dir images --input_pairs data/euroc_sample_pairs.txt --output_dir results
"""
if __name__ == '__main__':
    points = np.loadtxt("results/output.ply", skiprows=10)[:, :3]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    plt.show()
    exit()
    
    path = 'results/foc2/IMG_2_4_3_IMG_2_5_3_matches.npz'
    npz = np.load(path)
    
    keypoints1 = npz['keypoints0'] 
    keypoints2 = npz['keypoints1']  
    matches = npz['matches']

 
    confidence_threshold = 0.8

    if 'match_confidence' in npz:
        confidences = npz['match_confidence']
        matches = npz['matches']

        valid_matches_idx = (matches > -1) & (confidences > confidence_threshold)

        keypoints1_matched = keypoints1[valid_matches_idx]
        keypoints2_matched = keypoints2[matches[valid_matches_idx]]

        print(f"[INFO] Total matches: {len(confidences[valid_matches_idx])}")
        print(f"[INFO] Confianza mínima: {confidences[valid_matches_idx].min():.4f}")
        print(f"[INFO] Confianza máxima: {confidences[valid_matches_idx].max():.4f}")
    else: 
        valid_matches_idx = matches > -1

        keypoints1_matched = keypoints1[valid_matches_idx]
        keypoints2_matched = keypoints2[matches[valid_matches_idx]]

    keypoints_cv1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in keypoints1_matched]
    keypoints_cv2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in keypoints2_matched]

    srcPts = keypoints1_matched  # Ya es un array con coordenadas (x, y)
    dstPts = keypoints2_matched
    x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))

    matched_points = np.hstack((x1, x2))
    matched_points = np.hstack((srcPts, dstPts))

    # Crear objetos DMatch con índices secuenciales
    matches_cv = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(keypoints_cv1))]

    img1 = cv2.imread('images/prueba/IMG_2_4_3.jpg')
    img2 = cv2.imread('images/prueba/IMG_2_5_3.jpg')

    img1 = cv2.resize(img1, (2000, 1126))
    img2 = cv2.resize(img2, (2000, 1126))
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