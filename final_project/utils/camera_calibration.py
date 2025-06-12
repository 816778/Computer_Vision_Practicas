import os
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt


def calibrate_camera(images, pattern_size, downsize_factor, criteria):
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Puntos del patrón en 3D
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    for fname in images:
        img = cv.imread(fname)
        img_rows = img.shape[1]
        img_cols = img.shape[0]
        new_img_size = (int(img_rows / downsize_factor), int(img_cols / downsize_factor))
        img = cv.resize(img, new_img_size, interpolation=cv.INTER_CUBIC)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        print('Processing calibration image:', fname)
        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv.imshow('Chessboard', img)
            cv.waitKey(500)

    cv.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=(cv.CALIB_ZERO_TANGENT_DIST))
    return objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs



def calculate_calibration(objpoints, imgpoints, image_size):
    # Calcula los parámetros de calibración
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    return mtx, dist, rvecs, tvecs



def save_calibration_data(mtx, dist, rvecs, tvecs, folder="data", fov=1):
    os.makedirs(folder, exist_ok=True)
    np.savez(os.path.join(folder, f"camera_calibration_{fov}.npz"), mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print(f"Calibration data saved to {folder}/camera_calibration_{fov}.npz")


def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    return mean_error / len(objpoints)


def load_calibration_data(file_path="data/camera_calibration.npz"):
    # Cargar los datos de calibración desde un archivo .npz
    data = np.load(file_path)
    mtx = data["mtx"]
    dist = data["dist"]
    rvecs = data["rvecs"]
    tvecs = data["tvecs"]
    print("Calibration data loaded successfully!")
    return mtx, dist, rvecs, tvecs




