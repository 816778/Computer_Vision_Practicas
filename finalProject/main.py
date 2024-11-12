import numpy as np
import lib.data_process as data_process
import lib.geometric_math as math
import lib.plot as plot
import lib.sfm as sfm
import lib.matching as matching

work_dir = "/home/hsunekichi/Escritorio/Computer_Vision_Practicas/p4/"

im1_pth = work_dir+"images/image1.png"
im2_pth = work_dir+"images/image2.png"
im3_pth = work_dir+"images/image3.png"


def load_data():
    T_wc1 = np.loadtxt(work_dir+"data/T_w_c1.txt")
    T_wc2 = np.loadtxt(work_dir+"data/T_w_c2.txt")
    T_wc3 = np.loadtxt(work_dir+"data/T_w_c3.txt")
    K_c = np.loadtxt(work_dir+"data/K_c.txt")
    X_w = np.loadtxt(work_dir+"data/X_w.txt")
    x1Data = np.loadtxt(work_dir+"data/x1Data.txt")
    x2Data = np.loadtxt(work_dir+"data/x2Data.txt")
    x3Data = np.loadtxt(work_dir+"data/x3Data.txt")

    # Convertir a coordenadas homogéneas
    x1Data = np.vstack((x1Data, np.ones((1, x1Data.shape[1]))))
    x2Data = np.vstack((x2Data, np.ones((1, x2Data.shape[1]))))
    x3Data = np.vstack((x3Data, np.ones((1, x3Data.shape[1]))))

    return T_wc1, T_wc2, T_wc3, K_c, X_w, x1Data, x2Data, x3Data



#T_wc1, T_wc2, T_wc3, K_c, X_w, x1Data, x2Data, x3Data = load_data()
_, _, _, K_c, _, x1_ref, x2_ref, _ = load_data()

model, device = matching.init_superglue()

image1 = data_process.read_image(im1_pth)
image2 = data_process.read_image(im2_pth)
image3 = data_process.read_image(im3_pth)

kpts1, kpts2, x1, x2 = matching.match_superglue(image1, image2, model, device)

#plot.visualize_matches(image1, image2, x1_rand, x2_rand, 'Matches')

T_wc1, T_wc2, X_w = sfm.bundlePoseEstimation(x1, x2, K_c)


P1 = math.projectionMatrix(K_c, T_wc1)
P2 = math.projectionMatrix(K_c, T_wc2)

X_w = sfm.triangulate_points(P1, P2, x1, x2)


# Step 6: Visualize Optimized Projection
x1_proj = math.project_points(K_c, T_wc1, X_w)
x2_proj = math.project_points(K_c, T_wc2, X_w)

plot.visualize_projection(image1, x1, x1_proj, 'Image 1', False)
plot.visualize_projection(image2, x2, x2_proj, 'Image 2', False)

plot.plot_3D_scene([T_wc1, T_wc2], X_w)

