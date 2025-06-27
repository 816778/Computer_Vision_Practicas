import numpy as np
import os

class Reconstruccion3D:
    def __init__(self):
        self.image_data = {}
        self.point_cloud = None  # (N, 3)

    def cargar(self, R1, t1, R2, t2, idx1, idx2, puntos_3d, num_kp1, num_kp2, img1_name, img2_name, T_wc1=None, T_wc2=None):
        M = puntos_3d.shape[0]

        ref1 = np.ones((num_kp1,), dtype=int) * -1
        ref2 = np.ones((num_kp2,), dtype=int) * -1

        ref1[idx1] = np.arange(M)
        ref2[idx2] = np.arange(M)

        self.image_data[img1_name] = {
            "R": R1,
            "t": t1,
            "ref": ref1,
            "t_wc": T_wc1
        }

        self.image_data[img2_name] = {
            "R": R2,
            "t": t2,
            "ref": ref2,
            "t_wc": T_wc2
        }

        self.point_cloud = puntos_3d

    def add_view_refs(self, img_name: str, ref: np.ndarray):
        self.image_data[img_name] = {"ref": ref.astype(int)}


    def add_r_t(self, img_name: str, R, t, T_wc):
        if img_name not in self.image_data:
            self.image_data[img_name] = {}
        self.image_data[img_name]["R"] = R
        self.image_data[img_name]["t"] = t
        self.image_data[img_name]["t_wc"] = T_wc

    def print_info(self):
        print(f"Total de imágenes: {len(self.image_data)}")
        for img_name, data in self.image_data.items():
            print(f"Imagen: {img_name}, R: {data['R'].shape}, t: {data['t'].shape}, ref: {data['ref'].shape if 'ref' in data else 'N/A'}, T_wc: {data['t_wc'].shape if 't_wc' in data else 'N/A'}")
        
    