import cv2
import glob
import numpy as np
import argparse
import json
import os

def parse_image_names(npz_filename: str):
    """
    Acepta <algo>/<imgA>_<imgB>_matches.npz y devuelve (imgA, imgB)
    so-porta nombres con guiones u otros '_' en la segunda mitad.
    """
    base = os.path.basename(npz_filename)
    if not base.endswith('_matches.npz'):
        raise ValueError(f'Nombre inesperado: {base}')

    stem = base[:-len('_matches.npz')]       # --> 'image1_image2'  o  'image1_old-image'
    img0, img1 = stem.split('_', 1)          # solo divide en el primer '_'  
                                             # 'image1', 'image2'  o  'image1', 'old-image'
    return f'{img0}.jpg', f'{img1}.jpg'


def dump_keypoints_txt(image_name: str, kpts: np.ndarray, out_dir: str):
    """
    Escribe  <out_dir>/<image_name>.txt   siguiendo el formato COLMAP
    (x+0.5, y+0.5, 0.0, 0.0, 128 ceros).
    """
    path = os.path.join(out_dir, f'{image_name}.txt')
    with open(path, 'w') as f:
        f.write(f'NUM_FEATURES {len(kpts)}\n')
        for x, y in kpts:
            f.write(f'{x + 0.5:.6f} {y + 0.5:.6f} 0.0 0.0 ' + ' '.join(['0'] * 128) + '\n')


def superglue_to_colmap(params, kpt_dir: str, matches_txt: str):
    """
    Crea (1) archivos de keypoints y (2) matches.txt.
    """
    os.makedirs(kpt_dir, exist_ok=True)

    image_kpts = {}          # { 'image1.jpg': ndarray(N,2), ... }
    matches_lines = []       # líneas a volcar en matches.txt

    all_npz = params['path_superglue'] + params.get('path_old_superglue', [])

    for npz_file in all_npz:
        data = np.load(npz_file)
        kpts0, kpts1, matches = data['keypoints0'], data['keypoints1'], data['matches']

        img0, img1 = parse_image_names(npz_file)

        # ––– Keypoints (los guardamos si aún no existen) –––
        if img0 not in image_kpts:
            image_kpts[img0] = kpts0
        if img1 not in image_kpts:
            image_kpts[img1] = kpts1

        # ––– Matches –––
        good = [(i, int(j)) for i, j in enumerate(matches) if j >= 0]
        if good:                              # puede que no haya matches útiles
            matches_lines.append(f'{img0} {img1}\n')
            matches_lines += [f'{i} {j}\n' for i, j in good]

    # 1.  Escribimos los .txt de keypoints
    for img, kpts in image_kpts.items():
        dump_keypoints_txt(img, kpts, kpt_dir)

    # 2.  Escribimos matches.txt
    with open(matches_txt, 'w') as f:
        f.writelines(matches_lines)

    print(f'✅  Keypoints guardados en:   {kpt_dir}')
    print(f'✅  Matches  guardados en:   {matches_txt}')
    print('\nAhora puedes importar en COLMAP con:')
    print(f'  colmap feature_importer  --database_path path/to/database.db '
          f'--image_path path/to/images  --import_path {kpt_dir}')
    print(f'  colmap matches_importer  --database_path path/to/database.db '
          f'--match_list_path {matches_txt}  --match_type inliers')




if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)
    parser = argparse.ArgumentParser(description="Ejemplo de script con argumento 'test' desde la línea de comandos.")
    
    # Definir el argumento para 'test', con valor por defecto 0
    parser.add_argument(
        '--test', 
        type=int, 
        default=0, 
        help="Valor de la variable 'test'. Valor por defecto es 0."
    )


    args = parser.parse_args()
    test = args.test

    with open('data/config.json', 'r') as f:
        config = json.load(f)

    if test == 0:
        params = config['test_0']
    elif test == 1:
        params = config['test_1']
    else:
        params = config['default']
       
    superglue_to_colmap(params, args.kpt_dir, args.matches_txt)
    
    