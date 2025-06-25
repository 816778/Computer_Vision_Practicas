Comando bash

```bash
python ../SuperGluePretrainedNetwork/match_pairs.py --resize 2000 1126 --superglue outdoor --max_keypoints 100000 --nms_radius 3 --resize_float --input_dir images/new --input_pairs data/pilar.txt --output_dir results --viz
```

```bash
python ../SuperGluePretrainedNetwork/match_pairs.py --resize 1126 2000 --superglue outdoor --max_keypoints 100000 --nms_radius 3 --resize_float --input_dir images/prueba --input_pairs data/prueba_casa.txt --output_dir results --viz
```

```bash
python ../SuperGluePretrainedNetwork/match_pairs.py --resize 2000 1126 --superglue outdoor --max_keypoints 100000 --nms_radius 3 --resize_float --input_dir images/prueba --input_pairs data/prueba_casa.txt --output_dir results/foc2 --viz
```

```bash
python ../SuperGluePretrainedNetwork/match_pairs.py --resize 2000 1126 --superglue outdoor --max_keypoints 100000 --nms_radius 3 --resize_float --input_dir images/new_2/foc_1 --input_pairs data/pilar_foc1.txt --output_dir results/foc1 --viz
```

```bash
python ../SuperGluePretrainedNetwork/match_pairs.py --resize 752 480 --superglue indoor --max_keypoints 100000 --nms_radius 3 --resize_float --input_dir images --input_pairs data/input_pairs.txt --output_dir results --viz
```