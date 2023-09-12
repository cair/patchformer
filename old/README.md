 # patchformer
PatchFormer - Improved dense predictions using implicit representation learning


## Chosen models:

* DCSwin (from https://github.com/WangLibo1995/GeoSeg/tree/main)
* ++

## Data

### Potsdam

Potsdam patch split training:
> python tools/potsdam_patch_split.py --img-dir "data/potsdam/train_images" --mask-dir "data/potsdam/train_masks" --output-img-dir "data/potsdam/train/images" --output-mask-dir "data/potsdam/train/masks" --mode "train" --split-size 512 --stride 512 --rgb-image

Potsdam patch split test:
> python tools/potsdam_patch_split.py --img-dir "data/potsdam/test_images" --mask-dir "data/potsdam/test_masks_eroded" --output-img-dir "data/potsdam/test/images" --output-mask-dir "data/potsdam/test/masks" --mode "val" --split-size 512 --stride 512 --eroded --rgb-image

Potsdam patch split colored masks for future visualization:
> python tools/potsdam_patch_split.py --img-dir "data/potsdam/test_images" --mask-dir "data/potsdam/test_masks" --output-img-dir "data/potsdam/test/images" --output-mask-dir "data/potsdam/test/masks_rgb" --mode "val" --split-size 512 --stride 512 --gt --rgb-image
 
### Inria



## Experiments

### Potsdam

2_10, 2_11, 2_12
3_10, 3_11, 3_12
4_10, 4_11, 4_12
5_10, 5_11, 5_12
6_7, 6_8, 6_9, 6_10, 6_11, 6_12
7_7, 7_8, 7_9, 7_10, 7_11, 7_12

Total number of images: 24

Split 1: [2_10, 3_10, 4_10] (3)
Split 2: [2_10, 3_10, 4_10, 5_10, 6_7, 7_7] (6)
Split 3: [2_10, 3_10, 4_10, 5_10, 6_7, 7_7, 2_11, 3_11, 4_11, 5_11, 6_8, 7_8] (12)
Split 4: [:] (24)

### Inria

Different split sizes for the training set:

Total number of images per city: 32 (4 is kept for validation)
Total number of cities: 5 (austin, chicago, kitsap, tyrol-w, vienna)

Split 1: [1, .., 4] (1/8 of the total data)
Split 2: [1, .., 8] (1/4 of the total data)
Split 3: [1, .., 16] (1/2 of the total data)
Split 4: [1, .., 32] (1/1 of the total data)

All splits will use all the test data for the Inria dataset

Experimental setup:

| Parameter | Value        |
|-|--------------|
| Learning rate | 0.0001 (1e-4) |
| Epochs | 50           |
| Batch size | 8            |
| Image size | 512          |
| Optimizer | AdamW        |
| Scheduler | CosineAnnealingWarmRestarts |
| T_0 | 100 |
| T_mult | 2 |
| eta_min | 1e-6 |

Models to test:
* Swin (DC Decoder)
* Hiera (DC Decoder)
* ?(ViT (DC Decoder))

