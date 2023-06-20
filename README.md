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