#!/bin/bash

dataset="cityscapes"
img_size=384
model="swindc"

# ViT PatchFormer Seed 12 COCO
python main.py -pl True -gpu 0 -d ${dataset} -e 20 -s 12 -lr 1e-4 -m ${model} -n patchformer -wb True -g patchformer -p 1.0 -is ${img_size} &

# ViT PatchFormer Seed 25 COCO 
python main.py -pl True -gpu 1 -d ${dataset} -e 20 -s 25 -lr 1e-4 -m ${model} -n patchformer -wb True -g patchformer -p 1.0 -is ${img_size} &

# ViT PatchFormer Seed 42 COCO
python main.py -pl True -gpu 2 -d ${dataset} -e 20 -s 42 -lr 1e-4 -m ${model} -n patchformer -wb True -g patchformer -p 1.0 -is ${img_size} &

# ViT Baseline Seed 12 COCO
python main.py -gpu 3 -d ${dataset} -e 20 -s 12 -lr 1e-4 -m ${model} -n baseline -wb True -g baseline -p 1.0 -is ${img_size} &

# ViT Baseline Seed 25 COCO 
python main.py -gpu 4 -d ${dataset} -e 20 -s 25 -lr 1e-4 -m ${model} -n baseline -wb True -g baseline -p 1.0 -is ${img_size} & 

# ViT Baseline Seed 42 COCO
python main.py -gpu 5 -d ${dataset} -e 20 -s 42 -lr 1e-4 -m ${model} -n baseline -wb True -g baseline -p 1.0 -is ${img_size} &

dataset="ade20k"

# ViT PatchFormer Seed 12 COCO
python main.py -pl True -gpu 6 -d ${dataset} -e 20 -s 12 -lr 1e-4 -m ${model} -n patchformer -wb True -g patchformer -p 1.0 -is ${img_size} &
# 
# # ViT PatchFormer Seed 25 COCO 
python main.py -pl True -gpu 7 -d ${dataset} -e 20 -s 25 -lr 1e-4 -m ${model} -n patchformer -wb True -g patchformer -p 1.0 -is ${img_size} &
# 
# # ViT PatchFormer Seed 42 COCO
python main.py -pl True -gpu 8 -d ${dataset} -e 20 -s 42 -lr 1e-4 -m ${model} -n patchformer -wb True -g patchformer -p 1.0 -is ${img_size} &
# 
# # ViT Baseline Seed 12 COCO
python main.py -gpu 9 -d ${dataset} -e 20 -s 12 -lr 1e-4 -m ${model} -n baseline -wb True -g baseline -p 1.0 -is ${img_size} &
# 
# # ViT Baseline Seed 25 COCO 
python main.py -gpu 10 -d ${dataset} -e 20 -s 25 -lr 1e-4 -m ${model} -n baseline -wb True -g baseline -p 1.0 -is ${img_size} & 
# 
# # ViT Baseline Seed 42 COCO
python main.py -gpu 11 -d ${dataset} -e 20 -s 42 -lr 1e-4 -m ${model} -n baseline -wb True -g baseline -p 1.0 -is ${img_size} &

dataset="potsdam"

# # ViT PatchFormer Seed 12 COCO
#python main.py -gpu 3 -d ${dataset} -e 20 -s 12 -lr 1e-4 -m hieradc -n baseline -wb True -g baseline -p 1.0 -is 224 &
# 
# # ViT PatchFormer Seed 25 COCO 
#python main.py -gpu 4 -d ${dataset} -e 20 -s 25 -lr 1e-4 -m hieradc -n baseline -wb True -g baseline -p 1.0 -is 224 &

# ViT PatchFormer Seed 42 COCO
#python main.py -gpu 5 -d ${dataset} -e 20 -s 42 -lr 1e-4 -m hieradc -n baseline -wb True -g baseline -p 1.0 -is 224 &

# ViT PatchFormer Seed 42 COCO
python main.py -pl True -gpu 12 -d ${dataset} -e 20 -s 12 -lr 1e-4 -m ${model} -n patchformer -wb True -g patchformer -p 1.0 -is ${img_size} &

# # ViT PatchFormer Seed 25 COCO 
python main.py -pl True -gpu 13 -d ${dataset} -e 20 -s 25 -lr 1e-4 -m ${model} -n patchformer -wb True -g patchformer -p 1.0 -is ${img_size} &

# ViT PatchFormer Seed 42 COCO
python main.py -pl True -gpu 14 -d ${dataset} -e 20 -s 42 -lr 1e-4 -m ${model} -n patchformer -wb True -g patchformer -p 1.0 -is ${img_size} &


#python main.py -gpu 12 -d ${dataset} -e 20 -s 12 -lr 1e-4 -m swindc -n baseline -wb True -g baseline -p 1.0 -is 384 &
# 
# # ViT PatchFormer Seed 25 COCO 
#python main.py -gpu 13 -d ${dataset} -e 20 -s 25 -lr 1e-4 -m swindc -n baseline -wb True -g baseline -p 1.0 -is 384 &

# ViT PatchFormer Seed 42 COCO
#python main.py -gpu 14 -d ${dataset} -e 20 -s 42 -lr 1e-4 -m swindc -n baseline -wb True -g baseline -p 1.0 -is 384 &