
import argparse
import os

parser = argparse.ArgumentParser(description='Generate commands')
parser.add_argument('-f', "--file", type=str, help='Output file path')
args = parser.parse_args()

EPOCHS = 50
LR = 1e-4

datasets = ["ade20k", "cityscapes", "potsdam"]
model_size = "base"
image_sizes = [224, 384]
models = ["swin", "hiera", "vit"]
seeds = [12, 25, 42]



if os.path.exists(args.file):
    os.remove(args.file)

with open(args.file, 'w') as f:
    for dataset in datasets:
        for model in models:
            for seed in seeds:

                if model_size == "tiny":
                    if model == "vit":
                        img_size = 384
                    else:
                        img_size = 224
                else:
                    if model == "hiera":
                        img_size = 224
                    else:
                        img_size = 384


                patch_learning_command = f"python main.py -pl True -ms {model_size} -d {dataset} -e {EPOCHS} -s {seed} -lr {LR} -m {model} -n patchformer -wb True -g patchformer -p 1.0 -is {img_size}"
                command = f"python main.py -ms {model_size} -d {dataset} -e {EPOCHS} -s {seed} -lr {LR} -m {model} -n baseline -wb True -g baseline -p 1.0 -is {img_size}"

                f.write(patch_learning_command + '\n')
                f.write(command + '\n')
