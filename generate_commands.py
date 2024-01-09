
import argparse
import os

parser = argparse.ArgumentParser(description='Generate commands')
parser.add_argument('-f', "--file", type=str, help='Output file path')
args = parser.parse_args()

EPOCHS = 50
LR = 1e-4

datasets = ["ade20k", "potsdam"]
model_sizes = ["base", "small", "tiny"]
models = ["vit"]
cls_types = ["conv1x1", "conv3x3", "mlp"]
seeds = [12, 25, 42]

if os.path.exists(args.file):
    os.remove(args.file)

with open(args.file, 'w') as f:
    for dataset in datasets:
        for model in models:
            for model_size in model_sizes:
                for seed in seeds:

                    if model_size == "tiny" or model_size == "small":
                        img_size = 224
                    else:
                        if model == "hiera" or model == "vit" or model == "hieradc":
                            img_size = 224
                        else:
                            img_size = 384


                    patch_learning_command = f"python main.py -pl True -ms {model_size} -d {dataset} -e {EPOCHS} -s {seed} -lr {LR} -m {model} -n patchformer2 -wb True -g patchformer2 -p 1.0 -is {img_size}"
                    command = f"python main.py -ms {model_size} -d {dataset} -e {EPOCHS} -s {seed} -lr {LR} -m {model} -n baseline2 -wb True -g baseline2 -p 1.0 -is {img_size}"

                    f.write(patch_learning_command + '\n')
                    f.write(command + '\n')

