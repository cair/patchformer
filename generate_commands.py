
import argparse
import os

parser = argparse.ArgumentParser(description='Generate commands')
parser.add_argument('-f', "--file", type=str, help='Output file path')
args = parser.parse_args()

EPOCHS = 5
LR = 2e-4

datasets = ["ade20k"]
model_sizes = ["tiny"]
models = ["vit"]
cls_types = ["conv1x1", "conv3x3", "mlp"]
seeds = [12]

dp = 0.1

if os.path.exists(args.file):
    os.remove(args.file)

with open(args.file, 'w') as f:
    for dataset in datasets:
        for model in models:
            for model_size in model_sizes:
                for ct in cls_types:
                    for seed in seeds:

                        if model_size == "tiny" or model_size == "small":
                            img_size = 224
                        else:
                            if model == "hiera" or model == "vit" or model == "hieradc":
                                img_size = 224
                            else:
                                img_size = 384


                        patch_learning_command = f"python main.py -pl True -ms {model_size} -d {dataset} -e {EPOCHS} -s {seed} -lr {LR} -m {model} -n patchformer -wb True -g patchformers -p {dp} -is {img_size} -ct {ct}"
                        command = f"python main.py -ms {model_size} -d {dataset} -e {EPOCHS} -s {seed} -lr {LR} -m {model} -n baseline -wb True -g baseline -p {dp} -is {img_size} -ct {ct}"

                        f.write(patch_learning_command + '\n')
                        f.write(command + '\n')

