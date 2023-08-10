import argparse
from datetime import datetime

from lightning_models.dcswin_lightning import DCSwinLightning
from lightning_models.hiera_lightning import HieraLightning
from lightning_models.vit_lightning import ViTLightning
from utils.model_utils import get_model, get_patch_classifier
from loaders.dataset import get_datasets
from loaders.loader import get_loader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-gpu", "--gpu", type=int, required=True)
    parser.add_argument("-b", "--binary", type=bool, default=False)
    parser.add_argument("-pl", "--patch_learning", type=bool, default=False)
    parser.add_argument("-wb", "--wandb", type=bool, default=False)
    parser.add_argument("-ms", "--model_size", type=str, default="tiny")
    parser.add_argument("-is", "--image_size", type=int, default=256)
    parser.add_argument("-bs", "--batch_size", type=int, default=8)
    parser.add_argument("-n", "--name", type=str, default="")
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-dual", "--dual", type=bool, default=False, help="Use dual attention")
    parser.add_argument("-split", "--split", type=int, default=None, help="Use a split of the dataset")
    parser.add_argument("-seed", "--seed", type=int, default=12, help="Seed for reproducibility")

    return parser.parse_args()


def get_logger(wandb: bool, project: str, name: str, model: L.LightningModule):

    # Create default logger (CSVLogger)
    logger = CSVLogger("lightning_logs", name)
    loggers = [logger]

    if wandb:
        wandblogger = WandbLogger(name=name, project=project)
        wandblogger.watch(model, log_graph=False)
        # wandblogger.experiment.config.update(config)
        loggers.append(wandblogger)

    return loggers


def get_callbacks(project, name):

    config = {}

    config["save_top_k"] = 1
    config["monitor"] = "val_loss"
    config["mode"] = "max"

    config["dirpath"] = "lightning_logs" + "/" + project
    config["filename"] = name + datetime.now().strftime("%d/%m/%Y %H:%M:%S") \
        .replace("/", "").replace(" ", "_").replace(":", "")

    callbacks = []

    callbacks.append(ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        # mode=config["mode"],
        filename="{epoch:02d}-{val_loss:.2f}",
        dirpath=config["dirpath"]
    ))

    return callbacks


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()

    if args.name == "":
        args.name = input("Please enter a name for this run: ")

    seed_everything(args.seed) # 12 used, 5 used, 46

    train_dataset, val_dataset = get_datasets(args.dataset, args.image_size, args.split)
    train_loader, val_loader = get_loader(train_dataset, args.batch_size, True, 4), get_loader(val_dataset, 1, False, 4)
    
    num_classes = train_dataset.num_classes

    model = get_model(args.model, args.model_size, num_classes, args.binary, input_size=(args.image_size, args.image_size))

    if args.model == "dcswin":
        lm = DCSwinLightning(model, train_loader, val_loader, num_classes, learning_rate=args.learning_rate, patch_learning=args.patch_learning, dual=args.dual)
    elif args.model == "hiera":
        lm = HieraLightning(model, train_loader, val_loader, num_classes, learning_rate=args.learning_rate, patch_learning=args.patch_learning, dual=args.dual)
    elif args.model == "dino":
        lm = ViTLightning(model, train_loader, val_loader, num_classes, learning_rate=args.learning_rate, patch_learning=args.patch_learning, dual=args.dual)
    else:
        raise NotImplementedError


    project_name = f"{args.model}_{args.dataset}_{args.image_size}_{args.model_size}"

    loggers = get_logger(args.wandb, project_name, args.name + f"_{args.learning_rate}_split{args.split}_seed{args.seed}", lm)
    callbacks = get_callbacks(project_name, args.name)

    trainer = L.Trainer(max_epochs=args.epochs, accelerator="gpu", devices=[args.gpu], logger=loggers, callbacks=callbacks)

    trainer.fit(model=lm)




if __name__ == "__main__":

    main()
