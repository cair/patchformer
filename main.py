import torch
import lightning as L

from loaders.loader import get_dataloader
from lightning_models.vitupernet import ViTUperNet
from lightning_models.swinupernet import SwinUperNet
from lightning_models.swindc import SwinDC
from lightning_models.hieradc import HieraDC
from lightning_models.hieraupernet import HieraUperNet
from utils import get_callbacks, get_logger

import argparse

def seed(seed):
    import random
    import numpy as np
    import os

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-pl", "--patch_learning", type=bool, default=False)
    parser.add_argument("-gpu", "--gpu", type=int, nargs="+", default=0)
    parser.add_argument("-d", "--dataset", type=str, default="coco")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-s", "--seed", type=int, default=12, help="Seed for reproducibility")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-ms", "--model_size", type=str, default="base", help="Model size")
    parser.add_argument("-is", "--image_size", type=int, default=384, help="Image size")
    parser.add_argument("-m", "--model", type=str, default="swin", help="Model")
    parser.add_argument("-n", "--name", type=str, default="", help="Name for wandb")
    parser.add_argument("-wb", "--wandb", type=bool, default=False, help="Use wandb")
    parser.add_argument("-g", "--group", type=str, default=None, help="Group for wandb")
    parser.add_argument("-p", "--percentage", type=float, default=0.01, help="Percentage of dataset to use")
    parser.add_argument("-ct", "--cls_type", type=str, default="conv1x1", help="Type of classifier to use")
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="Batch size")
    
    
    return parser.parse_args()
    
def main():
    
    args = parse_args()
    
    seed(args.seed)
    
    train_loader, num_classes = get_dataloader(dataset_name=args.dataset, dataset_type="train", batch_size=args.batch_size, percentage=args.percentage, image_size=args.image_size)
    val_loader, num_classes = get_dataloader(dataset_name=args.dataset, dataset_type="val", batch_size=1, percentage=args.percentage, image_size=args.image_size)

    if args.model == "swin":
        lightning_model = SwinUperNet(num_classes=num_classes,
                                    learning_rate=args.learning_rate,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    patch_learning=args.patch_learning,
                                    model_size=args.model_size,
                                    cls_type=args.cls_type)
    elif args.model == "vit":
        lightning_model = ViTUperNet(num_classes=num_classes,
                                    learning_rate=args.learning_rate,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    patch_learning=args.patch_learning,
                                    model_size=args.model_size,
                                    cls_type=args.cls_type)
    elif args.model == "hiera":
        lightning_model = HieraUperNet(num_classes=num_classes,
                                      learning_rate=args.learning_rate,
                                      train_loader=train_loader,
                                      val_loader=val_loader,
                                      patch_learning=args.patch_learning,
                                      model_size=args.model_size,
                                      cls_type=args.cls_type)
    elif args.model == "swindc":
        lightning_model = SwinDC(num_classes=num_classes,
                                      learning_rate=args.learning_rate,
                                      train_loader=train_loader,
                                      val_loader=val_loader,
                                      patch_learning=args.patch_learning,
                                      model_size=args.model_size,
                                      cls_type=args.cls_type)
    elif args.model == "hieradc":
        lightning_model = HieraDC(num_classes=num_classes,
                                      learning_rate=args.learning_rate,
                                      train_loader=train_loader,
                                      val_loader=val_loader,
                                      patch_learning=args.patch_learning,
                                      model_size=args.model_size,
                                      cls_type=args.cls_type)
    
    project_name = f"{args.model}_{args.dataset}_{args.image_size}_{args.model_size}"
    
    loggers = get_logger(args.wandb, project_name, args.name + f"_{args.learning_rate}_{args.percentage}_seed{args.seed}_{args.cls_type}", args.group, lightning_model)
    callbacks = get_callbacks(project_name, args.name)
    
    trainer = L.Trainer(max_epochs=args.epochs, 
                        accelerator="gpu", 
                        devices=args.gpu, 
                        logger=loggers, 
                        callbacks=callbacks,
                        strategy="ddp_find_unused_parameters_true",)

    trainer.fit(model=lightning_model)


if __name__ == "__main__":
    main()