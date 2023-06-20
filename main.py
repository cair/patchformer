import argparse
from lightning_models.dcswin_lightning import DCSwinLightning
from utils.model_utils import get_model, get_patch_classifier
from loaders.dataset import get_datasets
from loaders.loader import get_loader
import lightning as L

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-gpu", "--gpu", type=int, required=True)
    parser.add_argument("-b", "--binary", type=bool, default=False)
    parser.add_argument("-pl", "--patch_learning", type=bool, default=False)

    return parser.parse_args()


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

    seed_everything(42)

    train_dataset, val_dataset = get_datasets(args.dataset, 256)
    train_loader, val_loader = get_loader(train_dataset, 8, True, 4), get_loader(val_dataset, 8, False, 4)

    num_classes = train_dataset.num_classes

    model = get_model("dcswin", "tiny", num_classes)

    if args.patch_learning:
        patch_classifiers = get_patch_classifier(model.encoder_channels, num_classes, args.binary)
    else:
        patch_classifiers = None

    lm = DCSwinLightning(model, patch_classifiers, train_loader, val_loader, patch_learning=args.patch_learning, binary=args.binary)

    trainer = L.Trainer(max_epochs=10, accelerator="gpu", devices=[args.gpu])

    trainer.fit(model=lm)




if __name__ == "__main__":
    main()