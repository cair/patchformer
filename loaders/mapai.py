import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from datasets import load_dataset

from pathlib import Path

from .transform import TrainingTransform, ValidationTransform

NEW_IMAGE_SIZE = [512, 512] # (width, height) 4:3
NUM_CLASSES = 2

class MapAI(Dataset):
    def __init__(self, root: str, type: str, percentage: float = 1.0, image_size: int = 384, image_suffix: str = "tif", mask_suffix: str = "png", transform=None):
        self.root = root
        self.dir = Path(root, type)
        
        self.dataset = load_dataset("sjyhne/mapai_dataset", split=type)
        
        exit("")
        
        self.image_size = image_size
        
        self.images = sorted(Path(self.dir, "images").glob(f"*.{image_suffix}"))
        self.masks = sorted(Path(self.dir, "masks").glob(f"*.{mask_suffix}"))
        
        if percentage != 1.0:
            self.images = self.images[:int(len(self.images) * percentage)]
            self.masks = self.masks[:int(len(self.masks) * percentage)]
        
        assert len(self.images) == len(self.masks), "Number of images and masks must be equal"
        
        self.transform = TrainingTransform([self.image_size, self.image_size], NUM_CLASSES) if type == "train" else ValidationTransform([self.image_size, self.image_size], NUM_CLASSES)
        
        self.num_classes = NUM_CLASSES

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        imagepath = self.images[idx]
        maskpath = self.masks[idx]

        image = Image.open(imagepath)
        mask = Image.open(maskpath)
        
        if image.mode in ["L", "P", "1"]:
            image = image.convert("RGB")
        
        image, mask = self.transform(image, mask)
        
        print(mask.unique())
        
        return image, mask


if __name__ == "__main__":
    
    train_dataset = MapAI(root="data/coco_stuff", type="train", percentage=0.01)
    val_dataset = MapAI(root="data/coco_stuff", type="val", percentage=0.01)
    
    print("train:", len(train_dataset))
    print("val:", len(val_dataset))
    
    for i in range(20):
    
      image, mask = train_dataset[i]
      image, mask = val_dataset[i]
      print(mask.unique())
    