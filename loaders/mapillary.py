import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from pathlib import Path

from .transform import TrainingTransform, ValidationTransform

NEW_IMAGE_SIZE = [384, 384] # (width, height) 4:3
NEW_MASK_SIZE = [384, 384] # (width, height) 4:3
NUM_CLASSES = 124

class Mapillary(Dataset):
    def __init__(self, root: str, type: str, percentage: float = 1.0, image_suffix: str = "png", mask_suffix: str = "png", transform=None):
        self.root = root
        self.dir = Path(root, type)
        
        self.images = sorted(Path(self.dir, "images").glob(f"*.{image_suffix}"))
        self.masks = sorted(Path(self.dir, "masks").glob(f"*.{mask_suffix}"))
        
        if percentage != 1.0:
            self.images = self.images[:int(len(self.images) * percentage)]
            self.masks = self.masks[:int(len(self.masks) * percentage)]
        
        assert len(self.images) == len(self.masks), f"Number of images and masks must be equal, {len(self.images)} != {len(self.masks)}"

        self.transform = TrainingTransform(NEW_IMAGE_SIZE, NUM_CLASSES) if type == "train" else ValidationTransform(NEW_IMAGE_SIZE, NUM_CLASSES)
        
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
        
        mask[mask == 255] = self.num_classes
        
        return image, mask

if __name__ == "__main__":
    
    train_dataset = Mapillary(root="data/coco_stuff", type="train", percentage=0.01)
    val_dataset = Mapillary(root="data/coco_stuff", type="val", percentage=0.01)
    
    print("train:", len(train_dataset))
    print("val:", len(val_dataset))
    
    for i in range(20):
    
      image, mask = train_dataset[i]
      image, mask = val_dataset[i]
      print(image.shape, mask.shape)
      print(mask.unique())
    