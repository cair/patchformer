import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from pathlib import Path

from .transform import TrainingTransform, ValidationTransform

NEW_IMAGE_SIZE = [384, 384] # (width, height) 4:3
NEW_MASK_SIZE = [384, 384] # (width, height) 4:3
NUM_CLASSES = 19

cityscapes_label_mapping = {
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,
    8: 1,
    9: 255,
    10: 255,
    11: 2,
    12: 3,
    13: 4,
    14: 255,
    15: 255,
    16: 255,
    17: 5,
    18: 255,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 255,
    30: 255,
    31: 16,
    32: 17,
    33: 18,
    -1: 255
}

def map_labels(tensor, mapping_dict):
    # Create an output tensor filled with 255 (assuming 255 is the most common "ignore" label)
    output = torch.full_like(tensor, 255)
    for src, tgt in mapping_dict.items():
        output[tensor == src] = tgt
    return output

class Cityscapes(Dataset):
    def __init__(self, root: str, type: str, percentage: float = 1.0, image_size: int = 384, image_suffix: str = "png", mask_suffix: str = "png", transform=None):
        self.root = root
        self.dir = Path(root, type)
        
        self.images = sorted(Path(self.dir, "images").glob(f"*.{image_suffix}"))
        self.masks = sorted(Path(self.dir, "masks").glob(f"*.{mask_suffix}"))
        
        self.image_size = [image_size, image_size]
        
        if percentage != 1.0:
            self.images = self.images[:int(len(self.images) * percentage)]
            self.masks = self.masks[:int(len(self.masks) * percentage)]
        
        assert len(self.images) == len(self.masks), f"Number of images and masks must be equal, {len(self.images)} != {len(self.masks)}"

        self.transform = TrainingTransform(self.image_size, NUM_CLASSES) if type == "train" else ValidationTransform(self.image_size, NUM_CLASSES)
        
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
        
        mask = map_labels(mask, cityscapes_label_mapping)
        
        mask[mask == 255] = self.num_classes
        
        return image, mask

if __name__ == "__main__":
    
    
    
    exit("")
        
    print("train:", len(train_dataset))
    print("val:", len(val_dataset))
    
    for i in range(20):
    
      image, mask = train_dataset[i]
      image, mask = val_dataset[i]
      print(image.shape, mask.shape)
      print(mask.unique())
    