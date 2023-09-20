from torchvision import transforms
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

def resize_and_pad(img, target_size, fill, interpolation_type=Image.BILINEAR):
    """
    Resize and pad `img` to `target_size` while maintaining aspect ratio.

    Args:
    img (PIL.Image.Image): The input image.
    target_size (tuple): The target width and height in the format (width, height).

    Returns:
    PIL.Image.Image: The resized and padded image.
    """
    # Calculate the aspect ratio of the image
    img_width, img_height = img.size
    img_aspect_ratio = img_width / img_height
    
    # Calculate the aspect ratio of the target size
    target_width, target_height = target_size
    target_aspect_ratio = target_width / target_height
    
    # Calculate the scaling factor and new size
    if img_aspect_ratio > target_aspect_ratio:
        scale_factor = target_width / img_width
        new_size = (target_width, int(img_height * scale_factor))
    else:
        scale_factor = target_height / img_height
        new_size = (int(img_width * scale_factor), target_height)
    
    # Create a transform for resizing the image
    resize_transform = transforms.Resize(new_size, interpolation=interpolation_type)
    
    # Resize the image
    img_resized = resize_transform(img)
    
    # Calculate padding
    pad_width = target_width - img_resized.width
    pad_height = target_height - img_resized.height
    padding = (pad_width // 2, pad_height // 2, pad_width - (pad_width // 2), pad_height - (pad_height // 2))
    
    # Create a transform for padding the image
    pad_transform = transforms.Pad(padding, fill=0, padding_mode='constant')
    
    # Pad the image
    img_padded = pad_transform(img_resized)
    
    return img_padded

class TrainingTransform:
    def __init__(self, image_size, num_classes):
        self.image_size = image_size
        self.num_classes = num_classes
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def __call__(self, img, mask):
        # Resize
        img = resize_and_pad(img, self.image_size, fill=self.num_classes, interpolation_type=Image.BILINEAR)
        mask = resize_and_pad(mask, self.image_size, fill=self.num_classes, interpolation_type=Image.NEAREST)

        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # Random rotation
        if torch.rand(1).item() > 0.80:
            angle = torch.rand(1).item() * 360
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)

        # To tensor
        img = TF.to_tensor(img)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        
        # Normalize image
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        img = normalize(img)

        return img, mask

class ValidationTransform:
    def __init__(self, image_size, num_classes):
        self.image_size = image_size
        self.num_classes = num_classes
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img, mask):
        # Resize
        img = resize_and_pad(img, self.image_size, fill=self.num_classes, interpolation_type=Image.BILINEAR)
        mask = resize_and_pad(mask, self.image_size, fill=self.num_classes, interpolation_type=Image.NEAREST)

        # To tensor
        img = TF.to_tensor(img)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        
        # Normalize image
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        img = normalize(img)

        return img, mask