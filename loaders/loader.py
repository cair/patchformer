import torch
from torch.utils.data import DataLoader

from loaders.coco_stuff import COCOStuff
from loaders.ade20k import ADE20K
from loaders.mapillary import Cityscapes
from loaders.potsdam import Potsdam

def get_dataloader(dataset_name: str, 
                   dataset_type: str, 
                   batch_size: int, 
                   percentage: float,
                   image_size: int = 384, 
                   num_workers: int = 4) -> DataLoader:
    
    shuffle = dataset_type == "train"
    
    if dataset_name == "coco":
        dataset = COCOStuff(root="data/coco_stuff", type=dataset_type, percentage=percentage, image_size=image_size)
    elif dataset_name == "ade20k":
        dataset = ADE20K(root="data/ADEChallengeData2016", type=dataset_type, percentage=percentage, image_size=image_size)
    elif dataset_name == "cityscapes":
        dataset = Cityscapes(root="data/cityscapes", type=dataset_type, percentage=percentage, image_size=image_size)
    elif dataset_name == "potsdam":
        dataset = Potsdam(root="data/potsdam", type=dataset_type, percentage=percentage, image_size=image_size)
    else:
        return NotImplementedError("Dataset not implemented")
    
    num_classes = dataset.num_classes
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader, num_classes

if __name__ == "__main__":
    
    dataloader = get_dataloader(dataset_name="coco", dataset_type="train", batch_size=8, percentage=0.01)
    
    for batch in dataloader:
        image, mask = batch
        print(image.shape)
        print(mask.shape)
        exit("")