import torch
from torch.utils.data import DataLoader

from loaders.coco_stuff import COCOStuff

def get_dataloader(dataset_name: str, 
                   dataset_type: str, 
                   batch_size: int, 
                   percentage: float, 
                   num_workers: int = 1) -> DataLoader:
    
    shuffle = dataset_type == "train"
    
    if dataset_name == "coco":
        dataset = COCOStuff(root="data/coco_stuff", type="train", percentage=percentage)
    else:
        return NotImplementedError("Dataset not implemented")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader

if __name__ == "__main__":
    
    dataloader = get_dataloader(dataset_name="coco", dataset_type="train", batch_size=8, percentage=0.01)
    
    for batch in dataloader:
        image, mask = batch
        print(image.shape)
        print(mask.shape)
        exit("")