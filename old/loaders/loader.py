from torch.utils.data import DataLoader
from loaders.dataset import get_datasets

def get_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":
    train, val = get_datasets("potsdam", 256)

    train_loader = get_loader(train, 4, True, 4)
    val_loader = get_loader(val, 4, False, 4)

    for batch in train_loader:
        img = batch["img"]
        mask = batch["mask"]
        print(img.shape)
        print(mask.shape)
        exit(":")